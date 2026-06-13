"""Tests for TensorRT-ready frozen SDXL inference helpers."""

import pytest
import torch
import torch.nn as nn

from lora_trainer.model import LoRALayer, merge_lora_layers
from lora_trainer.trt.backends import unet_engine_path
from lora_trainer.trt.cache import (
    build_engine_cache_key,
    file_sha256,
    resolve_engine_artifacts,
    write_metadata,
)
from lora_trainer.trt.config import (
    SDXL_RESOLUTIONS,
    get_resolution,
    infer_resolution_from_latents,
    parse_resolution,
    validate_latents_shape,
)
from lora_trainer.trt.inference import (
    _match_conditioning_sequence_lengths,
    make_initial_latents,
    sample_frozen_sdxl,
)
from lora_trainer.trt.latents import load_latents, save_latents


def test_sdxl_resolution_latent_shapes():
    assert SDXL_RESOLUTIONS["1024x1024"].latent_shape == (128, 128)
    assert SDXL_RESOLUTIONS["1216x832"].latent_shape == (104, 152)
    assert SDXL_RESOLUTIONS["640x1536"].latent_shape == (192, 80)


def test_parse_resolution_rejects_arbitrary_shape():
    with pytest.raises(ValueError, match="Unsupported SDXL resolution"):
        parse_resolution("1024x768")


def test_infer_resolution_from_latents():
    latents = torch.zeros(1, 4, 104, 152)
    spec = infer_resolution_from_latents(latents)
    assert spec.name == "1216x832"


def test_validate_latents_shape_checks_batch_and_spatial_dims():
    spec = get_resolution(896, 1152)
    validate_latents_shape(torch.zeros(1, 4, 144, 112), spec)

    with pytest.raises(ValueError, match="Expected latents shape"):
        validate_latents_shape(torch.zeros(2, 4, 144, 112), spec)


def test_latents_round_trip_safetensors(tmp_path):
    path = tmp_path / "latents.safetensors"
    latents = torch.randn(1, 4, 128, 128)
    save_latents(path, latents)

    loaded = load_latents(path, device="cpu", dtype=torch.float32)
    assert torch.allclose(loaded, latents)


def test_initial_latents_match_comfy_cpu_seed_path():
    spec = parse_resolution("1024x1024")
    latents = make_initial_latents(
        spec,
        batch_size=1,
        device="cpu",
        dtype=torch.float16,
        seed=123,
    )
    expected = torch.randn(
        (1, 4, 128, 128),
        generator=torch.Generator(device="cpu").manual_seed(123),
        device="cpu",
        dtype=torch.float32,
    ).to(dtype=torch.float16)

    assert torch.equal(latents, expected)


def test_partial_denoise_initial_latents_use_sliced_start_sigma(monkeypatch):
    spec = parse_resolution("1024x1024")
    captured: dict[str, torch.Tensor] = {}

    def fake_initial_latents(*args, **kwargs):
        return torch.ones(1, 4, 128, 128)

    class CaptureBackend:
        def __call__(self, sample, timestep, encoder_hidden_states, *, added_cond_kwargs):
            if "first_sample" not in captured:
                captured["first_sample"] = sample.detach().clone()
            return torch.zeros_like(sample)

    monkeypatch.setattr("lora_trainer.trt.inference.make_initial_latents", fake_initial_latents)

    prompt_embeds = torch.zeros(1, 77, 2048)
    pooled_embeds = torch.zeros(1, 1280)
    sample_frozen_sdxl(
        CaptureBackend(),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_embeds,
        pooled_negative_prompt_embeds=pooled_embeds,
        resolution=spec,
        sampler="euler",
        scheduler_name="normal",
        num_inference_steps=5,
        guidance_scale=1.0,
        device="cpu",
        dtype=torch.float32,
        seed=123,
        denoise=0.5,
        progress=False,
    )

    first_sample = captured["first_sample"]
    # The first UNet input is scheduler-scaled. If the raw latent starts at the
    # sliced sigma, scaling by sqrt(sigma^2 + 1) keeps the value below 1.0. The
    # old full-schedule sigma produced values near the full max-noise level.
    assert 0.0 < float(first_sample.mean()) < 1.0


def test_partial_denoise_provided_latents_are_noised(monkeypatch):
    spec = parse_resolution("1024x1024")
    captured: dict[str, torch.Tensor] = {}

    class CaptureBackend:
        def __call__(self, sample, timestep, encoder_hidden_states, *, added_cond_kwargs):
            if "first_sample" not in captured:
                captured["first_sample"] = sample.detach().clone()
            return torch.zeros_like(sample)

    def fake_randn(*args, **kwargs):
        return torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr("lora_trainer.trt.inference.torch.randn", fake_randn)

    prompt_embeds = torch.zeros(1, 77, 2048)
    pooled_embeds = torch.zeros(1, 1280)
    sample_frozen_sdxl(
        CaptureBackend(),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_embeds,
        pooled_negative_prompt_embeds=pooled_embeds,
        resolution=spec,
        sampler="euler",
        scheduler_name="normal",
        num_inference_steps=5,
        guidance_scale=1.0,
        device="cpu",
        dtype=torch.float32,
        latents=torch.zeros(1, 4, 128, 128),
        seed=123,
        denoise=0.5,
        progress=False,
    )

    first_sample = captured["first_sample"]
    assert 0.0 < float(first_sample.mean()) < 1.0


def test_partial_denoise_euler_ancestral_uses_scheduler_timesteps(monkeypatch):
    spec = parse_resolution("1024x1024")
    captured: dict[str, torch.Tensor] = {}

    class CaptureBackend:
        def __call__(self, sample, timestep, encoder_hidden_states, *, added_cond_kwargs):
            if "first_timestep" not in captured:
                captured["first_timestep"] = timestep.detach().clone()
            return torch.zeros_like(sample)

    def fake_randn(*args, **kwargs):
        return torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr("lora_trainer.trt.inference.torch.randn", fake_randn)

    prompt_embeds = torch.zeros(1, 77, 2048)
    pooled_embeds = torch.zeros(1, 1280)
    sample_frozen_sdxl(
        CaptureBackend(),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_embeds,
        pooled_negative_prompt_embeds=pooled_embeds,
        resolution=spec,
        sampler="euler_ancestral",
        scheduler_name="normal",
        num_inference_steps=5,
        guidance_scale=1.0,
        device="cpu",
        dtype=torch.float32,
        latents=torch.zeros(1, 4, 128, 128),
        seed=123,
        denoise=0.7,
        progress=False,
    )

    assert captured["first_timestep"].numel() == 2


def test_full_denoise_provided_latents_are_noised(monkeypatch):
    spec = parse_resolution("1024x1024")
    captured: dict[str, torch.Tensor] = {}

    class CaptureBackend:
        def __call__(self, sample, timestep, encoder_hidden_states, *, added_cond_kwargs):
            if "first_sample" not in captured:
                captured["first_sample"] = sample.detach().clone()
            return torch.zeros_like(sample)

    def fake_randn(*args, **kwargs):
        return torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr("lora_trainer.trt.inference.torch.randn", fake_randn)

    prompt_embeds = torch.zeros(1, 77, 2048)
    pooled_embeds = torch.zeros(1, 1280)
    sample_frozen_sdxl(
        CaptureBackend(),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_embeds,
        pooled_negative_prompt_embeds=pooled_embeds,
        resolution=spec,
        sampler="euler",
        scheduler_name="normal",
        num_inference_steps=5,
        guidance_scale=1.0,
        device="cpu",
        dtype=torch.float32,
        latents=torch.zeros(1, 4, 128, 128),
        seed=123,
        denoise=1.0,
        progress=False,
    )

    # KSampler's max-denoise path scales noise by sqrt(1 + sigma^2); after
    # model-input preconditioning the UNet sees unit-variance noise, not zeros.
    assert float(captured["first_sample"].mean()) == pytest.approx(1.0)


def test_conditioning_sequence_lengths_are_padded_by_chunks():
    prompt_embeds = torch.ones(1, 154, 2048)
    negative_embeds = torch.zeros(1, 77, 2048)

    prompt_out, negative_out = _match_conditioning_sequence_lengths(
        prompt_embeds,
        negative_embeds,
    )

    assert prompt_out.shape == negative_out.shape == (1, 154, 2048)
    assert torch.equal(negative_out[:, :77], negative_embeds)
    assert torch.equal(negative_out[:, 77:], negative_embeds)


def test_unet_engine_path_names_resolution_and_precision(tmp_path):
    spec = parse_resolution("1536x640")
    assert unet_engine_path(tmp_path, spec, "fp16") == tmp_path / "unet_fp16_1536x640.plan"


def test_file_sha256_hashes_local_file(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"abc")

    assert file_sha256(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_engine_cache_key_changes_with_lora_and_resolution(tmp_path):
    checkpoint = tmp_path / "base.safetensors"
    lora = tmp_path / "style.safetensors"
    checkpoint.write_bytes(b"base")
    lora.write_bytes(b"lora")

    square = parse_resolution("1024x1024")
    wide = parse_resolution("1216x832")
    base_key = build_engine_cache_key(str(checkpoint), square, precision="fp16")
    lora_key = build_engine_cache_key(
        str(checkpoint),
        square,
        precision="fp16",
        lora_checkpoint=lora,
    )
    wide_key = build_engine_cache_key(str(checkpoint), wide, precision="fp16")

    assert base_key.checkpoint_sha256 == file_sha256(checkpoint)
    assert lora_key.lora_sha256 == file_sha256(lora)
    assert base_key.digest != lora_key.digest
    assert base_key.digest != wide_key.digest


def test_engine_artifact_paths_are_cache_keyed(tmp_path):
    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"base")
    key = build_engine_cache_key(
        str(checkpoint),
        parse_resolution("1024x1024"),
        precision="fp16",
    )

    artifacts = resolve_engine_artifacts(tmp_path / "engines", tmp_path / "onnx", key)
    assert key.digest in str(artifacts.engine_path)
    assert artifacts.engine_path.name == "model.plan"
    assert artifacts.onnx_path.name == "model.onnx"

    write_metadata(artifacts)
    assert artifacts.metadata_path.exists()


def test_merge_lora_linear_replaces_wrapper_and_preserves_output():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = LoRALayer(nn.Linear(4, 3), rank=2, alpha=2)

        def forward(self, x):
            return self.proj(x)

    model = TinyModel()
    model.eval()
    x = torch.randn(5, 4)
    before = model(x)

    merge_lora_layers(model)
    after = model(x)

    assert isinstance(model.proj, nn.Linear)
    assert torch.allclose(after, before, atol=1e-6)
