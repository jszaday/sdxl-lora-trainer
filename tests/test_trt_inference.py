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


def test_unet_engine_path_names_resolution_and_precision(tmp_path):
    spec = parse_resolution("1536x640")
    assert unet_engine_path(tmp_path, spec, "fp16") == tmp_path / "unet_fp16_1536x640.plan"


def test_file_sha256_hashes_local_file(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"abc")

    assert file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223" "b00361a396177a9cb410ff61f20015ad"
    )


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
