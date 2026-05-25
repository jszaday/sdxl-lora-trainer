"""Tests for validation sampling functionality."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from diffusers import DDIMScheduler

from lora_trainer.sampling import (
    PromptSpec,
    _comfy_sdxl_tokenize,
    encode_latents,
    encode_prompts_for_sampling,
    load_prompt_specs,
    sample_with_cfg,
)
from lora_trainer.schedulers import build_noise_scheduler


class DummyUNet(nn.Module):
    """Minimal UNet for testing sampling."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # Simple passthrough with noise reduction simulation
        output = self.conv(sample)
        return type("Output", (), {"sample": output})()


class DummyTextEncoder(nn.Module):
    """Minimal text encoder for testing."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Return dummy hidden states
        hidden_states = [
            torch.randn(batch_size, seq_len, self.hidden_size, device=input_ids.device)
            for _ in range(3)
        ]

        # Return dummy pooled output
        pooled_output = torch.randn(batch_size, self.hidden_size, device=input_ids.device)

        return type(
            "Output",
            (),
            {"hidden_states": hidden_states, "__getitem__": lambda self, idx: pooled_output},
        )()


class DummyTokenizer:
    """Minimal tokenizer for testing."""

    def __init__(self):
        self.model_max_length = 77

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        input_ids = torch.randint(0, 1000, (batch_size, self.model_max_length))
        return type("Output", (), {"input_ids": input_ids})()


class FixedTokenizer:
    model_max_length = 8

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, str):
            input_ids = [101, 11, 12, 102]
            return {"input_ids": input_ids}
        return type("Output", (), {"input_ids": torch.tensor([[101, 11, 12, 102]])})()


def test_comfy_sdxl_tokenize_clip_l_pads_with_end():
    tokens = _comfy_sdxl_tokenize(["test"], FixedTokenizer(), pad_with_end=True)

    assert tokens.tolist() == [[[101, 11, 12, 102, 102, 102, 102, 102]]]


def test_comfy_sdxl_tokenize_clip_g_pads_with_zero():
    tokens = _comfy_sdxl_tokenize(["test"], FixedTokenizer(), pad_with_end=False)

    assert tokens.tolist() == [[[101, 11, 12, 102, 0, 0, 0, 0]]]


def test_comfy_sdxl_tokenize_chunks_long_prompts():
    class LongTokenizer:
        model_max_length = 8

        def __call__(self, prompt, **kwargs):
            return {"input_ids": [101, *range(20, 32), 102]}

    tokens = _comfy_sdxl_tokenize(["long"], LongTokenizer(), pad_with_end=False)

    assert tokens.shape == (1, 2, 8)
    assert tokens[0, 0].tolist() == [101, 20, 21, 22, 23, 24, 25, 102]
    assert tokens[0, 1].tolist() == [101, 26, 27, 28, 29, 30, 31, 102]


def test_encode_prompts_for_sampling():
    """Test that prompt encoding produces correct shapes."""
    prompts = ["a photo of a cat", "a photo of a dog"]
    text_encoder_1 = DummyTextEncoder(768)
    text_encoder_2 = DummyTextEncoder(1280)
    tokenizer_1 = DummyTokenizer()
    tokenizer_2 = DummyTokenizer()

    prompt_embeds, pooled_embeds = encode_prompts_for_sampling(
        prompts, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, "cpu"
    )

    # SDXL combines embeddings from both encoders
    assert prompt_embeds.shape == (2, 77, 768 + 1280)
    assert pooled_embeds.shape == (2, 1280)


def test_sample_with_cfg_shape():
    """Test that CFG sampling produces correct output shape."""
    unet = DummyUNet()
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )

    batch_size = 2
    prompt_embeds = torch.randn(batch_size, 77, 2048)
    negative_prompt_embeds = torch.randn(batch_size, 77, 2048)
    pooled_prompt_embeds = torch.randn(batch_size, 1280)
    pooled_negative_prompt_embeds = torch.randn(batch_size, 1280)

    latents = sample_with_cfg(
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        pooled_negative_prompt_embeds=pooled_negative_prompt_embeds,
        num_inference_steps=5,  # Use few steps for testing
        guidance_scale=7.0,
        height=512,
        width=512,
        device="cpu",
        dtype=torch.float32,
    )

    # Check output shape
    assert latents.shape == (batch_size, 4, 512 // 8, 512 // 8)


def test_sample_with_cfg_euler_uses_ksampler_loop():
    """Euler validation sampling should share the Comfy-style loop."""
    unet = DummyUNet()
    scheduler = build_noise_scheduler("normal", num_inference_steps=3, sampler_name="euler")

    prompt_embeds = torch.randn(1, 77, 2048)
    negative_prompt_embeds = torch.randn(1, 77, 2048)
    pooled_prompt_embeds = torch.randn(1, 1280)
    pooled_negative_prompt_embeds = torch.randn(1, 1280)

    latents = sample_with_cfg(
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        pooled_negative_prompt_embeds=pooled_negative_prompt_embeds,
        num_inference_steps=3,
        guidance_scale=7.0,
        height=256,
        width=256,
        device="cpu",
        dtype=torch.float32,
        sampler_name="euler",
        seeds=[123],
    )

    assert latents.shape == (1, 4, 32, 32)


@pytest.mark.skip("Requires real VAE - slow test")
def test_decode_latents():
    """Test that latent decoding produces valid images."""
    # This would require loading a real VAE, which is slow
    # Skip for now, but keep as placeholder
    pass


def test_encode_latents_uses_posterior_mode():
    """ComfyUI VAE encode uses posterior mode, not a fresh latent sample."""

    class FakeLatentDist:
        def sample(self):
            return torch.full((1, 4, 8, 8), 99.0)

        def mode(self):
            return torch.full((1, 4, 8, 8), 3.0)

    class FakeEncodeOutput:
        latent_dist = FakeLatentDist()

    class FakeVAE(nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = type("Config", (), {"scaling_factor": 0.5})()

        def encode(self, images):
            assert images.min() >= -1.0
            assert images.max() <= 1.0
            return FakeEncodeOutput()

    latents = encode_latents(FakeVAE(), torch.ones(1, 3, 64, 64))

    assert torch.all(latents == 1.5)


def test_sample_with_different_guidance_scales():
    """Test that sampling completes with different CFG scales."""
    unet = DummyUNet()
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
    )

    batch_size = 1
    prompt_embeds = torch.randn(batch_size, 77, 2048)
    negative_prompt_embeds = torch.randn(batch_size, 77, 2048)
    pooled_prompt_embeds = torch.randn(batch_size, 1280)
    pooled_negative_prompt_embeds = torch.randn(batch_size, 1280)

    # Sample with CFG scale 1.0
    latents_cfg_1 = sample_with_cfg(
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        pooled_negative_prompt_embeds=pooled_negative_prompt_embeds,
        num_inference_steps=3,
        guidance_scale=1.0,
        height=256,
        width=256,
        device="cpu",
        dtype=torch.float32,
    )

    # Sample with CFG scale 7.0
    latents_cfg_7 = sample_with_cfg(
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        pooled_negative_prompt_embeds=pooled_negative_prompt_embeds,
        num_inference_steps=3,
        guidance_scale=7.0,
        height=256,
        width=256,
        device="cpu",
        dtype=torch.float32,
    )

    # Just verify sampling completes and produces valid shapes
    assert latents_cfg_1.shape == (1, 4, 256 // 8, 256 // 8)
    assert latents_cfg_7.shape == (1, 4, 256 // 8, 256 // 8)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_sample_prompts_file_creation(temp_workspace):
    """Test that sample prompts can be loaded from a file."""
    prompts_file = temp_workspace / "prompts.json"
    prompts_file.write_text(
        '[{"prompt": "a cat", "negative": "blurry", "seed": 1}, {"prompt": "a dog"}]'
    )

    specs = load_prompt_specs(prompts_file, samples_per_prompt=2)
    assert len(specs) == 4
    assert specs[0] == PromptSpec(prompt="a cat", negative="blurry", seed=1)
    assert specs[1].prompt == "a cat"
    assert specs[2].prompt == "a dog"
    assert specs[2].negative == ""
    assert specs[2].seed is None


def test_encode_prompts_with_weighting():
    """Test that weighted prompts encode correctly."""
    prompts = ["a (cat:1.5)", "a ((very)) cute dog"]
    text_encoder_1 = DummyTextEncoder(768)
    text_encoder_2 = DummyTextEncoder(1280)
    tokenizer_1 = DummyTokenizer()
    tokenizer_2 = DummyTokenizer()

    # Test with weighting enabled
    prompt_embeds_weighted, pooled_embeds_weighted = encode_prompts_for_sampling(
        prompts,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        "cpu",
        enable_prompt_weighting=True,
    )

    # Should produce correct shapes
    assert prompt_embeds_weighted.shape == (2, 77, 768 + 1280)
    assert pooled_embeds_weighted.shape == (2, 1280)

    # Test with weighting disabled
    prompt_embeds_plain, pooled_embeds_plain = encode_prompts_for_sampling(
        prompts,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        "cpu",
        enable_prompt_weighting=False,
    )

    # Should produce correct shapes
    assert prompt_embeds_plain.shape == (2, 77, 768 + 1280)
    assert pooled_embeds_plain.shape == (2, 1280)

    # Weighted and non-weighted should produce different results (due to random dummy encoder)
    # We can't assert they're different because the dummy encoder returns random values
    # But we can assert the function completes successfully


def test_backward_compatibility():
    """Test that plain prompts work unchanged with weighting enabled."""
    prompts = ["a simple cat", "a simple dog"]
    text_encoder_1 = DummyTextEncoder(768)
    text_encoder_2 = DummyTextEncoder(1280)
    tokenizer_1 = DummyTokenizer()
    tokenizer_2 = DummyTokenizer()

    # Encode with weighting enabled (should take fast path)
    prompt_embeds, pooled_embeds = encode_prompts_for_sampling(
        prompts,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        "cpu",
        enable_prompt_weighting=True,
    )

    # Should produce correct shapes
    assert prompt_embeds.shape == (2, 77, 768 + 1280)
    assert pooled_embeds.shape == (2, 1280)


def test_mixed_batch():
    """Test batch with some weighted and some non-weighted prompts."""
    prompts = [
        "a simple cat",  # No weights
        "a (big:1.3) dog",  # Weighted
        "another plain prompt",  # No weights
    ]
    text_encoder_1 = DummyTextEncoder(768)
    text_encoder_2 = DummyTextEncoder(1280)
    tokenizer_1 = DummyTokenizer()
    tokenizer_2 = DummyTokenizer()

    prompt_embeds, pooled_embeds = encode_prompts_for_sampling(
        prompts,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        "cpu",
        enable_prompt_weighting=True,
    )

    # Should produce correct shapes for all prompts
    assert prompt_embeds.shape == (3, 77, 768 + 1280)
    assert pooled_embeds.shape == (3, 1280)


def test_empty_prompt_with_weighting():
    """Test that empty prompts work with weighting enabled."""
    prompts = [""]
    text_encoder_1 = DummyTextEncoder(768)
    text_encoder_2 = DummyTextEncoder(1280)
    tokenizer_1 = DummyTokenizer()
    tokenizer_2 = DummyTokenizer()

    prompt_embeds, pooled_embeds = encode_prompts_for_sampling(
        prompts,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        "cpu",
        enable_prompt_weighting=True,
    )

    # Should produce correct shapes
    assert prompt_embeds.shape == (1, 77, 768 + 1280)
    assert pooled_embeds.shape == (1, 1280)
