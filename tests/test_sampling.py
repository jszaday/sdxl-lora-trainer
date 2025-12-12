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
    encode_prompts_for_sampling,
    load_prompt_specs,
    sample_with_cfg,
)


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


@pytest.mark.skip("Requires real VAE - slow test")
def test_decode_latents():
    """Test that latent decoding produces valid images."""
    # This would require loading a real VAE, which is slow
    # Skip for now, but keep as placeholder
    pass


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

    # Read prompts
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    assert len(prompts) == 3
    assert prompts[0] == "a cat"
    assert prompts[1] == "a dog"
    assert prompts[2] == "a bird"
