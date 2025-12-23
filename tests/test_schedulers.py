"""Tests for noise scheduler selection."""

import pytest
from diffusers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler

from lora_trainer.schedulers import build_noise_scheduler


def test_build_simple_scheduler():
    """Test that 'simple' creates a DDPM scheduler."""
    scheduler = build_noise_scheduler("simple", num_inference_steps=50)
    assert isinstance(scheduler, DDPMScheduler)
    assert scheduler.config.beta_start == 0.00085
    assert scheduler.config.beta_end == 0.012
    assert scheduler.config.prediction_type == "epsilon"


def test_build_normal_scheduler():
    """Test that 'normal' creates a DDIM scheduler."""
    scheduler = build_noise_scheduler("normal", num_inference_steps=50)
    assert isinstance(scheduler, DDIMScheduler)
    assert scheduler.config.beta_start == 0.00085
    assert scheduler.config.beta_end == 0.012
    assert scheduler.config.prediction_type == "epsilon"


def test_build_karras_scheduler():
    """Test that 'karras' creates an Euler scheduler with trailing timesteps."""
    scheduler = build_noise_scheduler("karras", num_inference_steps=50)
    assert isinstance(scheduler, EulerDiscreteScheduler)
    assert scheduler.config.beta_start == 0.00085
    assert scheduler.config.beta_end == 0.012
    assert scheduler.config.timestep_spacing == "trailing"


def test_unknown_scheduler_raises_error():
    """Test that unknown scheduler name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_noise_scheduler("invalid_scheduler")


def test_scheduler_error_message_includes_valid_options():
    """Test that error message lists valid scheduler options."""
    with pytest.raises(ValueError, match="simple, normal, karras"):
        build_noise_scheduler("bad_name")


def test_build_normal_scheduler_with_sampler():
    """Test that 'normal' config adjusts scheduler per sampler name."""
    scheduler = build_noise_scheduler("normal", num_inference_steps=50, sampler_name="heun")
    assert scheduler.config.timestep_spacing == "linspace"
