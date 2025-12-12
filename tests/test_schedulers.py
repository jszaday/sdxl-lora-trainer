"""Tests for noise scheduler selection."""

import pytest
from diffusers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler

from lora_trainer.schedulers import build_noise_scheduler, build_sampler


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


def test_build_euler_sampler():
    """Test that 'euler' sampler is valid."""
    sampler = build_sampler("euler", scheduler=None)
    assert sampler == "euler"


def test_build_euler_ancestral_sampler():
    """Test that 'euler_ancestral' sampler is valid."""
    sampler = build_sampler("euler_ancestral", scheduler=None)
    assert sampler == "euler_ancestral"


def test_build_ddim_sampler():
    """Test that 'ddim' sampler is valid."""
    sampler = build_sampler("ddim", scheduler=None)
    assert sampler == "ddim"


def test_build_heun_sampler():
    """Test that 'heun' sampler is valid."""
    sampler = build_sampler("heun", scheduler=None)
    assert sampler == "heun"


def test_unknown_sampler_raises_error():
    """Test that unknown sampler name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sampler"):
        build_sampler("invalid_sampler", scheduler=None)


def test_sampler_error_message_includes_valid_options():
    """Test that error message lists valid sampler options."""
    with pytest.raises(ValueError, match="ddim, euler, euler_ancestral, heun"):
        build_sampler("bad_sampler", scheduler=None)
