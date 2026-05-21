"""Tests for noise scheduler selection."""

import pytest
from diffusers import EulerDiscreteScheduler

from lora_trainer.schedulers import build_noise_scheduler


def test_build_simple_scheduler():
    """Test that 'simple' uses the selected sampler class with linspace spacing."""
    scheduler = build_noise_scheduler("simple", num_inference_steps=50)
    assert isinstance(scheduler, EulerDiscreteScheduler)
    assert scheduler.config.timestep_spacing == "linspace"
    assert scheduler.config.use_karras_sigmas is False


def test_build_normal_scheduler():
    """Test that 'normal' creates an Euler scheduler by default."""
    scheduler = build_noise_scheduler("normal", num_inference_steps=50)
    assert isinstance(scheduler, EulerDiscreteScheduler)
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


def test_build_exponential_scheduler():
    """Test that 'exponential' uses trailing spacing without Karras sigmas."""
    scheduler = build_noise_scheduler("exponential")
    assert hasattr(scheduler.config, "use_karras_sigmas")
    assert scheduler.config.use_karras_sigmas is False
    assert scheduler.config.timestep_spacing == "trailing"


def test_build_sgm_uniform_scheduler():
    """Test that 'sgm_uniform' uses leading timestep spacing."""
    scheduler = build_noise_scheduler("sgm_uniform")
    assert hasattr(scheduler.config, "timestep_spacing")
    assert scheduler.config.timestep_spacing == "leading"


def test_build_dpmpp_2m_sampler():
    """Test that 'dpmpp_2m' creates a DPMSolverMultistepScheduler."""
    from diffusers import DPMSolverMultistepScheduler

    scheduler = build_noise_scheduler("normal", sampler_name="dpmpp_2m")
    assert isinstance(scheduler, DPMSolverMultistepScheduler)
    assert scheduler.config.algorithm_type == "dpmsolver++"


def test_build_dpmpp_2m_sde_sampler():
    """Test that 'dpmpp_2m_sde' creates a DPMSolverMultistepScheduler with SDE."""
    from diffusers import DPMSolverMultistepScheduler

    scheduler = build_noise_scheduler("normal", sampler_name="dpmpp_2m_sde")
    assert isinstance(scheduler, DPMSolverMultistepScheduler)
    assert scheduler.config.algorithm_type == "sde-dpmsolver++"


def test_unknown_scheduler_raises_error():
    """Test that unknown scheduler name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scheduler 'invalid_scheduler'"):
        build_noise_scheduler("invalid_scheduler")


def test_scheduler_error_message_includes_valid_options():
    """Test that error message lists valid scheduler options."""
    with pytest.raises(ValueError, match="karras, exponential, sgm_uniform"):
        build_noise_scheduler("bad_name")


def test_unknown_sampler_raises_error():
    """Test that unknown sampler name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown sampler 'unknown'"):
        build_noise_scheduler("normal", sampler_name="unknown")


def test_build_normal_scheduler_with_sampler():
    """Test that 'normal' config adjusts scheduler per sampler name."""
    scheduler = build_noise_scheduler("normal", num_inference_steps=50, sampler_name="heun")
    assert scheduler.config.timestep_spacing == "linspace"
