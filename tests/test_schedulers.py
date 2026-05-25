"""Tests for noise scheduler selection."""

import pytest
from diffusers import EulerDiscreteScheduler

from lora_trainer.schedulers import (
    build_noise_scheduler,
    comfy_sigmas,
    comfy_timesteps_for_sigmas,
    set_scheduler_timesteps,
)


def test_build_simple_scheduler():
    """Test that 'simple' uses the selected sampler class."""
    scheduler = build_noise_scheduler("simple", num_inference_steps=50)
    assert isinstance(scheduler, EulerDiscreteScheduler)
    assert scheduler.comfy_scheduler_name == "simple"
    assert scheduler.config.use_karras_sigmas is False


def test_simple_scheduler_uses_comfy_sigma_table_selection():
    """ComfyUI simple picks uniformly from the model sigma table, not linspace timesteps."""
    scheduler = build_noise_scheduler("simple", num_inference_steps=5)
    base_sigmas = scheduler.sigmas[scheduler.sigmas > 0].clone()

    set_scheduler_timesteps(scheduler, 5, device="cpu")

    stride = len(base_sigmas) / 5
    expected = [float(base_sigmas[int(x * stride)]) for x in range(5)] + [0.0]
    assert scheduler.sigmas.tolist() == pytest.approx(expected)


def test_normal_scheduler_uses_comfy_log_sigma_interpolation():
    scheduler = build_noise_scheduler("normal", num_inference_steps=5)
    model_sigmas = scheduler.sigmas[scheduler.sigmas > 0].flip(0)
    timesteps = [999.0, 749.25, 499.5, 249.75, 0.0]
    log_sigmas = model_sigmas.log()
    expected = []
    for timestep in timesteps:
        low = int(timestep)
        high = int(timestep) if timestep.is_integer() else int(timestep) + 1
        weight = timestep - low
        expected.append(float(((1 - weight) * log_sigmas[low] + weight * log_sigmas[high]).exp()))
    expected.append(0.0)

    assert comfy_sigmas(scheduler, 5) == pytest.approx(expected)


def test_comfy_timesteps_use_nearest_log_sigma_index():
    scheduler = build_noise_scheduler("normal", num_inference_steps=5)
    set_scheduler_timesteps(scheduler, 5, device="cpu")

    timesteps = comfy_timesteps_for_sigmas(scheduler, scheduler.sigmas[:-1])

    assert timesteps.tolist() == pytest.approx([999.0, 749.0, 500.0, 250.0, 0.0])


def test_sgm_uniform_scheduler_uses_comfy_offset_linspace():
    scheduler = build_noise_scheduler("sgm_uniform", num_inference_steps=5)
    sigmas = comfy_sigmas(scheduler, 5)
    normal_sigmas = comfy_sigmas(build_noise_scheduler("normal", num_inference_steps=5), 5)

    assert sigmas is not None
    assert normal_sigmas is not None
    assert sigmas[-1] == 0.0
    assert sigmas[0] == pytest.approx(normal_sigmas[0])
    assert sigmas[-2] > normal_sigmas[-2]


def test_karras_scheduler_uses_comfy_formula():
    scheduler = build_noise_scheduler("karras", num_inference_steps=5)
    model_sigmas = scheduler.sigmas[scheduler.sigmas > 0]
    sigma_max = float(model_sigmas[0])
    sigma_min = float(model_sigmas[-1])
    rho = 7.0
    ramp = [0.0, 0.25, 0.5, 0.75, 1.0]
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    expected = [(max_inv_rho + r * (min_inv_rho - max_inv_rho)) ** rho for r in ramp] + [0.0]

    assert comfy_sigmas(scheduler, 5) == pytest.approx(expected)


def test_exponential_scheduler_uses_comfy_formula():
    scheduler = build_noise_scheduler("exponential", num_inference_steps=5)
    model_sigmas = scheduler.sigmas[scheduler.sigmas > 0]
    sigma_max = float(model_sigmas[0])
    sigma_min = float(model_sigmas[-1])
    expected = [
        sigma_max,
        (sigma_max**0.75) * (sigma_min**0.25),
        (sigma_max**0.5) * (sigma_min**0.5),
        (sigma_max**0.25) * (sigma_min**0.75),
        sigma_min,
        0.0,
    ]

    assert comfy_sigmas(scheduler, 5) == pytest.approx(expected)


def test_build_normal_scheduler():
    """Test that 'normal' creates an Euler scheduler by default."""
    scheduler = build_noise_scheduler("normal", num_inference_steps=50)
    assert isinstance(scheduler, EulerDiscreteScheduler)
    assert scheduler.config.beta_start == 0.00085
    assert scheduler.config.beta_end == 0.012
    assert scheduler.config.prediction_type == "epsilon"


def test_build_karras_scheduler():
    """Test that 'karras' creates an Euler scheduler with Karras sigmas."""
    scheduler = build_noise_scheduler("karras", num_inference_steps=50)
    assert isinstance(scheduler, EulerDiscreteScheduler)
    assert scheduler.config.beta_start == 0.00085
    assert scheduler.config.beta_end == 0.012
    assert scheduler.config.timestep_spacing == "linspace"
    assert scheduler.config.use_karras_sigmas is True


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
