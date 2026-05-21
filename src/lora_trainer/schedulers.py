"""Noise scheduler selection for training and sampling.

Maps user-facing scheduler names to diffusers scheduler instances.

TODO: Expand to match full ComfyUI scheduler/sampler set:
- Schedulers: sgm_uniform, exponential, ddim_uniform, beta, linear_quadratic, kl_optimal
- Samplers: euler_cfg_pp, heunpp2, dpm_2/2_ancestral, lms, dpmpp_* variants, uni_pc, etc.
For now, we support the most common ones that match ComfyUI's names.
"""

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


def build_noise_scheduler(
    name: str, num_inference_steps: int = 50, sampler_name: str | None = None
) -> object:
    """Build a diffusers noise scheduler from a user-facing name.

    Args:
        name: Scheduler name (simple, normal, karras, exponential, sgm_uniform)
        num_inference_steps: Number of denoising steps for sampling
        sampler_name: Sampler algorithm (euler, euler_ancestral, heun, dpmpp_2m, etc.)

    Returns:
        Diffusers scheduler instance configured for SDXL

    Raises:
        ValueError: If scheduler or sampler name is not recognized
    """
    # SDXL-compatible scheduler configuration
    scheduler_config = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
    }

    # Map sampler names to classes
    sampler_name = (sampler_name or "euler").lower()

    if sampler_name == "euler":
        cls = EulerDiscreteScheduler
    elif sampler_name == "euler_ancestral":
        cls = EulerAncestralDiscreteScheduler
    elif sampler_name == "heun":
        cls = HeunDiscreteScheduler
    elif sampler_name == "dpmpp_2m":
        cls = DPMSolverMultistepScheduler
        scheduler_config["solver_order"] = 2
        scheduler_config["algorithm_type"] = "dpmsolver++"
    elif sampler_name == "dpmpp_2m_sde":
        cls = DPMSolverMultistepScheduler
        scheduler_config["solver_order"] = 2
        scheduler_config["algorithm_type"] = "sde-dpmsolver++"
    elif sampler_name == "dpmpp_sde":
        # Stochastic 2nd-order ODE solver (k-diffusion sample_dpmpp_sde)
        cls = DPMSolverSDEScheduler
    elif sampler_name == "lms":
        cls = LMSDiscreteScheduler
    elif sampler_name == "ddim":
        cls = DDIMScheduler
    elif sampler_name == "pndm":
        cls = PNDMScheduler
    elif sampler_name == "ddpm":
        cls = DDPMScheduler
    else:
        raise ValueError(f"Unknown sampler '{sampler_name}'")

    # Map scheduler (sigma) names to timestep_spacing/use_karras_sigmas.
    # ComfyUI reference: comfy/samplers.py SCHEDULER_HANDLERS.
    name = name.lower()
    if name == "simple":
        # ComfyUI simple_scheduler: evenly spaced steps from the model's sigma table.
        # Closest diffusers equivalent: uniform linspace without Karras.
        scheduler_config["timestep_spacing"] = "linspace"
        scheduler_config["use_karras_sigmas"] = False
    elif name == "normal":
        scheduler_config["timestep_spacing"] = "linspace"
        scheduler_config["use_karras_sigmas"] = False
    elif name == "karras":
        scheduler_config["timestep_spacing"] = "trailing"
        scheduler_config["use_karras_sigmas"] = True
    elif name == "exponential":
        # ComfyUI: get_sigmas_exponential — log-linear from sigma_max to sigma_min.
        # Diffusers doesn't expose this for all sampler classes; trailing without
        # Karras is the closest available fallback.
        scheduler_config["timestep_spacing"] = "trailing"
        scheduler_config["use_karras_sigmas"] = False
    elif name == "sgm_uniform":
        # ComfyUI sgm_uniform: normal_scheduler(sgm=True) — N+1 linspace timesteps,
        # drop the last. Leading spacing is the diffusers approximation.
        scheduler_config["timestep_spacing"] = "leading"
        scheduler_config["use_karras_sigmas"] = False
    else:
        supported_sched = ["simple", "normal", "karras", "exponential", "sgm_uniform"]
        raise ValueError(f"Unknown scheduler '{name}'. Supported: {', '.join(supported_sched)}")

    # Special handling for DDIM (doesn't support use_karras_sigmas in all versions)
    if cls == DDIMScheduler:
        # DDIM uses different flags
        ddim_config = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "clip_sample": False,
            "set_alpha_to_one": False,
        }
        return DDIMScheduler(**ddim_config)

    # Instantiate
    try:
        return cls(**scheduler_config)
    except TypeError:
        # Fallback for older diffusers versions or schedulers that don't support some flags
        if "use_karras_sigmas" in scheduler_config:
            del scheduler_config["use_karras_sigmas"]
        if "timestep_spacing" in scheduler_config:
            del scheduler_config["timestep_spacing"]
        return cls(**scheduler_config)
