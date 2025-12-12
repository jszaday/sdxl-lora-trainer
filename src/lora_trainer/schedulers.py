"""Noise scheduler selection for training and sampling.

Maps user-facing scheduler names to diffusers scheduler instances.

TODO: Expand to match full ComfyUI scheduler/sampler set:
- Schedulers: sgm_uniform, exponential, ddim_uniform, beta, linear_quadratic, kl_optimal
- Samplers: euler_cfg_pp, heunpp2, dpm_2/2_ancestral, lms, dpmpp_* variants, uni_pc, etc.
For now, we support the most common ones that match ComfyUI's names.
"""

from diffusers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler


def build_noise_scheduler(name: str, num_inference_steps: int = 50) -> object:
    """Build a diffusers noise scheduler from a user-facing name.

    Args:
        name: Scheduler name (simple, normal, karras)
        num_inference_steps: Number of denoising steps for sampling

    Returns:
        Diffusers scheduler instance configured for SDXL

    Raises:
        ValueError: If scheduler name is not recognized
    """
    # SDXL-compatible scheduler configuration
    scheduler_config = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
    }

    if name == "simple":
        # Simple linear scheduler (DDPM)
        return DDPMScheduler(**scheduler_config)

    elif name == "normal":
        # Normal scheduler without special noise scaling (DDIM-like)
        return DDIMScheduler(
            **scheduler_config,
            clip_sample=False,
            set_alpha_to_one=False,
        )

    elif name == "karras":
        # Karras noise schedule (uses special timestep spacing)
        # Use Euler scheduler with timestep_spacing="trailing" for Karras-like behavior
        return EulerDiscreteScheduler(
            **scheduler_config,
            timestep_spacing="trailing",
        )

    else:
        raise ValueError(f"Unknown scheduler '{name}'. " f"Valid options: simple, normal, karras")


def build_sampler(name: str, scheduler) -> str:
    """Validate sampler name and return it.

    Args:
        name: Sampler algorithm name
        scheduler: The scheduler to use with this sampler

    Returns:
        The validated sampler name

    Raises:
        ValueError: If sampler name is not recognized
    """
    valid_samplers = {"euler", "euler_ancestral", "ddim", "heun"}

    if name not in valid_samplers:
        raise ValueError(
            f"Unknown sampler '{name}'. " f"Valid options: {', '.join(sorted(valid_samplers))}"
        )

    return name
