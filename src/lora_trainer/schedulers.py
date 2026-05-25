"""Noise scheduler selection for training and sampling.

Maps user-facing scheduler names to diffusers scheduler instances.

TODO: Expand to match full ComfyUI scheduler/sampler set:
- Schedulers: sgm_uniform, exponential, ddim_uniform, beta, linear_quadratic, kl_optimal
- Samplers: euler_cfg_pp, heunpp2, dpm_2/2_ancestral, lms, dpmpp_* variants, uni_pc, etc.
For now, we support the most common ones that match ComfyUI's names.
"""

import torch
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
    scheduler_key = name
    if name == "simple":
        # ComfyUI simple_scheduler: evenly spaced steps from the model's sigma table.
        # Closest diffusers equivalent: uniform linspace without Karras.
        scheduler_config["timestep_spacing"] = "linspace"
        scheduler_config["use_karras_sigmas"] = False
    elif name == "normal":
        scheduler_config["timestep_spacing"] = "linspace"
        scheduler_config["use_karras_sigmas"] = False
    elif name == "karras":
        # "linspace" causes _convert_to_karras to use the actual model sigma_min (~0.029),
        # matching ComfyUI's get_sigmas_karras exactly (max diff ~9e-6).
        # "trailing" was wrong: it used the last trailing sigma (~0.176) as sigma_min,
        # producing a completely different schedule and severely degraded image quality.
        scheduler_config["timestep_spacing"] = "linspace"
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
        scheduler = cls(**scheduler_config)
    except TypeError:
        import warnings
        warnings.warn(
            f"{cls.__name__} rejected use_karras_sigmas/timestep_spacing — "
            "falling back to defaults. Sigma schedule may not match ComfyUI.",
            stacklevel=3,
        )
        scheduler_config.pop("use_karras_sigmas", None)
        scheduler_config.pop("timestep_spacing", None)
        scheduler = cls(**scheduler_config)

    scheduler.comfy_scheduler_name = scheduler_key
    scheduler.comfy_model_sigmas = scheduler.sigmas.detach().cpu().float().clone()
    return scheduler


def _model_sigmas_ascending(scheduler) -> torch.Tensor:
    sigmas = getattr(scheduler, "comfy_model_sigmas", scheduler.sigmas).detach().cpu().float()
    sigmas = sigmas[sigmas > 0]
    if float(sigmas[0]) > float(sigmas[-1]):
        sigmas = sigmas.flip(0)
    return sigmas


def _sigma_from_timestep(scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """Match ComfyUI ModelSamplingDiscrete.sigma log-linear interpolation."""
    model_sigmas = _model_sigmas_ascending(scheduler)
    log_sigmas = model_sigmas.log()
    t = torch.clamp(timesteps.float(), min=0, max=len(model_sigmas) - 1)
    low_idx = t.floor().long()
    high_idx = t.ceil().long()
    weight = t.frac()
    log_sigma = (1 - weight) * log_sigmas[low_idx] + weight * log_sigmas[high_idx]
    return log_sigma.exp()


def comfy_timesteps_for_sigmas(scheduler, sigmas: torch.Tensor) -> torch.Tensor:
    """Match ComfyUI ModelSamplingDiscrete.timestep nearest log-sigma lookup."""
    model_sigmas = _model_sigmas_ascending(scheduler)
    log_sigmas = model_sigmas.log()
    sigmas_cpu = sigmas.detach().cpu().float().clamp_min(torch.finfo(torch.float32).tiny)
    distances = sigmas_cpu.log().reshape(-1, 1) - log_sigmas.reshape(1, -1)
    return distances.abs().argmin(dim=1).to(device=sigmas.device, dtype=torch.float32)


def _comfy_normal_sigmas(scheduler, steps: int, *, sgm: bool = False) -> list[float]:
    model_sigmas = _model_sigmas_ascending(scheduler)
    start = torch.tensor(float(len(model_sigmas) - 1))
    end = torch.tensor(0.0)

    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps)

    return [*_sigma_from_timestep(scheduler, timesteps).tolist(), 0.0]


def _comfy_simple_sigmas(scheduler, steps: int) -> list[float]:
    """Return ComfyUI's `simple` sigma selection from the model sigma table."""
    if steps <= 0:
        return []

    descending = _model_sigmas_ascending(scheduler).flip(0)
    stride = len(descending) / steps
    sigmas = [float(descending[int(x * stride)]) for x in range(steps)]
    sigmas.append(0.0)
    return sigmas


def _comfy_karras_sigmas(scheduler, steps: int) -> list[float]:
    model_sigmas = _model_sigmas_ascending(scheduler)
    sigma_min = float(model_sigmas[0])
    sigma_max = float(model_sigmas[-1])
    rho = 7.0
    ramp = torch.linspace(0, 1, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return [*sigmas.tolist(), 0.0]


def _comfy_exponential_sigmas(scheduler, steps: int) -> list[float]:
    model_sigmas = _model_sigmas_ascending(scheduler)
    sigma_min = float(model_sigmas[0])
    sigma_max = float(model_sigmas[-1])
    sigmas = torch.linspace(
        torch.log(torch.tensor(sigma_max)),
        torch.log(torch.tensor(sigma_min)),
        steps,
    )
    return [*sigmas.exp().tolist(), 0.0]


def comfy_sigmas(scheduler, steps: int) -> list[float] | None:
    """Return ComfyUI-compatible sigmas for supported scheduler names."""
    name = getattr(scheduler, "comfy_scheduler_name", None)
    if name == "normal":
        return _comfy_normal_sigmas(scheduler, steps)
    if name == "sgm_uniform":
        return _comfy_normal_sigmas(scheduler, steps, sgm=True)
    if name == "simple":
        return _comfy_simple_sigmas(scheduler, steps)
    if name == "karras":
        return _comfy_karras_sigmas(scheduler, steps)
    if name == "exponential":
        return _comfy_exponential_sigmas(scheduler, steps)
    return None


def set_scheduler_timesteps(scheduler, steps: int, *, device: str) -> None:
    """Set timesteps, using exact ComfyUI-compatible sigmas where diffusers allows it."""
    sigmas = comfy_sigmas(scheduler, steps)
    if sigmas is not None:
        try:
            scheduler.set_timesteps(device=device, sigmas=sigmas)
            return
        except TypeError:
            pass

    scheduler.set_timesteps(steps, device=device)


def start_sigma_for_timesteps(
    scheduler,
    timesteps: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the sigma corresponding to the first timestep in a sliced schedule."""
    start_index = scheduler.index_for_timestep(timesteps[0], scheduler.timesteps)
    return scheduler.sigmas[start_index].to(device=timesteps.device, dtype=dtype)
