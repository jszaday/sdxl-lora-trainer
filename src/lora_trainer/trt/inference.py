"""Device-resident SDXL denoise loop for frozen inference."""

from collections.abc import Callable

import torch
from tqdm import tqdm

from lora_trainer.schedulers import build_noise_scheduler

from .backends import UnetBackend
from .config import ResolutionSpec, validate_latents_shape


def make_initial_latents(
    resolution: ResolutionSpec,
    *,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    seed: int | None,
) -> torch.Tensor:
    """Create initial SDXL latent noise on the target device."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (batch_size, 4, resolution.latent_height, resolution.latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )


@torch.inference_mode()
def sample_frozen_sdxl(
    unet_backend: UnetBackend,
    *,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    pooled_negative_prompt_embeds: torch.Tensor,
    resolution: ResolutionSpec,
    sampler: str,
    scheduler_name: str,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
    dtype: torch.dtype,
    latents: torch.Tensor | None = None,
    seed: int | None = None,
    denoise: float = 1.0,
    progress: bool = True,
    on_step: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Run SDXL sampling while keeping loop tensors on device."""
    batch_size = int(prompt_embeds.shape[0])
    scheduler = build_noise_scheduler(
        scheduler_name,
        num_inference_steps=num_inference_steps,
        sampler_name=sampler,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    if latents is None:
        latents = make_initial_latents(
            resolution,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        latents = latents * scheduler.init_noise_sigma
    else:
        latents = latents.to(device=device, dtype=dtype)
        validate_latents_shape(latents, resolution, batch_size=batch_size)

    prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds]).to(
        device=device,
        dtype=dtype,
    )
    text_embeds = torch.cat([pooled_negative_prompt_embeds, pooled_prompt_embeds]).to(
        device=device,
        dtype=dtype,
    )
    time_ids = torch.tensor(
        [[resolution.height, resolution.width, 0, 0, resolution.height, resolution.width]],
        device=device,
        dtype=dtype,
    ).repeat(batch_size * 2, 1)
    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

    timesteps = scheduler.timesteps
    if denoise < 1.0:
        keep = max(1, int(len(timesteps) * denoise))
        timesteps = timesteps[-keep:]

    iterator = tqdm(timesteps, desc="Sampling", leave=False) if progress else timesteps
    if device == "cuda":
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("sdxl_denoise")
    try:
        for i, timestep in enumerate(iterator):
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
            timestep_input = timestep.to(device=device, dtype=torch.float32).expand(
                latent_model_input.shape[0]
            )

            noise_pred = unet_backend(
                latent_model_input,
                timestep_input,
                prompt_embeds_input,
                added_cond_kwargs=added_cond_kwargs,
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample
            if on_step is not None:
                on_step(i + 1, len(timesteps))
    finally:
        if device == "cuda":
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

    return latents
