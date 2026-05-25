"""Device-resident SDXL denoise loop for frozen inference."""

import math
from collections.abc import Callable

import torch
from tqdm import tqdm

from lora_trainer.schedulers import (
    build_noise_scheduler,
    comfy_timesteps_for_sigmas,
    set_scheduler_timesteps,
)

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
        generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (batch_size, 4, resolution.latent_height, resolution.latent_width),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    return noise.to(device=device, dtype=dtype)


def _randn_latents_like(
    latents: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    seed: int | None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    return noise.to(device=device, dtype=dtype)


def _slice_schedule_for_denoise(
    scheduler,
    num_inference_steps: int,
    denoise: float,
    *,
    use_comfy_timesteps: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        # Non-sigma schedulers (e.g. DDIM, PNDM): use their native timesteps directly.
        timesteps = scheduler.timesteps
        if denoise < 1.0:
            timesteps = timesteps[-num_inference_steps:]
        return timesteps, None
    timesteps = scheduler.timesteps
    if denoise < 1.0:
        sigmas = sigmas[-(num_inference_steps + 1) :]
        timesteps = timesteps[-num_inference_steps:]
    if use_comfy_timesteps:
        timesteps = comfy_timesteps_for_sigmas(scheduler, sigmas[:-1])
    return timesteps, sigmas


def _prepare_ksampler_latents(
    latents: torch.Tensor | None,
    *,
    resolution: ResolutionSpec,
    batch_size: int,
    sigmas: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    seed: int | None,
    denoise: float,
) -> torch.Tensor:
    sigma = sigmas[0].to(device=device, dtype=dtype)
    if latents is None:
        noise = make_initial_latents(
            resolution,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        if denoise >= 0.9999:
            return noise * torch.sqrt(1.0 + sigma**2)
        return noise * sigma

    latents = latents.to(device=device, dtype=dtype)
    validate_latents_shape(latents, resolution, batch_size=batch_size)
    noise = _randn_latents_like(latents, device=device, dtype=dtype, seed=seed)
    if denoise >= 0.9999:
        return latents + noise * torch.sqrt(1.0 + sigma**2)
    return latents + noise * sigma


def _predict_denoised_cfg(
    unet_backend: UnetBackend,
    latents: torch.Tensor,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds_input: torch.Tensor,
    added_cond_kwargs: dict[str, torch.Tensor],
    guidance_scale: float,
    *,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigma = sigma.to(device=device, dtype=dtype)
    latent_model_input = torch.cat([latents, latents], dim=0)
    latent_model_input = latent_model_input / torch.sqrt(1.0 + sigma**2)
    timestep_input = timestep.to(device=device, dtype=torch.float32).expand(
        latent_model_input.shape[0]
    )

    noise_pred = unet_backend(
        latent_model_input,
        timestep_input,
        prompt_embeds_input,
        added_cond_kwargs=added_cond_kwargs,
    )
    noise_uncond, noise_text = noise_pred.chunk(2)
    denoised_uncond = latents - noise_uncond * sigma
    denoised_text = latents - noise_text * sigma
    return denoised_uncond + guidance_scale * (denoised_text - denoised_uncond)


def _sample_euler_ksampler(
    unet_backend: UnetBackend,
    *,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds_input: torch.Tensor,
    added_cond_kwargs: dict[str, torch.Tensor],
    guidance_scale: float,
    device: str,
    dtype: torch.dtype,
    progress: bool,
    on_step: Callable[[int, int], None] | None,
) -> torch.Tensor:
    iterator = range(len(sigmas) - 1)
    if progress:
        iterator = tqdm(iterator, total=len(sigmas) - 1, desc="Sampling", leave=False)

    for i in iterator:
        sigma = sigmas[i].to(device=device, dtype=dtype)
        next_sigma = sigmas[i + 1].to(device=device, dtype=dtype)
        denoised = _predict_denoised_cfg(
            unet_backend,
            latents,
            sigma,
            timesteps[i],
            prompt_embeds_input,
            added_cond_kwargs,
            guidance_scale,
            device=device,
            dtype=dtype,
        )
        derivative = (latents - denoised) / sigma
        latents = latents + derivative * (next_sigma - sigma)
        if on_step is not None:
            on_step(i + 1, len(sigmas) - 1)

    return latents


def _match_conditioning_sequence_lengths(
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_len = prompt_embeds.shape[1]
    neg_len = negative_prompt_embeds.shape[1]
    if pos_len == neg_len:
        return prompt_embeds, negative_prompt_embeds

    target_len = math.lcm(pos_len, neg_len)
    if target_len % 77 != 0:
        raise ValueError(
            f"Expected SDXL conditioning length to be a multiple of 77, got {target_len}"
        )

    # TODO: .repeat() copies the chunk verbatim including BOS/EOS tokens, which matches
    # ComfyUI's _cond_equal_size convention but is semantically questionable — a true
    # empty second chunk should probably be [BOS, EOS, EOS, EOS, ...] not a full copy.
    # Revisit once we have side-by-side quality comparisons vs. ComfyUI for long prompts.
    def pad_chunks(embeds: torch.Tensor) -> torch.Tensor:
        if embeds.shape[1] == target_len:
            return embeds
        return embeds.repeat(1, target_len // embeds.shape[1], 1)

    return pad_chunks(prompt_embeds), pad_chunks(negative_prompt_embeds)


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
    # Match ComfyUI's img2img sigma expansion: build a schedule for int(steps / denoise)
    # steps, then run only the last num_inference_steps timesteps so the starting sigma
    # correctly reflects the denoise strength rather than the highest noise level.
    expanded_steps = (
        max(num_inference_steps, round(num_inference_steps / denoise))
        if denoise < 1.0
        else num_inference_steps
    )
    scheduler = build_noise_scheduler(
        scheduler_name,
        num_inference_steps=expanded_steps,
        sampler_name=sampler,
    )
    set_scheduler_timesteps(scheduler, expanded_steps, device=device)

    # For img2img/partial denoise, Comfy builds the expanded schedule and starts
    # from the first sigma of the sliced tail. Diffusers' init_noise_sigma always
    # points at the full schedule maximum, which is wrong when denoise < 1.
    timesteps, sigmas = _slice_schedule_for_denoise(
        scheduler,
        num_inference_steps,
        denoise,
        use_comfy_timesteps=sampler == "euler",
    )

    prompt_embeds, negative_prompt_embeds = _match_conditioning_sequence_lengths(
        prompt_embeds,
        negative_prompt_embeds,
    )

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

    if sampler == "euler":
        if sigmas is None:
            raise ValueError(
                f"Euler sampler requires a sigma-based scheduler (got {scheduler_name!r}). "
                "Use a sigma-compatible scheduler such as 'karras' or 'normal'."
            )
        latents = _prepare_ksampler_latents(
            latents,
            resolution=resolution,
            batch_size=batch_size,
            sigmas=sigmas,
            device=device,
            dtype=dtype,
            seed=seed,
            denoise=denoise,
        )
        return _sample_euler_ksampler(
            unet_backend,
            latents=latents,
            sigmas=sigmas,
            timesteps=timesteps,
            prompt_embeds_input=prompt_embeds_input,
            added_cond_kwargs=added_cond_kwargs,
            guidance_scale=guidance_scale,
            device=device,
            dtype=dtype,
            progress=progress,
            on_step=on_step,
        )

    if latents is None:
        latents = make_initial_latents(
            resolution,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        if denoise < 1.0:
            if sigmas is not None:
                start_sigma = sigmas[0].to(device=device, dtype=dtype)
                latents = latents * start_sigma
            else:
                latents = latents * scheduler.init_noise_sigma
        else:
            latents = latents * scheduler.init_noise_sigma
    else:
        latents = latents.to(device=device, dtype=dtype)
        validate_latents_shape(latents, resolution, batch_size=batch_size)
        if denoise < 1.0:
            noise = _randn_latents_like(latents, device=device, dtype=dtype, seed=seed)
            if sigmas is not None:
                start_sigma = sigmas[0].to(device=device, dtype=dtype)
                latents = latents + noise * start_sigma
            else:
                latents = scheduler.add_noise(latents, noise, timesteps[:1].to(device=device)).to(
                    dtype=dtype
                )

    iterator = tqdm(timesteps, desc="Sampling", leave=False) if progress else timesteps
    if device == "cuda":
        # Tell cudagraph trees this is a new independent inference run so pool
        # allocations from the previous pass are not considered live outputs.
        torch.compiler.cudagraph_mark_step_begin()
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
