"""Validation sampling with schedulers, samplers, and CFG.

Generates sample images during training to visualize LoRA progress.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from .schedulers import build_noise_scheduler


@dataclass
class PromptSpec:
    prompt: str
    negative: str = ""
    seed: int | None = None


def _normalize_prompt_entry(entry: object) -> PromptSpec:
    """Normalize a prompt entry from JSON/JSONL into PromptSpec."""
    if isinstance(entry, str):
        return PromptSpec(prompt=entry)

    if isinstance(entry, dict):
        prompt = entry.get("prompt") or entry.get("positive") or entry.get("text")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt entry missing 'prompt'/'positive' text")

        negative = entry.get("negative") or ""
        if not isinstance(negative, str):
            raise ValueError("Prompt entry 'negative' must be a string if provided")

        seed = entry.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("Prompt entry 'seed' must be an integer if provided")

        return PromptSpec(prompt=prompt, negative=negative, seed=seed)

    raise ValueError("Prompt entry must be a string or object with prompt/negative/seed")


def load_prompt_specs(path: Path, samples_per_prompt: int) -> list[PromptSpec]:
    """Load prompts from JSON/JSONL with optional negative prompts and seeds."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    content = path.read_text()
    if path.suffix.lower() == ".jsonl":
        entries = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    else:
        data = json.loads(content)
        if isinstance(data, list):
            entries = data
        else:
            entries = [data]

    specs: list[PromptSpec] = []
    for entry in entries:
        spec = _normalize_prompt_entry(entry)
        specs.extend([spec] * samples_per_prompt)

    return specs


def encode_prompts_for_sampling(
    prompts: list[str],
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    device: str,
    clip_skip: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text prompts using SDXL's dual text encoders for sampling.

    Args:
        prompts: List of text prompts
        text_encoder_1: First CLIP text encoder
        text_encoder_2: Second CLIP text encoder
        tokenizer_1: First tokenizer
        tokenizer_2: Second tokenizer
        device: Device for computation

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds)
    """
    # Tokenize with both tokenizers
    tokens_1 = tokenizer_1(
        prompts,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    tokens_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # Encode with both encoders
    with torch.no_grad():
        encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
        encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)

    # SDXL uses penultimate hidden states by default; clip_skip lets us step back further.
    idx = -(clip_skip + 1)

    # SDXL uses penultimate hidden states from encoder 1 and encoder 2
    prompt_embeds = torch.cat(
        [
            encoder_output_1.hidden_states[idx],
            encoder_output_2.hidden_states[idx],
        ],
        dim=-1,
    )

    # Get pooled embeddings from encoder 2
    pooled_prompt_embeds = encoder_output_2[0]

    return prompt_embeds, pooled_prompt_embeds


def sample_with_cfg(
    unet: nn.Module,
    scheduler,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    pooled_negative_prompt_embeds: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype,
    seeds: list[int | None] | None = None,
    sampler_name: str = "ddim",
    denoise: float = 1.0,
) -> torch.Tensor:
    """Run diffusion sampling with classifier-free guidance.

    Args:
        unet: SDXL UNet model
        scheduler: Noise scheduler
        prompt_embeds: Conditional prompt embeddings
        negative_prompt_embeds: Unconditional prompt embeddings
        pooled_prompt_embeds: Pooled conditional embeddings
        pooled_negative_prompt_embeds: Pooled unconditional embeddings
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale (1.0 = no guidance)
        height: Output image height
        width: Output image width
        device: Device for computation
        dtype: Data type for computation

    Returns:
        Denoised latents tensor
    """
    # Set timesteps/sigmas
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Prepare latents (random noise)
    batch_size = prompt_embeds.shape[0]
    if seeds is not None:
        latents = []
        for seed in seeds:
            seed_val = (
                seed
                if seed is not None
                else int(torch.randint(low=0, high=2**31 - 1, size=(1,)).item())
            )
            generator = torch.Generator(device=device).manual_seed(seed_val)
            latents.append(
                torch.randn(
                    (4, height // 8, width // 8),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
        latents = torch.stack(latents, dim=0)
    else:
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            device=device,
            dtype=dtype,
        )

    # Scale initial noise by scheduler
    latents = latents * scheduler.init_noise_sigma

    # Prepare added_cond_kwargs for SDXL
    # Concatenate conditional and unconditional for CFG
    text_embeds = torch.cat([pooled_negative_prompt_embeds, pooled_prompt_embeds])
    time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device).repeat(
        batch_size * 2, 1
    )

    added_cond_kwargs = {
        "text_embeds": text_embeds,
        "time_ids": time_ids,
    }

    # Denoising loop
    timesteps = scheduler.timesteps
    if denoise < 1.0:
        cut = max(1, int(len(timesteps) * denoise))
        timesteps = timesteps[:cut]

    for t in tqdm(timesteps, desc="Sampling", leave=False):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Concatenate prompt embeddings for CFG
        prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])

        # Predict noise
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                prompt_embeds_input,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        # Perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to images using VAE.

    Args:
        vae: SDXL VAE decoder
        latents: Latent tensors

    Returns:
        Image tensors in [0, 1] range
    """
    # Unscale latents
    latents = latents / vae.config.scaling_factor

    # Decode
    with torch.no_grad():
        images = vae.decode(latents.to(vae.dtype)).sample

    # Convert from [-1, 1] to [0, 1]
    images = (images / 2 + 0.5).clamp(0, 1)

    return images


def run_validation_samples(
    unet: nn.Module,
    vae: AutoencoderKL,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    config,
    global_step: int,
    samples_dir: Path,
    writer: SummaryWriter,
    device: str,
) -> None:
    """Run validation sampling and log results.

    Args:
        unet: SDXL UNet model with LoRA
        vae: SDXL VAE for decoding
        text_encoder_1: First text encoder
        text_encoder_2: Second text encoder
        tokenizer_1: First tokenizer
        tokenizer_2: Second tokenizer
        config: TrainingConfig with sampling parameters
        global_step: Current training step
        samples_dir: Directory to save sample images
        writer: TensorBoard writer
        device: Device for computation
    """
    # Check if we should sample
    if config.sample_prompts is None:
        return

    # Load structured prompts
    prompt_specs = load_prompt_specs(config.sample_prompts, config.samples_per_prompt)

    if not prompt_specs:
        print("Warning: No prompts found in sample_prompts file")
        return

    prompts = [spec.prompt for spec in prompt_specs]
    negative_prompts = [spec.negative for spec in prompt_specs]
    seeds = [spec.seed for spec in prompt_specs]

    print(f"\nGenerating {len(prompts)} validation samples...")

    # Build scheduler
    scheduler = build_noise_scheduler(
        config.scheduler, config.sampler_steps, sampler_name=config.sampler
    )

    # Encode prompts
    prompt_embeds, pooled_prompt_embeds = encode_prompts_for_sampling(
        prompts,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        device,
        clip_skip=config.sample_clip_skip,
    )

    # Encode negative prompts (empty string)
    negative_prompt_embeds, pooled_negative_prompt_embeds = encode_prompts_for_sampling(
        negative_prompts,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        device,
        clip_skip=config.sample_clip_skip,
    )

    # Set model to eval mode
    unet.eval()
    vae.eval()

    # Determine dtype from model
    dtype = next(unet.parameters()).dtype

    # Sample
    latents = sample_with_cfg(
        unet=unet,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        pooled_negative_prompt_embeds=pooled_negative_prompt_embeds,
        num_inference_steps=config.sampler_steps,
        guidance_scale=config.cfg,
        height=config.image_size,
        width=config.image_size,
        device=device,
        dtype=dtype,
        seeds=seeds,
        sampler_name=config.sampler,
    )

    # Decode to images
    images = decode_latents(vae, latents)

    # Create grid
    grid = make_grid(images, nrow=config.samples_per_prompt, padding=2, normalize=False)

    # Save to file
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_path = samples_dir / f"step_{global_step:06d}.png"

    # Convert to PIL and save
    grid_np = grid.to(torch.float32).cpu().permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype("uint8")
    Image.fromarray(grid_np).save(save_path)

    # Log to TensorBoard
    writer.add_image("samples", grid, global_step)

    print(f"Saved validation samples to {save_path}")

    # Return to train mode
    unet.train()
