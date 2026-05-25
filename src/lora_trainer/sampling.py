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
from tqdm import tqdm

from .schedulers import build_noise_scheduler, set_scheduler_timesteps


@dataclass
class PromptSpec:
    prompt: str
    negative: str = ""
    seed: int | None = None
    name: str | None = None


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

        name = entry.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("Prompt entry 'name' must be a string if provided")

        return PromptSpec(prompt=prompt, negative=negative, seed=seed, name=name)

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


def _encode_single_encoder(
    prompts: list[str],
    text_encoder,
    tokenizer,
    device: str,
    clip_skip: int = 1,
    *,
    pad_with_end: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts with a single text encoder.

    Args:
        prompts: List of text prompts
        text_encoder: CLIP text encoder
        tokenizer: Tokenizer for the encoder
        device: Device for computation
        clip_skip: Number of layers to skip (1 = penultimate)

    Returns:
        Tuple of (hidden_states, pooled_embeds)
    """
    tokens = _comfy_sdxl_tokenize(prompts, tokenizer, pad_with_end=pad_with_end).to(device)
    batch_size, chunk_count, chunk_length = tokens.shape

    with torch.no_grad():
        encoder_output = text_encoder(
            tokens.reshape(batch_size * chunk_count, chunk_length),
            output_hidden_states=True,
        )

    idx = -(clip_skip + 1)
    hidden_states = encoder_output.hidden_states[idx].reshape(
        batch_size,
        chunk_count * chunk_length,
        -1,
    )
    pooled_embeds = encoder_output[0].reshape(batch_size, chunk_count, -1)[:, 0]

    return hidden_states, pooled_embeds


def _comfy_sdxl_tokenize(
    prompts: list[str],
    tokenizer,
    *,
    pad_with_end: bool,
) -> torch.Tensor:
    """Tokenize one SDXL CLIP stream with ComfyUI's padding convention."""
    max_length = tokenizer.model_max_length
    chunk_capacity = max_length - 2
    encoded: list[list[list[int]]] = []
    max_chunks = 1
    for prompt in prompts:
        token_output = tokenizer(prompt)
        input_ids = (
            token_output["input_ids"]
            if isinstance(token_output, dict)
            else token_output.input_ids
        )
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.flatten().tolist()
        token_ids = list(input_ids)
        if not token_ids:
            raise ValueError("Tokenizer returned no tokens")

        start_token = token_ids[0]
        end_token = token_ids[-1]
        content = token_ids[1:-1]
        pad_token = end_token if pad_with_end else 0

        content_chunks = [
            content[i : i + chunk_capacity] for i in range(0, len(content), chunk_capacity)
        ] or [[]]
        prompt_chunks = []
        for chunk in content_chunks:
            ids = [start_token, *chunk, end_token]
            ids.extend([pad_token] * (max_length - len(ids)))
            prompt_chunks.append(ids)
        encoded.append(prompt_chunks)
        max_chunks = max(max_chunks, len(prompt_chunks))

    for prompt_chunks in encoded:
        while len(prompt_chunks) < max_chunks:
            start_token = prompt_chunks[0][0]
            if pad_with_end:
                end_token = prompt_chunks[0][-1]
            else:
                first_pad = (
                    prompt_chunks[0].index(0) if 0 in prompt_chunks[0] else len(prompt_chunks[0])
                )
                end_token = prompt_chunks[0][first_pad - 1]
            pad_token = end_token if pad_with_end else 0
            ids = [start_token, end_token]
            ids.extend([pad_token] * (max_length - len(ids)))
            prompt_chunks.append(ids)

    return torch.tensor(encoded, dtype=torch.long)


def encode_prompts_for_sampling(
    prompts: list[str],
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    device: str,
    clip_skip: int = 1,
    enable_prompt_weighting: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text prompts using SDXL's dual text encoders for sampling.

    Supports A1111/ComfyUI-style prompt weighting:
    - (text) -> weight *= 1.1
    - ((text)) -> weight *= 1.21
    - (text:1.5) -> weight = 1.5

    Args:
        prompts: List of text prompts
        text_encoder_1: First CLIP text encoder
        text_encoder_2: Second CLIP text encoder
        tokenizer_1: First tokenizer
        tokenizer_2: Second tokenizer
        device: Device for computation
        clip_skip: Number of layers to skip (1 = penultimate)
        enable_prompt_weighting: Whether to parse and apply prompt weights

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds)
    """
    # Check if any prompts have weighting syntax
    has_weights = enable_prompt_weighting and any("(" in p or ":" in p for p in prompts)

    if not has_weights:
        # Fast path: no weighting needed, use original implementation
        hidden_1, pooled_1 = _encode_single_encoder(
            prompts, text_encoder_1, tokenizer_1, device, clip_skip, pad_with_end=True
        )
        hidden_2, pooled_2 = _encode_single_encoder(
            prompts, text_encoder_2, tokenizer_2, device, clip_skip, pad_with_end=False
        )

        cut_to = min(hidden_1.shape[1], hidden_2.shape[1])
        hidden_1 = hidden_1[:, :cut_to]
        hidden_2 = hidden_2[:, :cut_to]
        prompt_embeds = torch.cat([hidden_1, hidden_2], dim=-1)
        return prompt_embeds, pooled_2

    # Weighted path: parse and apply weights
    from .prompt_weighting import apply_prompt_weights, get_token_positions, parse_weighted_prompt

    # Encode empty prompts as reference baseline
    empty_hidden_1, _ = _encode_single_encoder(
        [""] * len(prompts),
        text_encoder_1,
        tokenizer_1,
        device,
        clip_skip,
        pad_with_end=True,
    )
    empty_hidden_2, _ = _encode_single_encoder(
        [""] * len(prompts),
        text_encoder_2,
        tokenizer_2,
        device,
        clip_skip,
        pad_with_end=False,
    )

    weighted_hidden_1 = []
    weighted_hidden_2 = []
    pooled_embeds_list = []

    for i, prompt in enumerate(prompts):
        # Parse weighted segments
        segments = parse_weighted_prompt(prompt)

        # Check if this prompt actually has weights
        if not segments or all(s.weight == 1.0 for s in segments):
            # No weights, encode normally
            hidden_1, _ = _encode_single_encoder(
                [prompt], text_encoder_1, tokenizer_1, device, clip_skip, pad_with_end=True
            )
            hidden_2, pooled = _encode_single_encoder(
                [prompt], text_encoder_2, tokenizer_2, device, clip_skip, pad_with_end=False
            )
            cut_to = min(hidden_1.shape[1], hidden_2.shape[1])
            weighted_hidden_1.append(hidden_1[:, :cut_to][0])
            weighted_hidden_2.append(hidden_2[:, :cut_to][0])
            pooled_embeds_list.append(pooled[0])
            continue

        # Encode the clean text (syntax stripped) so token positions match the
        # segment boundaries tracked below — the raw prompt has `(`, `:weight`,
        # `)` tokens that shift everything if encoded as-is.
        segment_texts = [s.text for s in segments]
        plain_text = "".join(segment_texts)
        positions_1 = get_token_positions(segment_texts, tokenizer_1, tokenizer_1.model_max_length)
        positions_2 = get_token_positions(segment_texts, tokenizer_2, tokenizer_2.model_max_length)

        hidden_1, _ = _encode_single_encoder(
            [plain_text], text_encoder_1, tokenizer_1, device, clip_skip, pad_with_end=True
        )
        hidden_2, pooled = _encode_single_encoder(
            [plain_text], text_encoder_2, tokenizer_2, device, clip_skip, pad_with_end=False
        )
        cut_to = min(hidden_1.shape[1], hidden_2.shape[1])
        hidden_1 = hidden_1[:, :cut_to]
        hidden_2 = hidden_2[:, :cut_to]
        empty_1 = empty_hidden_1[i, :cut_to]
        empty_2 = empty_hidden_2[i, :cut_to]

        # Apply weights to each encoder's embeddings
        weighted_1 = apply_prompt_weights(hidden_1[0], segments, empty_1, positions_1)
        weighted_2 = apply_prompt_weights(hidden_2[0], segments, empty_2, positions_2)

        weighted_hidden_1.append(weighted_1)
        weighted_hidden_2.append(weighted_2)
        pooled_embeds_list.append(pooled[0])  # Pooled embeddings remain unweighted

    # Stack and concatenate
    prompt_embeds = torch.cat(
        [
            torch.stack(weighted_hidden_1, dim=0),
            torch.stack(weighted_hidden_2, dim=0),
        ],
        dim=-1,
    )
    pooled_prompt_embeds = torch.stack(pooled_embeds_list, dim=0)

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
    if sampler_name == "euler":
        from .trt.backends import TorchUnetBackend
        from .trt.config import parse_resolution_free
        from .trt.inference import sample_frozen_sdxl

        resolution = parse_resolution_free(f"{width}x{height}")
        backend = TorchUnetBackend(unet)
        if seeds is not None and len(set(seeds)) > 1:
            latents = []
            for i, seed in enumerate(seeds):
                latents.append(
                    sample_frozen_sdxl(
                        backend,
                        prompt_embeds=prompt_embeds[i : i + 1],
                        negative_prompt_embeds=negative_prompt_embeds[i : i + 1],
                        pooled_prompt_embeds=pooled_prompt_embeds[i : i + 1],
                        pooled_negative_prompt_embeds=pooled_negative_prompt_embeds[i : i + 1],
                        resolution=resolution,
                        sampler=sampler_name,
                        scheduler_name=getattr(scheduler, "comfy_scheduler_name", "normal"),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        device=device,
                        dtype=dtype,
                        seed=seed,
                        denoise=denoise,
                        progress=False,
                    )
                )
            return torch.cat(latents, dim=0)

        seed = seeds[0] if seeds else None
        return sample_frozen_sdxl(
            backend,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            pooled_negative_prompt_embeds=pooled_negative_prompt_embeds,
            resolution=resolution,
            sampler=sampler_name,
            scheduler_name=getattr(scheduler, "comfy_scheduler_name", "normal"),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            dtype=dtype,
            seed=seed,
            denoise=denoise,
            progress=True,
        )

    # Set timesteps/sigmas
    set_scheduler_timesteps(scheduler, num_inference_steps, device=device)

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
        # TODO: this path is currently unreachable (callers always pass denoise=1.0).
        # When wired up, replace with ComfyUI-style sigma expansion:
        #   expanded = round(num_inference_steps / denoise)
        #   scheduler.set_timesteps(expanded); timesteps = scheduler.timesteps[-num_inference_steps:]  # noqa: E501
        # Also [:cut] is wrong — it takes the high-noise end; img2img needs [-cut:].
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


def encode_latents(vae: AutoencoderKL, images: torch.Tensor) -> torch.Tensor:
    """VAE-encode [0, 1] pixel images to scaled latent space."""
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=vae.dtype) * 2.0 - 1.0
    with torch.inference_mode():
        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.mode()
    return latents * vae.config.scaling_factor


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

    # Temporarily move VAE and text encoders to device if they're on CPU
    # (This happens when using cached data - models are kept on CPU to save VRAM)
    vae_original_device = next(vae.parameters()).device
    te1_original_device = next(text_encoder_1.parameters()).device
    te2_original_device = next(text_encoder_2.parameters()).device

    need_to_move = vae_original_device.type == "cpu"
    if need_to_move:
        print("Moving VAE and text encoders to GPU for sampling...")
        vae = vae.to(device)
        text_encoder_1 = text_encoder_1.to(device)
        text_encoder_2 = text_encoder_2.to(device)

    try:
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
            enable_prompt_weighting=getattr(config, "enable_prompt_weighting", True),
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
            enable_prompt_weighting=getattr(config, "enable_prompt_weighting", True),
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

        # Save individual images
        samples_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        for idx, (image, spec) in enumerate(zip(images, prompt_specs, strict=True)):
            # Determine filename
            if spec.name:
                # Use custom name if provided
                filename = f"step_{global_step:06d}_{spec.name}.png"
            else:
                # Fall back to index-based naming
                filename = f"step_{global_step:06d}_{idx}.png"

            save_path = samples_dir / filename

            # Convert to PIL and save
            image_np = image.to(torch.float32).cpu().permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype("uint8")
            Image.fromarray(image_np).save(save_path)
            saved_paths.append(save_path)

            # Log to TensorBoard with unique tag
            tag = f"samples/{spec.name}" if spec.name else f"samples/{idx}"
            writer.add_image(tag, image, global_step)

        print(f"Saved {len(saved_paths)} validation samples to {samples_dir}")

        # Return to train mode
        unet.train()

    finally:
        # Move models back to original device (CPU) if we moved them
        if need_to_move:
            print("Moving VAE and text encoders back to CPU...")
            vae = vae.to(vae_original_device)
            text_encoder_1 = text_encoder_1.to(te1_original_device)
            text_encoder_2 = text_encoder_2.to(te2_original_device)
            torch.cuda.empty_cache()
