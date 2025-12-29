"""Core training loop with progress tracking and checkpointing."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lora_converter.converter import convert_lora_state

from .logging import log_perf_metrics
from .sampling import run_validation_samples

if TYPE_CHECKING:
    from .config import TrainingConfig


# TODO: Prompt weighting for training
# To add training-time prompt weighting:
# 1. Import from prompt_weighting module (parse_weighted_prompt,
#    apply_prompt_weights, get_token_positions)
# 2. Add enable_training_prompt_weighting config flag (default False for safety)
# 3. Apply same logic as encode_prompts_for_sampling() in sampling.py:
#    - Detect if any captions have weighting syntax: "(" or ":" in caption
#    - Encode empty reference prompts for baseline
#    - For each caption, parse weighted segments
#    - Get token positions for each segment (separate for each encoder)
#    - Apply weights to hidden states before concatenation
# 4. Test impact on training convergence and loss computation
# Note: This affects loss computation and may change training dynamics.
# Consider starting with a small learning rate and monitoring loss curves carefully.
def encode_prompts(
    captions: list[str],
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    device: str,
) -> torch.Tensor:
    """Encode text prompts using SDXL's dual text encoders.

    Args:
        captions: List of text captions
        text_encoder_1: First CLIP text encoder
        text_encoder_2: Second CLIP text encoder
        tokenizer_1: First tokenizer
        tokenizer_2: Second tokenizer
        device: Device for computation

    Returns:
        Pooled prompt embeddings tensor
    """
    # Tokenize with both tokenizers
    tokens_1 = tokenizer_1(
        captions,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    tokens_2 = tokenizer_2(
        captions,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # Encode with both encoders
    with torch.no_grad():
        encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
        encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)

    # SDXL uses penultimate hidden states from encoder 1 and pooled output from encoder 2
    prompt_embeds = torch.cat(
        [
            encoder_output_1.hidden_states[-2],
            encoder_output_2.hidden_states[-2],
        ],
        dim=-1,
    )

    # Get pooled embeddings
    pooled_prompt_embeds = encoder_output_2[0]

    return prompt_embeds, pooled_prompt_embeds


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: "TrainingConfig",
    dirs: dict[str, Path],
    writer: SummaryWriter,
    lr_scheduler: LRScheduler | None = None,
    device: str = "cuda",
    vae=None,
    text_encoder_1=None,
    text_encoder_2=None,
    tokenizer_1=None,
    tokenizer_2=None,
    start_step: int = 0,
    unet_adapter=None,
    te1_adapter=None,
    te2_adapter=None,
    base_model: str | None = None,
    cached_data: bool = False,
) -> None:
    """Run the complete training loop.

    Args:
        model: UNet model to train
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        config: TrainingConfig instance
        dirs: Dictionary of output directories
        writer: TensorBoard writer
        device: Device to train on
        vae: VAE for encoding images (optional for testing)
        text_encoder_1: First text encoder (optional for testing)
        text_encoder_2: Second text encoder (optional for testing)
        tokenizer_1: First tokenizer (optional for testing)
        tokenizer_2: Second tokenizer (optional for testing)
        cached_data: Whether the dataloader yields cached latents/embeds
    """
    model.train()
    model = model.to(device)
    # Set up noise scheduler for diffusion training
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",  # Predict noise
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    # Determine if we're using real diffusion (on-the-fly encoding)
    use_real_diffusion = vae is not None and not cached_data

    if use_real_diffusion:
        vae = vae.to(device)
        vae.eval()
        if text_encoder_1 is not None:
            text_encoder_1 = text_encoder_1.to(device)
            text_encoder_1.eval()
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(device)
            text_encoder_2.eval()

    use_cuda_device = device.startswith("cuda") and torch.cuda.is_available()
    copy_stream = torch.cuda.Stream(device=device) if use_cuda_device else None

    def _async_to_device(batch):
        # Detect batch type: cached (has latents) or raw (has pixel_values)
        is_cached = "latents" in batch

        if copy_stream is None:
            # No async copying - just move to device
            if is_cached:
                result = {
                    "latents": batch["latents"].to(device),
                    "prompt_embeds": batch["prompt_embeds"].to(device),
                    "pooled_embeds": batch["pooled_embeds"].to(device),
                }
                if "time_ids" in batch:
                    result["time_ids"] = batch["time_ids"].to(device)
                return result
            else:
                result = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "caption": batch["caption"],
                }
                if "time_ids" in batch:
                    result["time_ids"] = batch["time_ids"].to(device)
                return result

        # Async copying with CUDA streams
        if is_cached:
            result = {}
            with torch.cuda.stream(copy_stream):
                result["latents"] = batch["latents"].to(device, non_blocking=True)
                result["prompt_embeds"] = batch["prompt_embeds"].to(device, non_blocking=True)
                result["pooled_embeds"] = batch["pooled_embeds"].to(device, non_blocking=True)
                if "time_ids" in batch:
                    result["time_ids"] = batch["time_ids"].to(device, non_blocking=True)
        else:
            result = {"caption": batch["caption"]}
            with torch.cuda.stream(copy_stream):
                result["pixel_values"] = batch["pixel_values"].to(device, non_blocking=True)
                if "time_ids" in batch:
                    result["time_ids"] = batch["time_ids"].to(device, non_blocking=True)
        return result

    def _wait_for_batch():
        if copy_stream is not None:
            torch.cuda.current_stream().wait_stream(copy_stream)

    global_step = start_step
    current_epoch = 0
    last_checkpoint_step: int | None = None

    # Loss accumulation for batched logging (reduces GPU sync overhead)
    accumulated_losses = []

    # Progress bar for global steps
    pbar = tqdm(total=config.steps, desc="Training", unit="step", initial=global_step)

    # Optional initial sampling before training starts
    if (
        vae is not None
        and config.sample_prompts is not None
        and global_step == 0
        and config.sample_every > 0
    ):
        run_validation_samples(
            unet=model,
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            config=config,
            global_step=global_step,
            samples_dir=dirs["samples"],
            writer=writer,
            device=device,
        )

    step_start_time = time.perf_counter()
    while global_step < config.steps:
        current_epoch += 1
        data_iter = iter(dataloader)
        try:
            prefetched = _async_to_device(next(data_iter))
        except StopIteration:
            break

        batch_idx = 0
        while global_step < config.steps and prefetched is not None:
            _wait_for_batch()
            batch = prefetched
            try:
                raw_next = next(data_iter)
            except StopIteration:
                raw_next = None
            prefetched = _async_to_device(raw_next) if raw_next is not None else None

            # Detect if we're using cached data or raw images
            batch_is_cached = "latents" in batch

            if batch_is_cached:
                # Using pre-cached latents and embeddings
                latents = batch["latents"].to(device)
                prompt_embeds = batch["prompt_embeds"].to(device)
                pooled_embeds = batch["pooled_embeds"].to(device)

                # SDXL uses pooled embeddings as added_cond_kwargs
                added_cond_kwargs = {"text_embeds": pooled_embeds}
                # Add time_ids (original size, crops, target size)
                time_ids = batch["time_ids"].to(device)
                added_cond_kwargs["time_ids"] = time_ids

            elif use_real_diffusion:
                # Real diffusion training with on-the-fly encoding
                pixel_values = batch["pixel_values"]
                captions = batch["caption"]

                with torch.no_grad():
                    # Encode images to latents (convert to VAE's dtype)
                    latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Encode prompts
                    if text_encoder_1 is not None and text_encoder_2 is not None:
                        prompt_embeds, pooled_embeds = encode_prompts(
                            captions,
                            text_encoder_1,
                            text_encoder_2,
                            tokenizer_1,
                            tokenizer_2,
                            device,
                        )
                        # SDXL uses pooled embeddings as added_cond_kwargs
                        added_cond_kwargs = {"text_embeds": pooled_embeds}
                        # Add time_ids (original size, crops, target size)
                        time_ids = batch["time_ids"].to(device)
                        added_cond_kwargs["time_ids"] = time_ids
                    else:
                        # Fallback: use unconditional embeddings
                        prompt_embeds = torch.zeros(
                            latents.shape[0],
                            77,
                            2048,  # SDXL's combined embedding dimension
                            device=device,
                        )
                        added_cond_kwargs = None

            # Perform diffusion training step (common for both cached and non-cached)
            if batch_is_cached or use_real_diffusion:
                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()

                # Add noise to latents
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                if added_cond_kwargs is not None:
                    noise_pred = model(
                        noisy_latents,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                else:
                    noise_pred = model(noisy_latents, timesteps, prompt_embeds).sample

                # Compute diffusion loss
                if config.min_snr_gamma is not None and config.min_snr_gamma > 0:
                    # Min-SNR weighting (epsilon prediction)
                    snr = alphas_cumprod[timesteps]
                    snr = snr / (1 - alphas_cumprod[timesteps])
                    gamma = torch.full_like(snr, config.min_snr_gamma)
                    weights = torch.minimum(snr, gamma) / snr
                    per_sample = torch.nn.functional.mse_loss(
                        noise_pred, noise, reduction="none"
                    ).mean(dim=tuple(range(1, noise_pred.ndim)))
                    loss = (per_sample * weights).mean()
                else:
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

            else:
                raise RuntimeError(
                    "Cannot run training without cached latents or a VAE/text encoder pipeline."
                )

            # Normalize loss by gradient accumulation steps
            loss = loss / config.grad_accum

            # Backward pass
            loss.backward()

            # Update weights every grad_accum steps
            if (batch_idx + 1) % config.grad_accum == 0:
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Accumulate loss on GPU (detached to avoid keeping computation graph)
                accumulated_losses.append((loss * config.grad_accum).detach())

                # Log to TensorBoard every log_every steps to reduce GPU sync overhead
                should_log = global_step % config.log_every == 0
                if should_log:
                    # Compute mean loss on GPU, then sync once
                    mean_loss = torch.stack(accumulated_losses).mean().item()
                    writer.add_scalar("train/loss", mean_loss, global_step)
                    accumulated_losses.clear()

                    # Log other metrics
                    if lr_scheduler is not None:
                        effective_lr = lr_scheduler.get_last_lr()[0]
                    else:
                        effective_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("train/lr", effective_lr, global_step)
                    writer.add_scalar("train/epoch", current_epoch, global_step)

                    step_duration = time.perf_counter() - step_start_time
                    log_perf_metrics(
                        writer=writer,
                        global_step=global_step,
                        step_time=step_duration,
                        effective_batch_size=config.effective_batch_size,
                        device=device,
                    )

                    # Update progress bar with loss details
                    pbar.set_postfix(
                        {
                            "loss": f"{mean_loss:.4f}",
                            "epoch": current_epoch,
                        }
                    )

                # Always update progress bar counter (no sync needed for this)
                pbar.update(1)

                # Save checkpoint periodically
                if global_step % config.sample_every == 0:
                    checkpoint_start = time.perf_counter()
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        checkpoint_dir=dirs["checkpoints"],
                        lora_rank=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        text_encoder_1=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        adapter_type=config.adapter,
                        unet_adapter=unet_adapter,
                        te1_adapter=te1_adapter,
                        te2_adapter=te2_adapter,
                        base_model=base_model,
                    )
                    checkpoint_duration = time.perf_counter() - checkpoint_start
                    writer.add_scalar("perf/checkpoint_time_sec", checkpoint_duration, global_step)
                    last_checkpoint_step = global_step

                    # Run validation sampling
                    if config.sample_prompts is not None and vae is not None:
                        run_validation_samples(
                            unet=model,
                            vae=vae,
                            text_encoder_1=text_encoder_1,
                            text_encoder_2=text_encoder_2,
                            tokenizer_1=tokenizer_1,
                            tokenizer_2=tokenizer_2,
                            config=config,
                            global_step=global_step,
                            samples_dir=dirs["samples"],
                            writer=writer,
                            device=device,
                        )

                # Check if we've reached the target number of steps
                if global_step >= config.steps:
                    break

                # Reset step timer after logging/checkpointing work
                step_start_time = time.perf_counter()

            batch_idx += 1

        # Break outer loop if we've finished
        if global_step >= config.steps:
            break

    # Final checkpoint
    already_saved = last_checkpoint_step == global_step
    checkpoint_start = time.perf_counter()
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        global_step=global_step,
        checkpoint_dir=dirs["checkpoints"],
        is_final=True,
        skip_step_save=already_saved,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        adapter_type=config.adapter,
        unet_adapter=unet_adapter,
        te1_adapter=te1_adapter,
        te2_adapter=te2_adapter,
        base_model=base_model,
    )
    checkpoint_duration = time.perf_counter() - checkpoint_start
    writer.add_scalar("perf/checkpoint_time_sec", checkpoint_duration, global_step)

    pbar.close()
    print(f"\nTraining complete! Final step: {global_step}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    checkpoint_dir: Path,
    lr_scheduler: LRScheduler | None = None,
    is_final: bool = False,
    skip_step_save: bool = False,
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
    text_encoder_1: nn.Module | None = None,
    text_encoder_2: nn.Module | None = None,
    adapter_type: str = "lora",
    unet_adapter=None,
    te1_adapter=None,
    te2_adapter=None,
    base_model: str | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        model: UNet model to save
        optimizer: Optimizer to save
        global_step: Current training step
        checkpoint_dir: Directory to save checkpoints
        is_final: Whether this is the final checkpoint
        skip_step_save: Skip writing the step checkpoint if it already exists
        lora_rank: LoRA rank for metadata
        lora_alpha: LoRA alpha for metadata
        text_encoder_1: Optional text encoder 1 with adapter
        text_encoder_2: Optional text encoder 2 with adapter
        adapter_type: Adapter backend ('lora' or 'lycoris')
        unet_adapter: Optional UNet adapter object (for LyCORIS)
        te1_adapter: Optional text_encoder_1 adapter object (for LyCORIS)
        te2_adapter: Optional text_encoder_2 adapter object (for LyCORIS)
        base_model: Base model checkpoint path or HF model ID
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
        "global_step": global_step,
        "adapter_type": adapter_type,
    }

    # Save base model reference for resume
    if base_model is not None:
        checkpoint["base_model"] = base_model

    # Save adapter state dicts based on adapter type
    if adapter_type == "lycoris":
        # For LyCORIS, save adapter state dicts directly
        if unet_adapter is not None:
            checkpoint["unet_adapter_state_dict"] = unet_adapter.state_dict()
        if te1_adapter is not None:
            checkpoint["te1_adapter_state_dict"] = te1_adapter.state_dict()
        if te2_adapter is not None:
            checkpoint["te2_adapter_state_dict"] = te2_adapter.state_dict()
    else:
        # For LoRA, extract LoRA weights from model states
        from .model import extract_lora_state_dict

        checkpoint["model_state_dict"] = extract_lora_state_dict(model)
        if text_encoder_1 is not None:
            checkpoint["text_encoder_1_state_dict"] = extract_lora_state_dict(text_encoder_1)
        if text_encoder_2 is not None:
            checkpoint["text_encoder_2_state_dict"] = extract_lora_state_dict(text_encoder_2)

    # Always save full checkpoints as .pt for training resumes
    checkpoint_path = checkpoint_dir / f"step_{global_step:06d}.pt"
    should_save_step = not (skip_step_save and checkpoint_path.exists())
    if should_save_step:
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    if is_final:
        final_checkpoint = checkpoint_dir / "final.pt"
        torch.save(checkpoint, final_checkpoint)
        print(f"Saved final checkpoint: {final_checkpoint}")

    # Save LoRA-only weights in safetensors format for ComfyUI, etc.
    # Only export a safetensors LoRA file for the final checkpoint
    if is_final and adapter_type == "lora":
        lora_path = checkpoint_dir / "final_lora.safetensors"
        try:
            from safetensors.torch import save_file

            from .model import build_lora_metadata, extract_lora_state_dict, infer_lora_hparams

            # Extract LoRA weights and alpha values from all models
            lora_state = extract_lora_state_dict(model)
            unet_keys = len(lora_state)

            if text_encoder_1 is not None:
                te1_state = extract_lora_state_dict(text_encoder_1)
                lora_state.update(te1_state)
                print(f"  Including {len(te1_state)} text_encoder_1 LoRA tensors")
            else:
                print("  Skipping text_encoder_1 (not loaded)")

            if text_encoder_2 is not None:
                te2_state = extract_lora_state_dict(text_encoder_2)
                lora_state.update(te2_state)
                print(f"  Including {len(te2_state)} text_encoder_2 LoRA tensors")
            else:
                print("  Skipping text_encoder_2 (not loaded)")

            print(f"  Exporting {unet_keys} UNet LoRA tensors")

            if lora_state:
                # Convert to ComfyUI format
                converted = convert_lora_state(lora_state)

                # Build metadata
                inferred_rank, inferred_alpha = infer_lora_hparams(model)
                rank = lora_rank if lora_rank is not None else inferred_rank
                alpha = lora_alpha if lora_alpha is not None else inferred_alpha
                metadata = build_lora_metadata(rank, alpha)

                # Save to file
                save_file(converted, str(lora_path), metadata=metadata)
                print(f"Saved LoRA weights: {lora_path} ({len(converted)} tensors)")
            else:
                print("Warning: No LoRA weights found to export.")
        except Exception as e:
            print(f"Warning: Failed to save LoRA safetensors ({e})")
    elif is_final and adapter_type == "lycoris":
        # Convert LyCORIS checkpoint to safetensors using converter
        lycoris_path = checkpoint_dir / "final_lycoris.safetensors"
        try:
            from lora_converter.converter import convert_lycoris_checkpoint

            # Convert the just-saved checkpoint
            convert_lycoris_checkpoint(
                checkpoint_path,
                lycoris_path,
                overwrite=True,
            )
        except Exception as e:
            print(f"Warning: Failed to convert LyCORIS checkpoint to safetensors ({e})")
    elif is_final:
        print(f"Skipping adapter safetensors export for adapter_type={adapter_type}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    lr_scheduler: LRScheduler | None = None,
    text_encoder_1: nn.Module | None = None,
    text_encoder_2: nn.Module | None = None,
    unet_adapter=None,
    te1_adapter=None,
    te2_adapter=None,
) -> int:
    """Load model/optimizer state from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        model: UNet model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device for loading
        text_encoder_1: Optional text encoder 1 to load weights into
        text_encoder_2: Optional text encoder 2 to load weights into
        unet_adapter: Optional UNet adapter to load state into (LyCORIS)
        te1_adapter: Optional text_encoder_1 adapter to load state into (LyCORIS)
        te2_adapter: Optional text_encoder_2 adapter to load state into (LyCORIS)

    Returns:
        global_step stored in the checkpoint (0 if missing)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    adapter_type = checkpoint.get("adapter_type", "lora")

    # Load adapter state dicts based on adapter type
    if adapter_type == "lycoris":
        # For LyCORIS, load adapter state dicts
        if unet_adapter is not None and "unet_adapter_state_dict" in checkpoint:
            unet_adapter.load_state_dict(checkpoint["unet_adapter_state_dict"])
            print("  Loaded UNet LyCORIS adapter weights")

        if te1_adapter is not None and "te1_adapter_state_dict" in checkpoint:
            te1_adapter.load_state_dict(checkpoint["te1_adapter_state_dict"])
            print("  Loaded text_encoder_1 LyCORIS adapter weights")

        if te2_adapter is not None and "te2_adapter_state_dict" in checkpoint:
            te2_adapter.load_state_dict(checkpoint["te2_adapter_state_dict"])
            print("  Loaded text_encoder_2 LyCORIS adapter weights")
    else:
        # For LoRA, load extracted LoRA weights into models
        if "model_state_dict" in checkpoint:
            # Use strict=False since we're only loading LoRA weights, not full model
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("  Loaded UNet LoRA weights")

        # Load text encoder LoRA weights if available
        if text_encoder_1 is not None and "text_encoder_1_state_dict" in checkpoint:
            text_encoder_1.load_state_dict(checkpoint["text_encoder_1_state_dict"], strict=False)
            print("  Loaded text_encoder_1 LoRA weights")

        if text_encoder_2 is not None and "text_encoder_2_state_dict" in checkpoint:
            text_encoder_2.load_state_dict(checkpoint["text_encoder_2_state_dict"], strict=False)
            print("  Loaded text_encoder_2 LoRA weights")

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("  Loaded optimizer state")
    if lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        print("  Loaded lr scheduler state")

    return int(checkpoint.get("global_step", 0))
