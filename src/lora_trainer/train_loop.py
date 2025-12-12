"""Core training loop with progress tracking and checkpointing."""

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
    config,  # TrainingConfig
    dirs: dict[str, Path],
    writer: SummaryWriter,
    device: str = "cuda",
    vae=None,
    text_encoder_1=None,
    text_encoder_2=None,
    tokenizer_1=None,
    tokenizer_2=None,
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

    # Determine if we're using real diffusion or dummy mode
    use_real_diffusion = vae is not None

    if use_real_diffusion:
        vae = vae.to(device)
        vae.eval()
        if text_encoder_1 is not None:
            text_encoder_1 = text_encoder_1.to(device)
            text_encoder_1.eval()
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(device)
            text_encoder_2.eval()

    global_step = 0
    current_epoch = 0

    # Progress bar for global steps
    pbar = tqdm(total=config.steps, desc="Training", unit="step")

    while global_step < config.steps:
        current_epoch += 1

        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            captions = batch["caption"]

            if use_real_diffusion:
                # Real diffusion training
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
                        # For now, use default values for 1024x1024
                        time_ids = torch.tensor(
                            [[1024, 1024, 0, 0, 1024, 1024]], device=device
                        ).repeat(latents.shape[0], 1)
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
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            else:
                # Dummy mode for testing without VAE
                # Simple MSE loss on model output
                dummy_target = torch.zeros(
                    config.batch_size, 4, config.image_size // 8, config.image_size // 8
                ).to(device)
                dummy_timesteps = torch.zeros(config.batch_size, device=device).long()
                dummy_embeds = torch.zeros(config.batch_size, 77, 2048, device=device)
                output = model(pixel_values, dummy_timesteps, dummy_embeds).sample
                loss = torch.nn.functional.mse_loss(output, dummy_target)

            # Normalize loss by gradient accumulation steps
            loss = loss / config.grad_accum

            # Backward pass
            loss.backward()

            # Update weights every grad_accum steps
            if (batch_idx + 1) % config.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # Log to TensorBoard
                writer.add_scalar("train/loss", loss.item() * config.grad_accum, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/epoch", current_epoch, global_step)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item() * config.grad_accum:.4f}",
                        "epoch": current_epoch,
                    }
                )

                # Save checkpoint periodically
                if global_step % config.sample_every == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        checkpoint_dir=dirs["checkpoints"],
                    )

                # Check if we've reached the target number of steps
                if global_step >= config.steps:
                    break

        # Break outer loop if we've finished
        if global_step >= config.steps:
            break

    # Final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=global_step,
        checkpoint_dir=dirs["checkpoints"],
        is_final=True,
    )

    pbar.close()
    print(f"\nTraining complete! Final step: {global_step}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    checkpoint_dir: Path,
    is_final: bool = False,
) -> None:
    """Save a training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        global_step: Current training step
        checkpoint_dir: Directory to save checkpoints
        is_final: Whether this is the final checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
    }

    if is_final:
        checkpoint_path = checkpoint_dir / "final.pt"
    else:
        checkpoint_path = checkpoint_dir / f"step_{global_step:06d}.pt"

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
