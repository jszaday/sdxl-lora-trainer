"""Core training loop with progress tracking and checkpointing."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config,  # TrainingConfig
    dirs: dict[str, Path],
    writer: SummaryWriter,
    device: str = "cuda",
) -> None:
    """Run the complete training loop.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        config: TrainingConfig instance
        dirs: Dictionary of output directories
        writer: TensorBoard writer
        device: Device to train on
    """
    model.train()
    model = model.to(device)

    global_step = 0
    current_epoch = 0

    # Progress bar for global steps
    pbar = tqdm(total=config.steps, desc="Training", unit="step")

    # Dummy target for Phase 1 (simplified loss)
    # In Phase 2, this will be replaced with proper diffusion loss
    dummy_target = torch.zeros(config.batch_size, 3, 1, 1).to(device)

    while global_step < config.steps:
        current_epoch += 1

        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)

            # Phase 1: Simple MSE loss on dummy output
            # Phase 2: Will use proper diffusion noise prediction loss
            output = model(pixel_values)
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
