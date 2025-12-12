"""Logging utilities for TensorBoard, run directories, and progress tracking."""

from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def create_run_dirs(workspace: Path) -> dict[str, Path]:
    """Create output directories for a training run.

    Args:
        workspace: Root workspace directory for this training run

    Returns:
        Dictionary with paths to subdirectories:
        - 'root': workspace root
        - 'tb': TensorBoard logs
        - 'checkpoints': model checkpoints
        - 'samples': validation sample images
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    subdirs = {
        "root": workspace,
        "tb": workspace / "tb",
        "checkpoints": workspace / "checkpoints",
        "samples": workspace / "samples",
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    return subdirs


def init_tensorboard(logdir: Path) -> SummaryWriter:
    """Initialize TensorBoard SummaryWriter.

    Args:
        logdir: Directory for TensorBoard event files

    Returns:
        Configured SummaryWriter instance
    """
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(logdir))


def log_hparams(writer: SummaryWriter, config: Any) -> None:
    """Log hyperparameters to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        config: TrainingConfig instance
    """
    # Extract relevant hyperparameters as a flat dict
    hparams = {
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "effective_batch": config.effective_batch_size,
        "steps": config.steps,
        "num_epochs": config.num_epochs,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "image_size": config.image_size,
        "scheduler": config.scheduler,
        "sampler": config.sampler,
        "cfg": config.cfg,
        "sampler_steps": config.sampler_steps,
        "seed": config.seed,
    }

    writer.add_hparams(hparams, {})
