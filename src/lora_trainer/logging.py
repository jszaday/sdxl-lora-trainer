"""Logging utilities for TensorBoard, run directories, and progress tracking."""

from pathlib import Path
from typing import Any

import torch
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


_nvml_initialized = False
_nvml_available = False


def _init_nvml() -> None:
    global _nvml_initialized, _nvml_available
    if _nvml_initialized:
        return
    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_available = True
    except Exception:
        _nvml_available = False
    finally:
        _nvml_initialized = True


def collect_gpu_metrics(device: torch.device | str | None = None) -> dict[str, float]:
    """Collect basic GPU metrics using NVML if available, else torch.cuda stats."""
    if not torch.cuda.is_available():
        return {}

    dev = torch.device(device) if device is not None else torch.device("cuda")
    if dev.type != "cuda":
        return {}
    index = dev.index if dev.index is not None else torch.cuda.current_device()

    metrics: dict[str, float] = {}

    # torch memory stats
    try:
        metrics["gpu_mem_allocated_bytes"] = float(torch.cuda.memory_allocated(index))
        metrics["gpu_mem_reserved_bytes"] = float(torch.cuda.memory_reserved(index))
    except Exception:
        pass

    # NVML util + total memory if available
    _init_nvml()
    if _nvml_available:
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["gpu_utilization"] = float(util.gpu)
            metrics["gpu_mem_used_bytes"] = float(mem.used)
            metrics["gpu_mem_total_bytes"] = float(mem.total)
        except Exception:
            pass

    return metrics


def log_perf_metrics(
    writer: SummaryWriter,
    global_step: int,
    *,
    step_time: float,
    effective_batch_size: int,
    device: torch.device | str,
) -> None:
    """Log performance metrics to TensorBoard."""
    writer.add_scalar("perf/step_time_sec", step_time, global_step)
    if step_time > 0:
        imgs_per_sec = effective_batch_size / step_time
        writer.add_scalar("perf/images_per_sec", imgs_per_sec, global_step)

    gpu_metrics = collect_gpu_metrics(device)
    for key, value in gpu_metrics.items():
        writer.add_scalar(f"perf/{key}", value, global_step)


def write_config_yaml(workspace: Path, config: Any) -> Path:
    """Write a simple YAML file containing the config summary."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    summary = config.print_summary()
    yaml_body = "\n".join(f"  {line}" for line in summary.splitlines())
    contents = f"summary: |\n{yaml_body}\n"
    config_path = workspace / "config.yaml"
    config_path.write_text(contents, encoding="utf-8")
    return config_path
