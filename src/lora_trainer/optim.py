"""Optimizer factory for spec parsing (adamw/8bit, lion/8bit, adafactor, prodigy)."""

import torch
from torch.optim.lr_scheduler import LRScheduler

from .utils import parse_spec


def parse_optimizer_spec(spec: str) -> tuple[str, dict]:
    """Parse optimizer string like 'prodigy(lr=1e-4, weight_decay=0)'."""
    name, kwargs = parse_spec(spec)
    return name, kwargs


def build_optimizer(params, spec: str, base_lr: float) -> torch.optim.Optimizer:
    """Build optimizer from spec string."""
    params = list(params)
    if not params:
        raise ValueError("No parameters provided to optimizer")
    name, kwargs = parse_optimizer_spec(spec)

    # Common default lr unless overridden
    kwargs.setdefault("lr", base_lr)

    if name == "adamw":
        if "fused" not in kwargs:
            try:
                if any(p.device.type == "cuda" for p in params):
                    kwargs["fused"] = True
            except Exception:
                pass
        return torch.optim.AdamW(params, **kwargs)

    if name == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except Exception as e:
            raise ImportError("Install bitsandbytes to use adamw8bit optimizer") from e
        return AdamW8bit(params, **kwargs)

    if name == "lion":
        try:
            from bitsandbytes.optim import Lion
        except Exception as e:
            raise ImportError("Install bitsandbytes to use lion optimizer") from e
        return Lion(params, **kwargs)

    if name == "lion8bit":
        try:
            from bitsandbytes.optim import Lion8bit
        except Exception as e:
            raise ImportError("Install bitsandbytes to use lion8bit optimizer") from e
        return Lion8bit(params, **kwargs)

    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except Exception as e:
            raise ImportError("Install/upgrade transformers to use adafactor optimizer") from e
        return Adafactor(params, **kwargs)

    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except Exception as e:
            raise ImportError("Install prodigyopt to use prodigy optimizer") from e
        return Prodigy(params, **kwargs)

    supported = [
        "adamw",
        "adamw8bit",
        "lion",
        "lion8bit",
        "adafactor",
        "prodigy",
    ]
    raise ValueError(f"Unknown optimizer '{name}'. Supported: {', '.join(supported)}")


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    num_cycles: int = 1,
    power: float = 1.0,
) -> LRScheduler:
    """Build a training LR scheduler using transformers' get_scheduler."""
    try:
        from transformers.optimization import get_scheduler
    except Exception as e:
        raise ImportError("Install/upgrade transformers to use LR schedulers") from e

    scheduler_name = name.strip().lower()
    supported = {
        "constant",
        "constant_with_warmup",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
    }
    if scheduler_name not in supported:
        raise ValueError(
            f"Unknown lr_scheduler '{name}'. Supported: {', '.join(sorted(supported))}"
        )

    kwargs = {
        "optimizer": optimizer,
        "num_warmup_steps": num_warmup_steps,
        "num_training_steps": num_training_steps,
    }
    if scheduler_name == "cosine_with_restarts":
        kwargs["num_cycles"] = num_cycles
    if scheduler_name == "polynomial":
        kwargs["power"] = power

    return get_scheduler(scheduler_name, **kwargs)
