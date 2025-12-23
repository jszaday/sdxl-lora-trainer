"""Optimizer factory for spec parsing (adamw/8bit, lion/8bit, adafactor, prodigy)."""

import ast

import torch
from torch.optim.lr_scheduler import LRScheduler


def parse_optimizer_spec(spec: str) -> tuple[str, dict]:
    """Parse optimizer string like 'prodigy(lr=1e-4, weight_decay=0)'."""
    spec = spec.strip()
    if "(" not in spec:
        return spec.lower(), {}

    try:
        expr = ast.parse(spec, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid optimizer spec '{spec}'") from exc

    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        raise ValueError(f"Invalid optimizer spec '{spec}'")

    if expr.args:
        raise ValueError(f"Positional arguments are not supported in optimizer spec '{spec}'")

    name = expr.func.id.strip().lower()
    kwargs: dict = {}
    for kw in expr.keywords:
        if kw.arg is None:
            raise ValueError(f"Keyword expansion is not supported in optimizer spec '{spec}'")
        try:
            kwargs[kw.arg.strip()] = ast.literal_eval(kw.value)
        except Exception as exc:
            raise ValueError(f"Invalid optimizer value for '{kw.arg}' in spec '{spec}'") from exc

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
