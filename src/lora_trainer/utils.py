"""Small shared utility functions."""

import ast
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_spec(spec: str) -> tuple[str, dict]:
    """Parse function-like specs like 'name(key=value, ...)'."""
    spec = spec.strip()
    if "(" not in spec:
        return spec.lower(), {}

    try:
        expr = ast.parse(spec, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid spec '{spec}'") from exc

    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        raise ValueError(f"Invalid spec '{spec}'")

    if expr.args:
        raise ValueError(f"Positional arguments are not supported in spec '{spec}'")

    name = expr.func.id.strip().lower()
    kwargs: dict = {}
    for kw in expr.keywords:
        if kw.arg is None:
            raise ValueError(f"Keyword expansion is not supported in spec '{spec}'")
        try:
            kwargs[kw.arg.strip()] = ast.literal_eval(kw.value)
        except Exception as exc:
            raise ValueError(f"Invalid value for '{kw.arg}' in spec '{spec}'") from exc

    return name, kwargs


def resolve_lr_scheduler_spec(
    spec: str,
    *,
    warmup_steps: int,
    num_cycles: int,
    power: float,
) -> tuple[str, int, int, float]:
    """Resolve LR scheduler spec into name and scheduler args."""
    name, kwargs = parse_spec(spec)

    if "warmup_steps" in kwargs:
        warmup_steps = int(kwargs.pop("warmup_steps"))
    if "num_warmup_steps" in kwargs:
        warmup_steps = int(kwargs.pop("num_warmup_steps"))
    if "cycles" in kwargs:
        num_cycles = int(kwargs.pop("cycles"))
    if "num_cycles" in kwargs:
        num_cycles = int(kwargs.pop("num_cycles"))
    if "power" in kwargs:
        power = float(kwargs.pop("power"))

    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unknown lr scheduler kwargs: {unknown}")

    return name, warmup_steps, num_cycles, power


def resolve_adapter_spec(
    spec: str,
    *,
    lora_rank: int | None,
    lora_alpha: float | None,
    lycoris_algo: str,
    lycoris_dim: int | None,
    lycoris_alpha: float | None,
    lycoris_factor: int,
    lycoris_dropout: float | None,
) -> dict:
    """Resolve adapter spec strings into adapter settings."""
    name, kwargs = parse_spec(spec)

    def _pop_any(keys: tuple[str, ...], default):
        for key in keys:
            if key in kwargs:
                return kwargs.pop(key)
        return default

    if name == "lora":
        rank = _pop_any(("rank", "dim"), lora_rank)
        alpha = _pop_any(("alpha",), lora_alpha)
        if rank is not None:
            rank = int(rank)
        if alpha is not None:
            alpha = float(alpha)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(f"Unknown lora kwargs: {unknown}")
        return {
            "adapter": "lora",
            "lora_rank": rank,
            "lora_alpha": alpha,
        }

    if name in {"lycoris", "locon"}:
        algo = "locon" if name == "locon" else str(_pop_any(("algo",), lycoris_algo))
        dim_default = lycoris_dim if lycoris_dim is not None else (lora_rank or 16)
        alpha_default = lycoris_alpha if lycoris_alpha is not None else (lora_alpha or 16.0)
        dim = int(_pop_any(("dim", "rank", "linear_dim"), dim_default))
        alpha = float(_pop_any(("alpha", "linear_alpha"), alpha_default))
        factor = int(_pop_any(("factor",), lycoris_factor))
        dropout = _pop_any(("dropout",), lycoris_dropout)
        if dropout is not None:
            dropout = float(dropout)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(f"Unknown lycoris kwargs: {unknown}")
        return {
            "adapter": "lycoris",
            "lycoris_algo": algo,
            "lycoris_dim": dim,
            "lycoris_alpha": alpha,
            "lycoris_factor": factor,
            "lycoris_dropout": dropout,
        }

    raise ValueError(f"Unknown adapter spec '{spec}'")


def save_images(images: torch.Tensor, path: Path) -> None:
    """Save a [B, 3, H, W] float32 [0, 1] tensor batch to disk.

    Single-image batches write to path directly; multi-image batches write
    path/stem_0.ext, stem_1.ext, ...
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    for i, img in enumerate(images):
        p = path if images.shape[0] == 1 else path.with_name(f"{path.stem}_{i}{suffix}")
        arr = img.to(torch.float32).cpu().permute(1, 2, 0).numpy()
        arr = (arr * 255).round().clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(p)
