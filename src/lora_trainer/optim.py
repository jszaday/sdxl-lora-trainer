"""Optimizer factory with simple spec parsing (adamw, lion, prodigy)."""

import ast

import torch


def parse_optimizer_spec(spec: str) -> tuple[str, dict]:
    """Parse optimizer string like 'prodigy(lr=1e-4, weight_decay=0)'."""
    spec = spec.strip()
    if "(" not in spec:
        return spec.lower(), {}

    name, rest = spec.split("(", 1)
    name = name.strip().lower()
    rest = rest.rsplit(")", 1)[0]

    kwargs: dict = {}
    for part in rest.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid optimizer argument '{part}' in spec '{spec}'")
        key, val = part.split("=", 1)
        kwargs[key.strip()] = ast.literal_eval(val.strip())

    return name, kwargs


def build_optimizer(params, spec: str, base_lr: float) -> torch.optim.Optimizer:
    """Build optimizer from spec string."""
    name, kwargs = parse_optimizer_spec(spec)

    # Common default lr unless overridden
    kwargs.setdefault("lr", base_lr)

    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)

    if name == "lion":
        try:
            from torch_optimizer import Lion
        except Exception as e:
            raise ImportError("Install torch_optimizer to use lion optimizer") from e
        return Lion(params, **kwargs)

    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except Exception as e:
            raise ImportError("Install prodigyopt to use prodigy optimizer") from e
        return Prodigy(params, **kwargs)

    raise ValueError(f"Unknown optimizer '{name}'. Supported: adamw, lion, prodigy")
