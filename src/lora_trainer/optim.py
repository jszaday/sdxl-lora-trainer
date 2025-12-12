"""Optimizer factory with simple spec parsing (adamw, lion, prodigy)."""

import ast

import torch
from torch.optim import Optimizer


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
            import prodigyopt  # noqa: F401
        except Exception as e:
            raise ImportError("Install prodigyopt to use prodigy optimizer") from e
        return ProdigyWithTracking(params, **kwargs)

    raise ValueError(f"Unknown optimizer '{name}'. Supported: adamw, lion, prodigy")


class ProdigyWithTracking(Optimizer):
    """Prodigy wrapper that records effective step size per step."""

    def __init__(self, params, *args, **kwargs):
        from prodigyopt import Prodigy

        # Track last effective lr
        self.last_step_size: float | None = None

        # Instantiate inner optimizer
        self.inner = Prodigy(params, *args, **kwargs)

    @property
    def param_groups(self):
        return self.inner.param_groups

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state_dict):
        return self.inner.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        # Prodigy exposes step size via return value
        step_size = self.inner.step(closure)
        self.last_step_size = step_size
        return step_size

    def zero_grad(self, set_to_none: bool = False):
        return self.inner.zero_grad(set_to_none=set_to_none)
