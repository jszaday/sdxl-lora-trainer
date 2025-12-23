"""Optimizer factory for spec parsing (adamw/8bit, lion/8bit, adafactor, prodigy)."""

import ast

import torch
from torch.optim import Optimizer


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
            return AdafactorWithTracking(params, **kwargs)
        except Exception as e:
            raise ImportError("Install/upgrade transformers to use adafactor optimizer") from e

    if name == "prodigy":
        try:
            import prodigyopt  # noqa: F401
        except Exception as e:
            raise ImportError("Install prodigyopt to use prodigy optimizer") from e
        return ProdigyWithTracking(params, **kwargs)

    supported = [
        "adamw",
        "adamw8bit",
        "lion",
        "lion8bit",
        "adafactor",
        "prodigy",
    ]
    raise ValueError(f"Unknown optimizer '{name}'. Supported: {', '.join(supported)}")


class ProdigyWithTracking(Optimizer):
    """Prodigy wrapper that records effective step size per step."""

    def __init__(self, params, *args, **kwargs):
        from prodigyopt import Prodigy

        # Track last effective lr
        self.learning_rate: float | None = None
        self._state_moved = False

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
        self.inner.step(closure)
        self._record_learning_rate()
        if not self._state_moved:
            self._move_state_to_param_device()
        return self.learning_rate

    def zero_grad(self, set_to_none: bool = False):
        return self.inner.zero_grad(set_to_none=set_to_none)

    def _move_state_to_param_device(self):
        """Ensure Prodigy state tensors sit on the same device as model params."""
        for group in self.inner.param_groups:
            if not group.get("params"):
                continue
            device = group["params"][0].device
            for key in ("running_d_numerator", "running_d_denom"):
                if key in group:
                    group[key] = group[key].to(device)
        self._state_moved = True

    def _record_learning_rate(self):
        """Capture an effective lr from Prodigy's param group state."""
        for group in self.inner.param_groups:
            if not group.get("params"):
                continue
            base_lr = group.get("lr")
            d_scale = group.get("d")
            if base_lr is None or d_scale is None:
                continue
            self.learning_rate = float(base_lr) * float(d_scale)
            break


class AdafactorWithTracking(Optimizer):
    """Adafactor wrapper that records the effective lr used per step."""

    def __init__(self, params, *args, **kwargs):
        from transformers.optimization import Adafactor

        self.inner = Adafactor(params, *args, **kwargs)
        self.learning_rate: float | None = None

    @property
    def param_groups(self):
        return self.inner.param_groups

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state_dict):
        return self.inner.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.inner.step(closure)
        self._record_lr()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        return self.inner.zero_grad(set_to_none=set_to_none)

    def _record_lr(self):
        """Capture the lr Adafactor computed for logging."""
        for group in self.inner.param_groups:
            if not group.get("params"):
                continue
            param = group["params"][0]
            state = self.inner.state.get(param)
            if state is None:
                continue
            try:
                lr = self.inner._get_lr(group, state)
            except Exception:
                continue
            self.learning_rate = lr
            break
