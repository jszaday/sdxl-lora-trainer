"""Latent tensor loading and saving for frozen SDXL inference."""

from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

LATENTS_KEY = "latents"


def load_latents(path: Path, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Load latents from .safetensors or a torch checkpoint."""
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        state = load_safetensors(str(path), device=device)
        if LATENTS_KEY not in state:
            keys = ", ".join(state)
            raise ValueError(f"Latent safetensors must contain '{LATENTS_KEY}'. Found: {keys}")
        latents = state[LATENTS_KEY]
    else:
        value = torch.load(path, map_location=device)
        if isinstance(value, dict):
            if LATENTS_KEY not in value:
                keys = ", ".join(str(key) for key in value)
                raise ValueError(f"Latent checkpoint must contain '{LATENTS_KEY}'. Found: {keys}")
            value = value[LATENTS_KEY]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Unsupported latent checkpoint at {path}")
        latents = value

    return latents.to(device=device, dtype=dtype)


def save_latents(path: Path, latents: torch.Tensor) -> None:
    """Save latents to .safetensors or a torch checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latents_cpu = latents.detach().cpu()
    if path.suffix.lower() == ".safetensors":
        save_safetensors({LATENTS_KEY: latents_cpu}, str(path))
        return
    torch.save({LATENTS_KEY: latents_cpu}, path)
