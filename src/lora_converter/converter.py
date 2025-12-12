"""Utilities to convert LoRA checkpoints into ComfyUI-friendly safetensors."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LORA_MARKER = ".lora_"


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return its state dict."""
    if checkpoint_path.suffix.lower() == ".safetensors":
        return load_file(str(checkpoint_path))

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model_state_dict" in state:
            return state["model_state_dict"]
        if "state_dict" in state:
            return state["state_dict"]
        return state

    # If torch.load failed to give us a dict but the file is safetensors without extension
    try:
        return load_file(str(checkpoint_path))
    except Exception:
        pass

    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path} (type={type(state)})")


def _convert_key(key: str) -> str:
    """Turn a module path like 'down_blocks.0.to_q.lora_down.weight' into ComfyUI format."""
    if LORA_MARKER not in key:
        raise ValueError(f"Key does not look like a LoRA tensor: {key}")
    if key.startswith("lora_unet_"):
        return key

    module_path, suffix = key.split(LORA_MARKER, 1)
    module_path = module_path.replace(".", "_")
    return f"lora_unet_{module_path}.lora_{suffix}"


def convert_lora_state(lora_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert raw LoRA tensors into ComfyUI key space."""
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in lora_state.items():
        target = _convert_key(key)
        if target in converted:
            raise ValueError(f"Duplicate target key after conversion: {target}")
        converted[target] = tensor.detach().cpu()
    return converted


def convert_checkpoint(input_path: Path, output_path: Path, *, overwrite: bool = False) -> Path:
    """Convert a .pt checkpoint to ComfyUI-style safetensors."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    state = _load_state_dict(input_path)
    lora_state = {k: v for k, v in state.items() if LORA_MARKER in k}
    if not lora_state:
        raise ValueError(f"No LoRA tensors found in {input_path}")

    converted = convert_lora_state(lora_state)
    save_file(converted, str(output_path))
    return output_path
