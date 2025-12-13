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
    """Turn a module path into ComfyUI format.

    Examples:
        'add_embedding.linear_1.lora_down.weight'
            -> 'lora_unet_add_embedding_linear_1.lora_down.weight'
        'add_embedding.linear_1.alpha'
            -> 'lora_unet_add_embedding_linear_1.alpha'
        'text_model.encoder.layers.0.mlp.fc1.lora_down.weight'
            -> 'lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight'
    """
    # Already in ComfyUI format
    if key.startswith("lora_unet_") or key.startswith("lora_te1_") or key.startswith("lora_te2_"):
        return key

    # Determine if this is a text encoder key (contains text_model or encoder.layers)
    is_te = "text_model" in key
    prefix = "lora_te1" if is_te else "lora_unet"  # Default to TE1 for text encoder keys

    # Handle both .lora_down.weight/.lora_up.weight and .alpha
    if ".lora_down.weight" in key:
        module_path = key[: -len(".lora_down.weight")]
        suffix = "lora_down.weight"
    elif ".lora_up.weight" in key:
        module_path = key[: -len(".lora_up.weight")]
        suffix = "lora_up.weight"
    elif key.endswith(".alpha"):
        module_path = key[: -len(".alpha")]
        suffix = "alpha"
    else:
        # Fallback: check for generic .lora_ marker
        if LORA_MARKER not in key:
            raise ValueError(f"Key does not look like a LoRA tensor: {key}")
        module_path, suffix = key.split(LORA_MARKER, 1)
        suffix = f"lora_{suffix}"

    # Convert dots to underscores in module path
    module_path = module_path.replace(".", "_")

    # Build ComfyUI key
    return f"{prefix}_{module_path}.{suffix}"


def convert_lora_state(lora_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert raw LoRA tensors into ComfyUI key space.

    Handles:
    - UNet LoRA weights (module.lora_down.weight, module.lora_up.weight)
    - Text encoder LoRA weights
    - Alpha values (module.alpha)
    """
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
    # Infer rank from first lora_down tensor if possible
    rank = None
    for key, tensor in converted.items():
        if key.endswith("lora_down.weight") and tensor.ndim >= 2:
            rank = tensor.shape[0]
            break
    metadata = {
        "format": "pt",
        "network_dim": str(rank) if rank is not None else "",
        "network_alpha": str(rank) if rank is not None else "",
        "generator": "lora_trainer",
    }
    save_file(converted, str(output_path), metadata=metadata)
    return output_path
