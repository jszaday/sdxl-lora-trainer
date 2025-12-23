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


# LyCORIS weight patterns - used to identify LyCORIS layers in checkpoints
LYCORIS_MARKERS = [
    "lora_up.",  # Standard LoRA-style (LoConModule)
    "lora_down.",
    "hada_w1_a.",  # LoHa (Hadamard Product)
    "hada_w1_b.",
    "hada_w2_a.",
    "hada_w2_b.",
    "lokr_w1.",  # LoKr (Kronecker Product)
    "lokr_w2.",
    "lokr_w1_a.",
    "lokr_w1_b.",
    "lokr_w2_a.",
    "lokr_w2_b.",
    "oft_blocks.",  # OFT (Orthogonal Finetuning)
    "alpha",  # Alpha values for all types
]

# Known LyCORIS/LoRA suffixes for ComfyUI-compatible keys.
LYCORIS_SUFFIXES = [
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_mid.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_t2",
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".hada_t1",
    ".hada_t2",
    ".oft_blocks",
    ".rescale",
    ".a1.weight",
    ".a2.weight",
    ".b1.weight",
    ".b2.weight",
    ".alpha",
    ".dora_scale",
    ".w_norm",
    ".b_norm",
    ".diff_b",
    ".diff",
    ".set_weight",
]


def _is_lycoris_key(key: str) -> bool:
    """Check if a state dict key belongs to a LyCORIS layer."""
    return any(marker in key for marker in LYCORIS_MARKERS)


def _strip_adapter_prefix(key: str) -> str:
    for prefix in ("unet.", "text_encoder_1.", "text_encoder_2."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _convert_lycoris_key(key: str, prefix: str) -> str:
    """Convert a LyCORIS adapter key to ComfyUI key space."""
    if key.startswith(("lora_unet_", "lora_te1_", "lora_te2_", "lycoris_")):
        return key

    key = _strip_adapter_prefix(key)
    suffix = None
    for candidate in LYCORIS_SUFFIXES:
        if key.endswith(candidate):
            suffix = candidate
            break

    if suffix is None:
        raise ValueError(f"Key does not look like a LyCORIS tensor: {key}")

    module_path = key[: -len(suffix)].replace(".", "_")
    return f"{prefix}_{module_path}{suffix}"


def convert_lycoris_checkpoint(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a .pt checkpoint with LyCORIS weights to safetensors.

    Extracts LyCORIS weights from a full training checkpoint and saves them
    to a single safetensors file in native LyCORIS format (no conversion).
    Combines UNet and text encoder weights with prefixes.

    Args:
        input_path: Path to input .pt checkpoint (full training checkpoint)
        output_path: Path to output .safetensors file
        overwrite: Whether to overwrite existing output file

    Returns:
        Path to the created safetensors file

    Raises:
        FileExistsError: If output exists and overwrite=False
        ValueError: If no LyCORIS tensors found in checkpoint

    Example:
        >>> convert_lycoris_checkpoint(
        ...     Path("checkpoints/checkpoint_step_1000.pt"),
        ...     Path("checkpoints/lycoris_step_1000.safetensors")
        ... )
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    # Load checkpoint (handles both .pt and .safetensors)
    if input_path.suffix == ".pt":
        checkpoint = torch.load(input_path, map_location="cpu")
    else:
        checkpoint = load_file(str(input_path))

    # Extract LyCORIS tensors
    lycoris_state = {}

    # Check for adapter state dicts (new format - adapters saved separately)
    if "unet_adapter_state_dict" in checkpoint:
        # New format: adapter states saved separately
        unet_adapter = checkpoint["unet_adapter_state_dict"]
        for key, tensor in unet_adapter.items():
            converted_key = _convert_lycoris_key(key, "lycoris")
            lycoris_state[converted_key] = tensor.detach().cpu()

        if "te1_adapter_state_dict" in checkpoint:
            te1_adapter = checkpoint["te1_adapter_state_dict"]
            for key, tensor in te1_adapter.items():
                converted_key = _convert_lycoris_key(key, "lora_te1")
                lycoris_state[converted_key] = tensor.detach().cpu()

        if "te2_adapter_state_dict" in checkpoint:
            te2_adapter = checkpoint["te2_adapter_state_dict"]
            for key, tensor in te2_adapter.items():
                converted_key = _convert_lycoris_key(key, "lora_te2")
                lycoris_state[converted_key] = tensor.detach().cpu()
    else:
        # Old format: scan model_state_dict for embedded LyCORIS weights
        if isinstance(checkpoint, dict):
            state = checkpoint.get("model_state_dict", checkpoint)
        else:
            state = checkpoint

        for key, tensor in state.items():
            if _is_lycoris_key(key):
                # Determine source based on key patterns
                if "text_model" in key or "text_encoder" in key:
                    # Text encoder key - determine TE1 vs TE2
                    if "clip_g" in key or "text_encoder_2" in key:
                        prefix = "lora_te2"
                    else:
                        prefix = "lora_te1"
                else:
                    # UNet key (no text_model/text_encoder in key)
                    prefix = "lycoris"

                converted_key = _convert_lycoris_key(key, prefix)
                lycoris_state[converted_key] = tensor.detach().cpu()

        # Also check separate text encoder state dicts if present (old format)
        if isinstance(checkpoint, dict):
            for te_key in ["text_encoder_1_state_dict", "text_encoder_2_state_dict"]:
                if te_key in checkpoint:
                    te_name = te_key.replace("_state_dict", "")
                    prefix = "lora_te1" if te_name == "text_encoder_1" else "lora_te2"
                    for key, tensor in checkpoint[te_key].items():
                        if _is_lycoris_key(key):
                            converted_key = _convert_lycoris_key(key, prefix)
                            lycoris_state[converted_key] = tensor.detach().cpu()

    if not lycoris_state:
        raise ValueError(f"No LyCORIS tensors found in {input_path}")

    # Infer metadata from tensors
    metadata = _infer_lycoris_metadata(lycoris_state)

    # Save to safetensors
    save_file(lycoris_state, str(output_path), metadata=metadata)
    print(f"Converted {len(lycoris_state)} LyCORIS tensors to {output_path}")
    return output_path


def _infer_lycoris_metadata(lycoris_state: dict) -> dict[str, str]:
    """Infer LyCORIS metadata from state dict tensors.

    Detects algorithm type from key patterns and infers dimension
    from tensor shapes.
    """
    meta = {
        "format": "pt",
        "generator": "lora_trainer",
        "network_type": "LyCORIS",
    }

    # Detect algorithm based on key patterns
    keys = list(lycoris_state.keys())
    has_hada = any("hada_w1_a" in k or "hada_w1_b" in k for k in keys)
    has_lokr = any("lokr_w1" in k or "lokr_w2" in k for k in keys)
    has_oft = any("oft_blocks" in k for k in keys)

    if has_hada:
        algo = "loha"
    elif has_lokr:
        algo = "lokr"
    elif has_oft:
        algo = "diag-oft"
    else:
        algo = "locon"  # Default/LoRA-style

    meta["network_algo"] = algo

    # Infer dimension from tensor shapes
    dim = None
    for key, tensor in lycoris_state.items():
        if tensor.ndim < 2:
            continue

        # Check different algorithm patterns
        if "lora_down" in key:
            dim = tensor.shape[0]
            break
        elif "hada_w1_a" in key:
            dim = tensor.shape[0]
            break
        elif "lokr_w1_a" in key:
            dim = tensor.shape[0]
            break
        elif "lokr_w1" in key:
            dim = min(tensor.shape)
            break

    if dim is not None:
        meta["network_dim"] = str(dim)
        meta["network_alpha"] = str(dim)  # Default: alpha = dim

    return meta
