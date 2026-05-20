"""Model loading and LoRA parameter selection."""

import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel
from safetensors.torch import load_file as load_safetensors
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

if TYPE_CHECKING:
    from lycoris import LycorisNetwork

# Cache for loaded pipeline components from single-file checkpoints
_SINGLE_FILE_CACHE = {}


def clear_single_file_cache() -> None:
    """Clear cached single-file diffusers pipelines."""
    _SINGLE_FILE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_lycoris() -> tuple["LycorisNetwork", Any]:
    """Import LyCORIS lazily to avoid hard dependency when unused."""
    try:
        from lycoris import LycorisNetwork, create_lycoris
    except ImportError as exc:  # pragma: no cover - exercised only when missing extra
        raise ImportError(
            "LyCORIS is not installed. Install it with `pip install lycoris-lora`."
        ) from exc
    return LycorisNetwork, create_lycoris


def apply_lycoris_adapter(
    model: nn.Module,
    *,
    linear_dim: int,
    linear_alpha: float,
    algo: str = "lokr",
    factor: int = -1,
    dropout: float | None = None,
    device: str = "cpu",
    preset: dict[str, list[str]] | None = None,
    multiplier: float = 1.0,
) -> "LycorisNetwork":
    """Attach a LyCORIS network to the given model."""
    lycoris_network_cls, create_lycoris = _require_lycoris()

    # Default preset: match attention modules
    if preset is None:
        preset = {"target_name": [".*attn.*"]}
    lycoris_network_cls.apply_preset(preset)

    # Freeze base weights and create LyCORIS layers
    model.requires_grad_(False)
    kwargs = {
        "linear_dim": linear_dim,
        "linear_alpha": linear_alpha,
        "algo": algo,
        "factor": factor,
    }
    if dropout is not None:
        params = inspect.signature(create_lycoris).parameters
        if "dropout" in params:
            kwargs["dropout"] = dropout
        elif "rank_dropout" in params:
            kwargs["rank_dropout"] = dropout
        elif "module_dropout" in params:
            kwargs["module_dropout"] = dropout
        else:
            print("Warning: LyCORIS dropout requested but not supported by this version.")

    lyco_net = create_lycoris(model, multiplier, **kwargs)
    lyco_net.apply_to()

    # Move LyCORIS parameters to the same device as the model
    lyco_net = lyco_net.to(device)

    return lyco_net


def _load_pipeline_from_single_file(
    checkpoint_path: str, dtype: torch.dtype, device: str = "cpu"
) -> StableDiffusionXLPipeline:
    """Load full SDXL pipeline from single file checkpoint.

    This loads the entire pipeline once and caches it so we can extract
    individual components (UNet, VAE, text encoders) without reloading.
    """
    # Include dtype in cache key so fp16/fp32 pipelines don't collide.
    cache_key = f"{checkpoint_path}_{device}_{str(dtype)}"
    if cache_key not in _SINGLE_FILE_CACHE:
        print(f"Loading full pipeline from {checkpoint_path}")
        pipeline = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=dtype,
            device=device,
        )
        _SINGLE_FILE_CACHE[cache_key] = pipeline
    return _SINGLE_FILE_CACHE[cache_key]


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.

    Wraps a linear layer and adds low-rank update matrices A and B.
    Forward pass computes: output = original(x) + (x @ A @ B) * scale
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices: down-project (A) and up-project (B)
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Keep LoRA params on the same dtype/device as the wrapped layer to avoid
        # mixed precision matmul issues (e.g., fp16 inputs with fp32 LoRA weights).
        param_kwargs = {
            "device": original_layer.weight.device,
            "dtype": original_layer.weight.dtype,
            "bias": False,
        }
        self.lora_down = nn.Linear(in_features, rank, **param_kwargs)
        self.lora_up = nn.Linear(rank, out_features, **param_kwargs)

        # Initialize: A with kaiming_uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        # Freeze original layer
        self.original_layer.requires_grad_(False)

        # Scaling factor
        self.rank = rank
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        original_output = self.original_layer(x)
        lora_output = self.lora_up(self.lora_down(x))
        scale = (self.alpha / self.rank) if self.rank > 0 else 1.0
        return original_output + lora_output * scale


class LoRAConv2d(nn.Module):
    """LoRA wrapper for Conv2d layers."""

    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int = 16,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = float(alpha)

        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        kernel_size = original_layer.kernel_size
        stride = original_layer.stride
        padding = original_layer.padding
        dilation = original_layer.dilation
        groups = original_layer.groups

        param_kwargs = {
            "device": original_layer.weight.device,
            "dtype": original_layer.weight.dtype,
            "bias": False,
        }
        self.lora_down = nn.Conv2d(
            in_channels,
            rank,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **param_kwargs,
        )
        self.lora_up = nn.Conv2d(
            rank,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            **param_kwargs,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.original_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        lora_output = self.lora_up(self.lora_down(x))
        scale = (self.alpha / self.rank) if self.rank > 0 else 1.0
        return original_output + lora_output * scale


def merge_lora_layers(model: nn.Module) -> nn.Module:
    """Merge in-place LoRA wrappers into their wrapped base layers.

    This is intended for frozen inference/export. It removes LoRA module
    overhead and leaves the model with regular Linear/Conv2d layers.
    """

    def merge_recursive(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALayer):
                merged = child.original_layer
                scale = (child.alpha / child.rank) if child.rank > 0 else 1.0
                delta = child.lora_up.weight @ child.lora_down.weight
                with torch.no_grad():
                    merged.weight.add_(delta.to(merged.weight.dtype) * scale)
                setattr(module, name, merged)
                continue

            if isinstance(child, LoRAConv2d):
                merged = child.original_layer
                if child.lora_down.groups != 1:
                    raise ValueError("Cannot merge grouped LoRAConv2d layers for frozen inference")
                scale = (child.alpha / child.rank) if child.rank > 0 else 1.0
                up = child.lora_up.weight[:, :, 0, 0]
                down = child.lora_down.weight
                delta = torch.einsum("or,rihw->oihw", up, down)
                with torch.no_grad():
                    merged.weight.add_(delta.to(merged.weight.dtype) * scale)
                setattr(module, name, merged)
                continue

            merge_recursive(child)

    merge_recursive(model)
    return model


def inject_lora_into_unet(
    unet: UNet2DConditionModel,
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
) -> UNet2DConditionModel:
    """Inject LoRA layers into UNet attention and feedforward layers.

    Args:
        unet: SDXL UNet model
        rank: LoRA rank
        alpha: LoRA alpha scaling
        target_modules: List of module name patterns to target.
                       Defaults to attention projections and feedforward layers.

    Returns:
        UNet with LoRA layers injected
    """
    if target_modules is None:
        # Default: target attention Q, K, V, out projections and feedforward layers
        # Match ComfyUI's LoRA coverage
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",  # attention output projection
            "ff.net.0.proj",  # feedforward first layer
            "ff.net.2",  # feedforward second layer
            "conv_in",
            "conv_out",
            "conv1",
            "conv2",
            "conv_shortcut",  # resnet shortcut convolution
            "time_emb_proj",  # per-resnet time embedding projection
            "time_embedding.linear_1",  # global time embedding MLP
            "time_embedding.linear_2",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "add_embedding.linear_1",
            "add_embedding.linear_2",
            "proj_in",  # attention input projection
            "proj_out",  # attention output projection
        ]

    lora_count = 0

    # Recursively replace Linear layers matching target patterns
    def inject_recursive(module: nn.Module, prefix: str = ""):
        nonlocal lora_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this layer should get LoRA
            if any(pattern in full_name for pattern in target_modules):
                if isinstance(child, nn.Linear):
                    lora_layer = LoRALayer(child, rank=rank, alpha=alpha)
                    setattr(module, name, lora_layer)
                    lora_count += 1
                    continue
                if isinstance(child, nn.Conv2d):
                    lora_layer = LoRAConv2d(child, rank=rank, alpha=alpha)
                    setattr(module, name, lora_layer)
                    lora_count += 1
                    continue
            # Recurse into children
            inject_recursive(child, full_name)

    inject_recursive(unet)
    print(f"Injected LoRA into {lora_count} layers")

    return unet


def inject_lora_into_text_encoder(
    text_encoder: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Inject LoRA into CLIP text encoder attention/MLP projections."""
    if target_modules is None:
        target_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "mlp.fc1",
            "mlp.fc2",
        ]

    lora_count = 0

    def inject_recursive(module: nn.Module, prefix: str = ""):
        nonlocal lora_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(
                pattern in full_name for pattern in target_modules
            ):
                setattr(module, name, LoRALayer(child, rank=rank, alpha=alpha))
                lora_count += 1
            else:
                inject_recursive(child, full_name)

    inject_recursive(text_encoder)
    print(f"Injected LoRA into text encoder ({lora_count} layers)")
    return text_encoder


def load_sdxl_unet(
    checkpoint_or_model_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    lora_rank: int | None = 16,
    lora_alpha: float = 16.0,
    adapter: str = "lora",
    lycoris_dim: int | None = None,
    lycoris_alpha: float | None = None,
    lycoris_algo: str = "lokr",
    lycoris_factor: int = -1,
    lycoris_dropout: float | None = None,
) -> tuple[UNet2DConditionModel, Any | None]:
    """Load SDXL UNet and inject LoRA layers.

    Args:
        checkpoint_or_model_id: HuggingFace model ID, diffusers directory, or .safetensors file
        device: Device to load model on
        dtype: Model dtype
        lora_rank: LoRA rank for adaptation
        lora_alpha: LoRA alpha scaling
        adapter: Adapter backend ('lora' or 'lycoris')
        lycoris_dim: LyCORIS linear dim (defaults to lora_rank)
        lycoris_alpha: LyCORIS alpha (defaults to lora_alpha)
        lycoris_algo: LyCORIS algorithm (e.g., 'lokr')
        lycoris_factor: LyCORIS factorization factor (-1 = auto)
        lycoris_dropout: LyCORIS dropout (optional)

    Returns:
        Tuple of (UNet with adapters, adapter handle or None)
    """
    print(f"Loading SDXL UNet from {checkpoint_or_model_id}")

    checkpoint_path = Path(checkpoint_or_model_id)

    # Check if it's a single file checkpoint (.safetensors or .ckpt)
    if checkpoint_path.exists() and checkpoint_path.is_file():
        # Load from single file - extract UNet from full pipeline
        pipeline = _load_pipeline_from_single_file(checkpoint_or_model_id, dtype, device)
        unet = pipeline.unet
    else:
        # Load from HuggingFace repo or diffusers directory
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_or_model_id,
            subfolder="unet" if "/" in checkpoint_or_model_id else None,
            torch_dtype=dtype,
        )

    # Move model to device BEFORE applying adapters so adapter params
    # are created on the right device
    unet = unet.to(device)

    adapter = adapter.lower()
    lyco_net = None
    if adapter == "lora":
        if lora_rank is not None:
            unet = inject_lora_into_unet(unet, rank=lora_rank, alpha=lora_alpha)
    elif adapter == "lycoris":
        lyco_net = apply_lycoris_adapter(
            unet,
            linear_dim=lycoris_dim or lora_rank,
            linear_alpha=lycoris_alpha or lora_alpha,
            algo=lycoris_algo,
            factor=lycoris_factor,
            dropout=lycoris_dropout,
            device=device,
        )
    else:
        raise ValueError(f"Unknown adapter type: {adapter}")

    return unet, lyco_net


def load_vae(
    checkpoint_or_model_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AutoencoderKL:
    """Load SDXL VAE for encoding images to latents.

    Args:
        checkpoint_or_model_id: HuggingFace model ID, diffusers directory, or .safetensors file
        device: Device to load VAE on
        dtype: VAE dtype

    Returns:
        VAE model
    """
    checkpoint_path = Path(checkpoint_or_model_id)

    # Check if it's a single file checkpoint
    if checkpoint_path.exists() and checkpoint_path.is_file():
        # Load from single file - extract VAE from full pipeline
        pipeline = _load_pipeline_from_single_file(checkpoint_or_model_id, dtype, device)
        vae = pipeline.vae
    else:
        # Load from HuggingFace repo or diffusers directory
        vae = AutoencoderKL.from_pretrained(
            checkpoint_or_model_id,
            subfolder="vae" if "/" in checkpoint_or_model_id else None,
            torch_dtype=dtype,
        )

    vae = vae.to(device)
    vae.requires_grad_(False)  # VAE is frozen during training
    return vae


def load_text_encoders(
    checkpoint_or_model_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    adapter: str = "lora",
    lycoris_dim: int | None = None,
    lycoris_alpha: float | None = None,
    lycoris_algo: str = "lokr",
    lycoris_factor: int = -1,
    lycoris_dropout: float | None = None,
) -> tuple[
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPTokenizer,
    tuple[Any | None, Any | None],
]:
    """Load SDXL text encoders and tokenizers.

    SDXL uses two text encoders: CLIP ViT-L and OpenCLIP ViT-bigG.

    Args:
        checkpoint_or_model_id: HuggingFace model ID, diffusers directory, or .safetensors file
        device: Device to load encoders on
        dtype: Encoder dtype
        adapter: Adapter backend ('lora' or 'lycoris')
        lycoris_factor: LyCORIS factorization factor (-1 = auto)
        lycoris_dropout: LyCORIS dropout (optional)

    Returns:
        Tuple of (text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, adapter handles)
    """
    checkpoint_path = Path(checkpoint_or_model_id)

    # Check if it's a single file checkpoint
    if checkpoint_path.exists() and checkpoint_path.is_file():
        # Load from single file - extract text encoders from full pipeline
        pipeline = _load_pipeline_from_single_file(checkpoint_or_model_id, dtype, device)
        text_encoder_1 = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        tokenizer_1 = pipeline.tokenizer
        tokenizer_2 = pipeline.tokenizer_2
    else:
        # Load from HuggingFace repo or diffusers directory
        base_model = checkpoint_or_model_id

        # Text encoder 1 (CLIP ViT-L)
        text_encoder_1 = CLIPTextModel.from_pretrained(
            base_model,
            subfolder="text_encoder" if "/" in base_model else None,
            torch_dtype=dtype,
        )
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            base_model,
            subfolder="tokenizer" if "/" in base_model else None,
        )

        # Text encoder 2 (OpenCLIP ViT-bigG)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_model,
            subfolder="text_encoder_2" if "/" in base_model else None,
            torch_dtype=dtype,
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            base_model,
            subfolder="tokenizer_2" if "/" in base_model else None,
        )

    # Freeze base parameters before attaching adapters so new adapter params stay trainable
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Move models to device BEFORE applying adapters so adapter params
    # are created on the right device
    text_encoder_1 = text_encoder_1.to(device)
    text_encoder_2 = text_encoder_2.to(device)

    adapter = adapter.lower()
    te1_adapter: Any | None = None
    te2_adapter: Any | None = None
    # Only inject adapter layers if rank/dim is specified
    if lora_rank is not None:
        if adapter == "lora":
            text_encoder_1 = inject_lora_into_text_encoder(
                text_encoder_1, rank=lora_rank, alpha=lora_alpha
            )
            text_encoder_2 = inject_lora_into_text_encoder(
                text_encoder_2, rank=lora_rank, alpha=lora_alpha
            )
        elif adapter == "lycoris":
            te1_adapter = apply_lycoris_adapter(
                text_encoder_1,
                linear_dim=lycoris_dim or lora_rank,
                linear_alpha=lycoris_alpha or lora_alpha,
                algo=lycoris_algo,
                factor=lycoris_factor,
                dropout=lycoris_dropout,
                device=device,
            )
            te2_adapter = apply_lycoris_adapter(
                text_encoder_2,
                linear_dim=lycoris_dim or lora_rank,
                linear_alpha=lycoris_alpha or lora_alpha,
                algo=lycoris_algo,
                factor=lycoris_factor,
                dropout=lycoris_dropout,
                device=device,
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter}")

    return text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, (te1_adapter, te2_adapter)


def select_lora_params(model: nn.Module) -> Iterator[nn.Parameter]:
    """Select only LoRA parameters for training.

    Args:
        model: Model with LoRA layers injected

    Returns:
        Iterator of LoRA parameters (only lora_down and lora_up weights)
    """
    for module in model.modules():
        if isinstance(module, LoRALayer | LoRAConv2d):
            yield from module.lora_down.parameters()
            yield from module.lora_up.parameters()


def infer_lora_hparams(model: nn.Module) -> tuple[int | None, float | None]:
    """Infer LoRA rank/alpha from the first LoRALayer in a model."""
    for module in model.modules():
        if isinstance(module, LoRALayer):
            return module.rank, module.alpha
    return None, None


def _load_lora_state(lora_path: Path) -> dict[str, torch.Tensor]:
    """Load LoRA tensors from a .pt or .safetensors file."""
    if lora_path.suffix.lower() == ".safetensors":
        return dict(load_safetensors(str(lora_path)))

    state = torch.load(lora_path, map_location="cpu")
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported LoRA checkpoint format at {lora_path}")
    return {k: v for k, v in state.items() if "lora_" in k}


def _build_lora_key_map(model: nn.Module, prefix: str) -> dict[str, str]:
    """Build mapping from LoRA keys to model state dict keys.

    Args:
        model: The model (UNet or text encoder) with LoRA layers already injected
        prefix: Prefix for LoRA keys (e.g., 'lora_unet', 'lora_te1', 'lora_te2')

    Returns:
        Dictionary mapping LoRA keys to model state dict keys
    """
    key_map = {}
    state_dict = model.state_dict()

    # After LoRA injection, keys look like: module.path.lora_down.weight
    # LoRA files have keys like: lora_unet_module_path.lora_down.weight
    # We need to map from the latter to the former

    for model_key in state_dict.keys():
        # Only process LoRA parameter keys
        if ".lora_down.weight" not in model_key and ".lora_up.weight" not in model_key:
            continue

        # Extract the base module path (everything before .lora_down or .lora_up)
        # Model keys look like: "add_embedding.linear_1.lora_down.weight"
        # We want the base to be: "add_embedding.linear_1"

        if ".lora_down.weight" in model_key:
            # Find the position of .lora_down.weight
            lora_pos = model_key.find(".lora_down.weight")
            base_key = model_key[:lora_pos]
            suffix = "lora_down.weight"
        elif ".lora_up.weight" in model_key:
            # Find the position of .lora_up.weight
            lora_pos = model_key.find(".lora_up.weight")
            base_key = model_key[:lora_pos]
            suffix = "lora_up.weight"
        else:
            continue

        # Create ComfyUI-style LoRA key: replace dots with underscores in module path
        lora_key_base = base_key.replace(".", "_")

        # Map ComfyUI format: lora_unet_module_name.suffix
        lora_key = f"{prefix}_{lora_key_base}.{suffix}"
        key_map[lora_key] = model_key

        # Also map diffusers format: module.name.suffix (without prefix)
        lora_key = f"{base_key}.{suffix}"
        key_map[lora_key] = model_key

    # Now handle alpha keys separately (they're not in state_dict, need to get from modules)
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer | LoRAConv2d):
            # name is like "add_embedding.linear_1"
            # Create corresponding alpha mapping
            lora_key_base = name.replace(".", "_")

            # ComfyUI format alpha key
            lora_alpha_key = f"{prefix}_{lora_key_base}.alpha"
            # We don't have alpha in state dict, but we want to map it for loading
            # Store None as placeholder - we'll handle it specially in load_lora_weights
            key_map[lora_alpha_key] = f"{name}.alpha"

            # Diffusers format alpha key
            lora_alpha_key = f"{name}.alpha"
            key_map[lora_alpha_key] = f"{name}.alpha"

    return key_map


def load_lora_weights(
    lora_path: Path,
    unet: nn.Module,
    text_encoder_1: nn.Module | None = None,
    text_encoder_2: nn.Module | None = None,
) -> None:
    """Load LoRA weights into UNet and text encoders.

    Supports multiple LoRA formats:
    - ComfyUI format: lora_unet_*, lora_te1_*, lora_te2_*
    - Diffusers format: module.path.lora_up.weight
    """
    state = _load_lora_state(Path(lora_path))
    if not state:
        print(f"Warning: no LoRA tensors found in {lora_path}")
        return

    # Build key mappings for each model
    unet_key_map = _build_lora_key_map(unet, "lora_unet")
    te1_key_map = _build_lora_key_map(text_encoder_1, "lora_te1") if text_encoder_1 else {}
    te2_key_map = _build_lora_key_map(text_encoder_2, "lora_te2") if text_encoder_2 else {}

    # Combine all mappings to find what each LoRA key maps to
    all_mappings = {**unet_key_map, **te1_key_map, **te2_key_map}

    # Group LoRA tensors by target model (weights only, not alphas)
    groups = {"unet": {}, "te1": {}, "te2": {}}
    # Group alpha values separately for manual application
    alphas = {"unet": {}, "te1": {}, "te2": {}}
    loaded_keys = set()

    for lora_key, lora_tensor in state.items():
        if lora_key in all_mappings:
            model_key = all_mappings[lora_key]

            # Separate alpha values from weights
            is_alpha = lora_key.endswith(".alpha")

            # Determine which model this belongs to
            if lora_key in unet_key_map:
                if is_alpha:
                    # Extract module name (remove .alpha suffix from model_key)
                    module_name = model_key[: -len(".alpha")]
                    alphas["unet"][module_name] = lora_tensor.item()
                else:
                    groups["unet"][model_key] = lora_tensor
            elif lora_key in te1_key_map:
                if is_alpha:
                    module_name = model_key[: -len(".alpha")]
                    alphas["te1"][module_name] = lora_tensor.item()
                else:
                    groups["te1"][model_key] = lora_tensor
            elif lora_key in te2_key_map:
                if is_alpha:
                    module_name = model_key[: -len(".alpha")]
                    alphas["te2"][module_name] = lora_tensor.item()
                else:
                    groups["te2"][model_key] = lora_tensor

            loaded_keys.add(lora_key)

    # Load weights into each model
    def _load_component(
        model: nn.Module | None,
        substate: dict[str, torch.Tensor],
        alpha_dict: dict[str, float],
    ) -> int:
        if model is None or not substate:
            return 0

        # Load weights
        result = model.load_state_dict(substate, strict=False)
        loaded_count = len(substate) - len(result.unexpected_keys)

        # Apply alpha values to LoRA modules
        for module_name, alpha_value in alpha_dict.items():
            # Navigate to the module
            parts = module_name.split(".")
            module = model
            try:
                for part in parts:
                    module = getattr(module, part)
                # Update alpha if this is a LoRA module
                if isinstance(module, LoRALayer | LoRAConv2d):
                    module.alpha = float(alpha_value)
            except AttributeError:
                # Module not found, skip
                pass

        return loaded_count

    applied_unet = _load_component(unet, groups["unet"], alphas["unet"])
    applied_te1 = _load_component(text_encoder_1, groups["te1"], alphas["te1"])
    applied_te2 = _load_component(text_encoder_2, groups["te2"], alphas["te2"])

    applied_total = applied_unet + applied_te1 + applied_te2
    dropped_count = len(state) - len(loaded_keys)

    print(
        f"Applied {applied_total} LoRA tensors "
        f"(UNet {applied_unet}, TE1 {applied_te1}, TE2 {applied_te2})"
    )

    if dropped_count > 0:
        print(f"Warning: dropped {dropped_count} LoRA tensors (no matching modules)")


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA weights and alpha values from a model.

    Returns a state dict with:
    - module.lora_down.weight
    - module.lora_up.weight
    - module.alpha (as a scalar tensor)

    Args:
        model: Model with LoRA layers injected

    Returns:
        Dictionary with LoRA weights and alpha values
    """
    lora_state = {}
    state_dict = model.state_dict()

    # First, get all the lora_down and lora_up weights
    for key, tensor in state_dict.items():
        if ".lora_down.weight" in key or ".lora_up.weight" in key:
            lora_state[key] = tensor.detach().cpu()

    # Now extract alpha values from LoRALayer and LoRAConv2d modules
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer | LoRAConv2d):
            # Alpha is stored as an instance variable
            alpha_key = f"{name}.alpha"
            # Store as a scalar tensor to match ComfyUI format
            lora_state[alpha_key] = torch.tensor(module.alpha, dtype=torch.float32)

    return lora_state


def build_lora_metadata(rank: int | None, alpha: float | None) -> dict[str, str]:
    """Build safetensors metadata for LoRA exports."""
    meta = {
        "format": "pt",
        "generator": "lora_trainer",
    }
    if rank is not None:
        meta["network_dim"] = str(rank)
    if alpha is not None:
        meta["network_alpha"] = str(alpha)
    return meta


# Backward compatibility: keep load_model() as alias for now
def load_model(checkpoint: str, device: str = "cpu") -> nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint: Path to checkpoint or HuggingFace model ID
        device: Device to load model on

    Returns:
        Model instance with LoRA injected
    """
    # Default to fp32 for compatibility
    model, _ = load_sdxl_unet(checkpoint, device=device, dtype=torch.float32)
    return model
