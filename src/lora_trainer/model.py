"""Model loading and LoRA parameter selection.

Phase 2: Real SDXL UNet loading with LoRA injection.
"""

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Cache for loaded pipeline components from single-file checkpoints
_SINGLE_FILE_CACHE = {}


def _load_pipeline_from_single_file(
    checkpoint_path: str, dtype: torch.dtype, device: str = "cpu"
) -> StableDiffusionXLPipeline:
    """Load full SDXL pipeline from single file checkpoint.

    This loads the entire pipeline once and caches it so we can extract
    individual components (UNet, VAE, text encoders) without reloading.
    """
    cache_key = f"{checkpoint_path}_{device}"
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

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize: A with kaiming_uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        # Freeze original layer
        self.original_layer.requires_grad_(False)

        # Scaling factor
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        original_output = self.original_layer(x)
        lora_output = self.lora_up(self.lora_down(x))
        return original_output + lora_output * self.scale


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
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",  # attention output projection
            "ff.net.0.proj",  # feedforward first layer
            "ff.net.2",  # feedforward second layer
        ]

    lora_count = 0

    # Recursively replace Linear layers matching target patterns
    def inject_recursive(module: nn.Module, prefix: str = ""):
        nonlocal lora_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this layer should get LoRA
            if isinstance(child, nn.Linear) and any(
                pattern in full_name for pattern in target_modules
            ):
                # Replace with LoRA layer
                lora_layer = LoRALayer(child, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
                lora_count += 1
            else:
                # Recurse into children
                inject_recursive(child, full_name)

    inject_recursive(unet)
    print(f"Injected LoRA into {lora_count} layers")

    return unet


def load_sdxl_unet(
    checkpoint_or_model_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
) -> UNet2DConditionModel:
    """Load SDXL UNet and inject LoRA layers.

    Args:
        checkpoint_or_model_id: HuggingFace model ID, diffusers directory, or .safetensors file
        device: Device to load model on
        dtype: Model dtype
        lora_rank: LoRA rank for adaptation
        lora_alpha: LoRA alpha scaling

    Returns:
        UNet with LoRA injected
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

    # Inject LoRA
    unet = inject_lora_into_unet(unet, rank=lora_rank, alpha=lora_alpha)

    unet = unet.to(device)
    return unet


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
) -> tuple[CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPTokenizer]:
    """Load SDXL text encoders and tokenizers.

    SDXL uses two text encoders: CLIP ViT-L and OpenCLIP ViT-bigG.

    Args:
        checkpoint_or_model_id: HuggingFace model ID, diffusers directory, or .safetensors file
        device: Device to load encoders on
        dtype: Encoder dtype

    Returns:
        Tuple of (text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2)
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

    text_encoder_1 = text_encoder_1.to(device)
    text_encoder_2 = text_encoder_2.to(device)

    # Freeze text encoders
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    return text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2


def select_lora_params(model: nn.Module) -> Iterator[nn.Parameter]:
    """Select only LoRA parameters for training.

    Args:
        model: Model with LoRA layers injected

    Returns:
        Iterator of LoRA parameters (only lora_down and lora_up weights)
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            yield from module.lora_down.parameters()
            yield from module.lora_up.parameters()


# Backward compatibility: keep load_model() as alias for now
def load_model(checkpoint: str, device: str = "cpu") -> nn.Module:
    """Load model from checkpoint.

    Phase 2: Loads real SDXL UNet with LoRA.

    Args:
        checkpoint: Path to checkpoint or HuggingFace model ID
        device: Device to load model on

    Returns:
        Model instance with LoRA injected
    """
    # Default to fp32 for compatibility
    return load_sdxl_unet(checkpoint, device=device, dtype=torch.float32)
