"""Shared SDXL frozen-inference pipeline — loading, flash attention, backend setup."""

from pathlib import Path

import torch

from .model import (
    detect_lora_rank,
    load_lora_weights,
    load_sdxl_unet,
    load_text_encoders,
    load_vae,
    merge_lora_layers,
)
from .trt.backends import TensorRTUnetBackend, TorchUnetBackend
from .trt.build import build_unet_engine
from .trt.cache import build_engine_cache_key, resolve_engine_artifacts
from .trt.config import ResolutionSpec

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "lms",
    "pndm",
    "ddim",
]
SCHEDULERS = ["karras", "simple", "normal", "exponential", "sgm_uniform"]
PRECISIONS = ["fp16", "bf16", "fp32"]


def dtype_from_precision(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)


def enable_flash_attention(unet) -> None:
    """Enable memory-efficient attention on a diffusers UNet.

    Tries xformers first; falls back to PyTorch 2.0 SDPA which is the default
    in diffusers >= 0.20 with torch >= 2.0, so the fallback is a no-op.
    """
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("Attention: xformers")
    except (ImportError, ModuleNotFoundError):
        pass  # AttnProcessor2_0 (SDPA) is already the diffusers default


def load_inference_models(
    checkpoint: str,
    *,
    device: str,
    dtype: torch.dtype,
    lora_checkpoint: str | Path | None = None,
    lora_rank: int = 16,
) -> tuple:
    """Load VAE + text encoders and optionally merge LoRA weights into the TEs.

    Returns (vae, te1, te2, tok1, tok2). VAE is cast to float32 for decode stability.
    """
    rank = None
    if lora_checkpoint:
        detected = detect_lora_rank(Path(lora_checkpoint))
        rank = detected if detected is not None else lora_rank
    vae = load_vae(checkpoint, device=device, dtype=dtype)
    vae.to(torch.float32)
    te1, te2, tok1, tok2, _ = load_text_encoders(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=rank,
        adapter="lora",
    )
    if lora_checkpoint:
        load_lora_weights(
            Path(lora_checkpoint),
            text_encoder_1=te1,
            text_encoder_2=te2,
        )
        merge_lora_layers(te1)
        merge_lora_layers(te2)
    for m in (vae, te1, te2):
        m.requires_grad_(False).eval()
    return vae, te1, te2, tok1, tok2


def load_torch_unet_backend(
    checkpoint: str,
    *,
    device: str,
    dtype: torch.dtype,
    lora_checkpoint: str | Path | None = None,
    lora_rank: int = 16,
    compile_unet: bool = False,
    flash_attention: bool = True,
) -> TorchUnetBackend:
    """Load UNet, merge LoRA if provided, enable flash attention, return TorchUnetBackend."""
    rank = None
    if lora_checkpoint:
        detected = detect_lora_rank(Path(lora_checkpoint))
        rank = detected if detected is not None else lora_rank
    unet, _ = load_sdxl_unet(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=rank,
        adapter="lora",
    )
    if lora_checkpoint:
        load_lora_weights(Path(lora_checkpoint), unet=unet)
        merge_lora_layers(unet)
    unet.requires_grad_(False).eval()
    if flash_attention:
        enable_flash_attention(unet)
    return TorchUnetBackend(unet, compile_unet=compile_unet)


def prepare_trt_engine(
    checkpoint: str,
    resolution: ResolutionSpec,
    *,
    engine_dir: Path,
    onnx_dir: Path,
    precision: str,
    device: str,
    lora_checkpoint: Path | None = None,
    lora_rank: int = 16,
    workspace_gb: float = 12.0,
    force: bool = False,
    no_build: bool = False,
) -> Path:
    """Build (or locate) a TRT engine and return its path.

    Flash attention for TRT is baked in at engine-build time via trt/build.py.
    Use build_trt_backend to also wrap the result in a TensorRTUnetBackend.
    """
    if no_build:
        key = build_engine_cache_key(
            checkpoint, resolution, precision=precision, lora_checkpoint=lora_checkpoint
        )
        return resolve_engine_artifacts(engine_dir, onnx_dir, key).engine_path
    artifacts = build_unet_engine(
        checkpoint,
        resolution,
        engine_dir=engine_dir,
        onnx_dir=onnx_dir,
        precision=precision,
        device=device,
        lora_checkpoint=lora_checkpoint,
        lora_rank=lora_rank,
        workspace_gb=workspace_gb,
        force=force,
    )
    return artifacts.engine_path


def build_trt_backend(
    checkpoint: str,
    resolution: ResolutionSpec,
    *,
    engine_dir: Path,
    onnx_dir: Path,
    precision: str,
    device: str,
    lora_checkpoint: Path | None = None,
    lora_rank: int = 16,
    workspace_gb: float = 12.0,
    force: bool = False,
    no_build: bool = False,
) -> TensorRTUnetBackend:
    """Build (or locate) a TRT engine and return a TensorRTUnetBackend."""
    engine_path = prepare_trt_engine(
        checkpoint,
        resolution,
        engine_dir=engine_dir,
        onnx_dir=onnx_dir,
        precision=precision,
        device=device,
        lora_checkpoint=lora_checkpoint,
        lora_rank=lora_rank,
        workspace_gb=workspace_gb,
        force=force,
        no_build=no_build,
    )
    return TensorRTUnetBackend(engine_path)
