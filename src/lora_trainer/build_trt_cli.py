"""Build TensorRT plan files for fixed-shape SDXL inference."""

import argparse
import sys
from pathlib import Path

import torch

from .trt.build import build_unet_engine
from .trt.config import SDXL_RESOLUTIONS, parse_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-shape TensorRT engines for SDXL inference",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Base SDXL checkpoint path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--resolution",
        choices=list(SDXL_RESOLUTIONS),
        required=True,
        help="SDXL-native resolution to build, e.g. 1216x832.",
    )
    parser.add_argument(
        "--engine_dir",
        type=Path,
        default=Path("engines"),
        help="Directory to write TensorRT plan files.",
    )
    parser.add_argument(
        "--onnx_dir",
        type=Path,
        default=Path("engines/onnx"),
        help="Directory to write intermediate ONNX exports.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="TensorRT engine precision.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Export device. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=Path,
        default=None,
        help="Optional LoRA checkpoint to merge before export.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank to instantiate before loading --lora_checkpoint.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--workspace_gb",
        type=float,
        default=12.0,
        help="TensorRT builder workspace limit in GiB.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export ONNX and rebuild even if outputs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda" and args.precision == "fp16":
        raise ValueError("fp16 TensorRT builds require CUDA; pass --device cuda")

    resolution = parse_resolution(args.resolution)
    print(f"Building UNet engine for {resolution.name}")
    print(f"Latent shape: 2x4x{resolution.latent_height}x{resolution.latent_width}")
    artifacts = build_unet_engine(
        args.checkpoint,
        resolution,
        engine_dir=args.engine_dir,
        onnx_dir=args.onnx_dir,
        precision=args.precision,
        device=device,
        lora_checkpoint=args.lora_checkpoint,
        lora_rank=args.lora_rank,
        opset=args.opset,
        workspace_gb=args.workspace_gb,
        force=args.force,
    )
    print(f"Cache:  {artifacts.cache_dir}")
    print(f"ONNX:   {artifacts.onnx_path}")
    print(f"Engine: {artifacts.engine_path}")
    print(f"Key:    {artifacts.key.digest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
