"""Benchmark spandrel upscale models: PyTorch fp32 vs torch.compile fp16 vs TRT fp16."""

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a spandrel upscale model across backends: "
            "PyTorch fp32 (ComfyUI baseline), torch.compile fp16, and optionally TRT fp16. "
            "Useful for evaluating hires-fix pipeline speed."
        ),
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the upscale model (.safetensors or .pth). "
        "Supported architectures: ESRGAN, Real-ESRGAN, SPAN, DAT, CUGAN (via spandrel).",
    )
    parser.add_argument(
        "--engine_dir",
        type=Path,
        default=Path("engines/upscaler"),
        help="Directory to write TRT plan files (default: engines/upscaler).",
    )
    parser.add_argument(
        "--onnx_dir",
        type=Path,
        default=Path("engines/upscaler/onnx"),
        help="Directory to write intermediate ONNX exports.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size for both ONNX export and inference. "
        "ComfyUI defaults to 512; use 256 if VRAM is tight (default: 512).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Tile overlap in pixels, same as ComfyUI default (default: 32).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Synthetic benchmark image size in pixels (default: 1024).",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="TRT engine precision (default: fp16).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device string, e.g. 'cuda' or 'cuda:1'. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset for export (default: 17).",
    )
    parser.add_argument(
        "--workspace_gb",
        type=float,
        default=4.0,
        help="TRT builder workspace memory limit in GiB (default: 4.0).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark iterations to average (default: 5).",
    )
    parser.add_argument(
        "--skip_trt",
        action="store_true",
        help="Skip TRT build and benchmark (faster; torch.compile is recommended on Blackwell).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export ONNX and rebuild TRT engine even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and args.precision == "fp16":
        print("Warning: fp16 TRT requires CUDA; switching to fp32", file=sys.stderr)
        args.precision = "fp32"

    from .trt.upscaler import benchmark_upscaler

    print(f"Model:     {args.model}")
    print(f"Tile size: {args.tile_size}px  overlap: {args.overlap}px")
    print(f"Benchmark: {args.image_size}×{args.image_size} synthetic image, {args.runs} runs")
    print(f"Precision: {args.precision}  device: {device}")
    print()

    results = benchmark_upscaler(
        args.model,
        engine_dir=args.engine_dir,
        onnx_dir=args.onnx_dir,
        image_size=args.image_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        precision=args.precision,
        device=device,
        opset=args.opset,
        workspace_gb=args.workspace_gb,
        force=args.force,
        runs=args.runs,
        skip_trt=args.skip_trt,
    )

    baseline = results["torch_fp32_s"]
    print()
    print("=" * 55)
    print(f"  PyTorch fp32 (ComfyUI baseline):  {baseline:.3f}s")
    compiled = results["compiled_fp32_s"]
    print(f"  torch.compile fp32:               {compiled:.3f}s  ({baseline/compiled:.1f}×)")
    if "trt_fp16_s" in results:
        import math
        trt = results["trt_fp16_s"]
        if not math.isnan(trt):
            print(f"  TRT fp16:                         {trt:.3f}s  ({baseline/trt:.1f}×)")
        else:
            print("  TRT fp16:                         FAILED (see above)")
    print("=" * 55)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
