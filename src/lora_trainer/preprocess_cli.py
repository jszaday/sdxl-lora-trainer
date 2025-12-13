"""CLI entry point for dataset preprocessing."""

import argparse
import sys
from pathlib import Path

import torch

from .preprocess import preprocess_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SDXL LoRA Trainer - Dataset Preprocessing (Cache latents and embeddings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to base SDXL checkpoint or HuggingFace model ID",
    )
    required.add_argument(
        "--train_data",
        type=Path,
        required=True,
        help="Directory containing training images (with optional .txt captions)",
    )
    required.add_argument(
        "--cache_dir",
        type=Path,
        required=True,
        help="Output directory for cached latents and embeddings",
    )

    # Optional arguments
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size for preprocessing (default: 1024, SDXL native resolution)",
    )
    optional.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for preprocessing (higher = faster, more VRAM) (default: 1)",
    )
    optional.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0', 'mps'). "
        "If not specified, automatically selects cuda if available, otherwise cpu.",
    )
    optional.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for preprocessing (default: fp16)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for preprocessing."""
    args = parse_args()

    # Determine device
    if args.device is not None:
        device = args.device
        print(f"Using device: {device}")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Warning: CUDA not available, using CPU (preprocessing will be slow)")

    # Determine dtype
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("\nPreprocessing Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Train data: {args.train_data}")
    print(f"  Cache dir: {args.cache_dir}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")

    # Run preprocessing
    try:
        preprocess_dataset(
            train_data=args.train_data,
            cache_dir=args.cache_dir,
            checkpoint=args.checkpoint,
            image_size=args.image_size,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"\nError during preprocessing: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n✓ Preprocessing complete!")
    print("\nYou can now train using cached data with:")
    print(f"  lora-train --cached_data {args.cache_dir} ...")


if __name__ == "__main__":
    main()
