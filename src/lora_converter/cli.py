"""CLI for converting LoRA checkpoints to ComfyUI safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

from .converter import convert_checkpoint, convert_lycoris_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .pt LoRA checkpoint to ComfyUI-friendly safetensors"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source .pt checkpoint (full training checkpoint or LoRA-only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output path for the .safetensors file "
            "(default: swap extension to .safetensors)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--lycoris",
        action="store_true",
        help="Convert LyCORIS checkpoint instead of LoRA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input.with_suffix(".safetensors")

    if args.lycoris:
        converted_path = convert_lycoris_checkpoint(
            args.input, output_path, overwrite=args.overwrite
        )
        print(f"Converted LyCORIS checkpoint -> {converted_path}")
    else:
        converted_path = convert_checkpoint(args.input, output_path, overwrite=args.overwrite)
        print(f"Converted LoRA checkpoint -> {converted_path}")


if __name__ == "__main__":
    main()
