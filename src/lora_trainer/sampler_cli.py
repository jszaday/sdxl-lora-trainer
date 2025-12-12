"""Standalone CLI to run validation sampling without training."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from .logging import create_run_dirs, init_tensorboard
from .model import load_sdxl_unet, load_text_encoders, load_vae
from .sampling import run_validation_samples
from .train_loop import load_checkpoint
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDXL LoRA Sampler - generate images from structured prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Base SDXL checkpoint path or HuggingFace model ID",
    )
    required.add_argument(
        "--sample_prompts",
        type=Path,
        required=True,
        help="Path to JSON/JSONL file with prompts (positive/negative/seed)",
    )
    required.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Output directory for samples and logs",
    )

    sampling = parser.add_argument_group("sampling arguments")
    sampling.add_argument(
        "--scheduler",
        type=str,
        default="normal",
        choices=["simple", "normal", "karras"],
        help="Noise scheduler for sampling (default: normal)",
    )
    sampling.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "euler_ancestral", "ddim", "heun"],
        help="Sampler algorithm (for UX parity; currently uses scheduler only)",
    )
    sampling.add_argument(
        "--cfg",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale (default: 7.0)",
    )
    sampling.add_argument(
        "--sampler_steps",
        type=int,
        default=30,
        help="Number of diffusion steps (default: 30)",
    )
    sampling.add_argument(
        "--samples_per_prompt",
        type=int,
        default=1,
        help="Number of samples to generate per prompt entry (default: 1)",
    )
    sampling.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size for sampling (default: 1024)",
    )

    lora_group = parser.add_argument_group("LoRA arguments")
    lora_group.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank of LoRA matrices (default: 16)",
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling parameter (default: 16.0)",
    )
    lora_group.add_argument(
        "--lora_checkpoint",
        type=Path,
        default=None,
        help="Optional LoRA checkpoint (.pt) to load before sampling",
    )

    misc = parser.add_argument_group("misc arguments")
    misc.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., cuda, cpu, mps). Defaults to cuda if available.",
    )
    misc.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (used when prompts do not specify seeds)",
    )
    misc.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode (default: fp16)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set seed for reproducibility when prompts don't provide seeds
    set_seed(args.seed)

    # Determine device
    if args.device is not None:
        device = args.device
        print(f"Using device: {device}")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Warning: CUDA not available, using CPU")

    # Determine dtype
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Create workspace dirs
    dirs = create_run_dirs(args.workspace)
    print(f"\nWorkspace: {dirs['root']}")
    print(f"  - Samples:     {dirs['samples']}")
    print(f"  - TensorBoard: {dirs['tb']}")

    writer = init_tensorboard(dirs["tb"])

    # Load models
    print(f"\nLoading UNet from: {args.checkpoint}")
    unet = load_sdxl_unet(
        checkpoint_or_model_id=args.checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    if args.lora_checkpoint is not None:
        print(f"Loading LoRA weights from: {args.lora_checkpoint}")
        resume_step = load_checkpoint(
            checkpoint_path=args.lora_checkpoint,
            model=unet,
            optimizer=None,
            device=device,
        )
        print(f"Loaded checkpoint step: {resume_step}")

    print("Loading VAE...")
    vae = load_vae(args.checkpoint, device=device, dtype=dtype)

    print("Loading text encoders...")
    text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2 = load_text_encoders(
        args.checkpoint, device=device, dtype=dtype
    )

    config_like = SimpleNamespace(
        scheduler=args.scheduler,
        sampler=args.sampler,
        cfg=args.cfg,
        sampler_steps=args.sampler_steps,
        sample_prompts=args.sample_prompts,
        samples_per_prompt=args.samples_per_prompt,
        image_size=args.image_size,
    )

    print("\nGenerating samples...")
    run_validation_samples(
        unet=unet,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        config=config_like,
        global_step=0,
        samples_dir=dirs["samples"],
        writer=writer,
        device=device,
    )

    writer.close()
    print("\nSampling complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
