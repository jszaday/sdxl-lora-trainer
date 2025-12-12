"""CLI entry point for SDXL LoRA training."""

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim

from .config import TrainingConfig
from .data import build_dataloader
from .logging import create_run_dirs, init_tensorboard, log_hparams
from .model import load_model, select_lora_params
from .train_loop import train
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SDXL LoRA Trainer - Single-purpose, high-UX training toolkit",
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
        "--steps",
        type=int,
        required=True,
        help="Total number of training steps to run",
    )
    required.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Training batch size per GPU",
    )
    required.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Output directory for checkpoints, logs, and samples",
    )

    # Optimizer arguments
    optim_group = parser.add_argument_group("optimizer arguments")
    optim_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer (default: 1e-4)",
    )
    optim_group.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )

    # Data arguments
    data_group = parser.add_argument_group("data arguments")
    data_group.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size for training (default: 1024, SDXL native resolution)",
    )
    data_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    # Sampling/validation arguments
    sampling_group = parser.add_argument_group("sampling/validation arguments")
    sampling_group.add_argument(
        "--scheduler",
        type=str,
        default="normal",
        choices=["simple", "normal", "karras"],
        help="Noise scheduler for validation sampling (default: normal)",
    )
    sampling_group.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "euler_ancestral", "ddim", "heun"],
        help="Sampler algorithm for validation sampling (default: euler)",
    )
    sampling_group.add_argument(
        "--cfg",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale for sampling (default: 7.0)",
    )
    sampling_group.add_argument(
        "--sampler_steps",
        type=int,
        default=30,
        help="Number of diffusion steps for validation sampling (default: 30)",
    )
    sampling_group.add_argument(
        "--sample_prompts",
        type=Path,
        default=None,
        help="Path to text file with validation prompts (one per line)",
    )
    sampling_group.add_argument(
        "--sample_every",
        type=int,
        default=500,
        help="Generate validation samples every N steps (default: 500)",
    )
    sampling_group.add_argument(
        "--samples_per_prompt",
        type=int,
        default=1,
        help="Number of samples to generate per prompt (default: 1)",
    )

    # LoRA arguments
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

    # Misc arguments
    misc_group = parser.add_argument_group("misc arguments")
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    misc_group.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode (default: fp16)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for training."""
    args = parse_args()

    # Build config from arguments
    try:
        config = TrainingConfig(
            checkpoint=args.checkpoint,
            train_data=args.train_data,
            steps=args.steps,
            batch_size=args.batch_size,
            workspace=args.workspace,
            learning_rate=args.learning_rate,
            grad_accum=args.grad_accum,
            image_size=args.image_size,
            num_workers=args.num_workers,
            scheduler=args.scheduler,
            sampler=args.sampler,
            cfg=args.cfg,
            sampler_steps=args.sampler_steps,
            sample_prompts=args.sample_prompts,
            sample_every=args.sample_every,
            samples_per_prompt=args.samples_per_prompt,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
            mixed_precision=args.mixed_precision,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print configuration summary
    print(config.print_summary())

    # Set random seed
    set_seed(config.seed)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\nWarning: CUDA not available, using CPU (training will be slow)")

    # Create output directories
    dirs = create_run_dirs(config.workspace)
    print(f"\nWorkspace: {dirs['root']}")
    print(f"  - Checkpoints: {dirs['checkpoints']}")
    print(f"  - TensorBoard: {dirs['tb']}")
    print(f"  - Samples:     {dirs['samples']}")

    # Initialize TensorBoard
    writer = init_tensorboard(dirs["tb"])
    log_hparams(writer, config)

    # Build dataloader
    print(f"\nLoading training data from: {config.train_data}")
    dataloader = build_dataloader(
        data_dir=config.train_data,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
    )
    print(f"Dataset size: {len(dataloader.dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")

    # Load model
    print(f"\nLoading model from: {config.checkpoint}")
    model = load_model(config.checkpoint, device=device)
    trainable_params = list(select_lora_params(model))
    num_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(trainable_params, lr=config.learning_rate)

    # Run training
    print("\nStarting training...\n")
    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        config=config,
        dirs=dirs,
        writer=writer,
        device=device,
    )

    # Cleanup
    writer.close()
    print("\nTraining session complete.")


if __name__ == "__main__":
    main()
