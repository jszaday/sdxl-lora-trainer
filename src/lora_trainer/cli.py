"""CLI entry point for SDXL LoRA/LyCORIS training."""

import argparse
import sys
from pathlib import Path

import torch

from .bucketing import BucketConfig
from .config import TrainingConfig
from .data import build_cached_dataloader, build_dataloader
from .logging import create_run_dirs, init_tensorboard, log_hparams, write_config_yaml
from .preprocess import preprocess_dataset
from .train_loop import load_checkpoint, train
from .utils import set_seed


def _resolve_resume_path(path: Path) -> Path:
    """Resolve a resume target that may be a checkpoint file or directory."""
    if path.is_file():
        return path

    # If a directory was provided, pick the newest .pt checkpoint by mtime.
    candidates = sorted(path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise ValueError(f"No checkpoint files found in {path}")
    return candidates[-1]


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
    optim_group.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="Optimizer spec, e.g., adamw or prodigy(lr=1e-4, weight_decay=0)",
    )

    # Data arguments
    data_group = parser.add_argument_group("data arguments")
    data_group.add_argument(
        "--use-cached-data",
        action="store_true",
        default=True,
        help="Use pre-cached latents/embeddings (default: True, saves VRAM)",
    )
    data_group.add_argument(
        "--no-cached-data",
        action="store_false",
        dest="use_cached_data",
        help="Disable caching and encode on-the-fly (uses more VRAM)",
    )
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
    data_group.add_argument(
        "--enable-buckets",
        action="store_true",
        default=True,
        help="Enable aspect-ratio bucketing (default: True)",
    )
    data_group.add_argument(
        "--no-buckets",
        action="store_false",
        dest="enable_buckets",
        help="Disable aspect-ratio bucketing (use fixed image size)",
    )
    data_group.add_argument(
        "--bucket-min-dim",
        type=int,
        default=512,
        help="Minimum bucket dimension in pixels (default: 512)",
    )
    data_group.add_argument(
        "--bucket-max-dim",
        type=int,
        default=2048,
        help="Maximum bucket dimension in pixels (default: 2048)",
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
    sampling_group.add_argument(
        "--sample_clip_skip",
        type=int,
        default=1,
        help="Clip skip for text_encoder_1 hidden states (1 = penultimate, like current)",
    )

    # LoRA arguments
    lora_group = parser.add_argument_group("LoRA arguments")
    lora_group.add_argument(
        "--adapter",
        type=str,
        default="lora",
        choices=["lora", "lycoris"],
        help="Adapter backend to train (default: lora)",
    )
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
        "--lycoris_algo",
        type=str,
        default="lokr",
        help="LyCORIS algorithm to use when adapter=lycoris (default: lokr)",
    )
    lora_group.add_argument(
        "--lycoris_dim",
        type=int,
        default=None,
        help="LyCORIS linear_dim (defaults to lora_rank when omitted)",
    )
    lora_group.add_argument(
        "--lycoris_alpha",
        type=float,
        default=None,
        help="LyCORIS linear_alpha (defaults to lora_alpha when omitted)",
    )
    lora_group.add_argument(
        "--lycoris_factor",
        type=int,
        default=-1,
        help="LyCORIS factorization factor (default: -1 for auto)",
    )

    # Misc arguments
    misc_group = parser.add_argument_group("misc arguments")
    misc_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing to save VRAM (slower but uses ~15-20GB less)",
    )
    misc_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda', 'cpu', 'cuda:0', 'mps'). "
        "If not specified, automatically selects cuda if available, otherwise cpu.",
    )
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    misc_group.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log metrics to TensorBoard every N steps. "
        "Higher values reduce GPU sync overhead (default: 10)",
    )
    misc_group.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode (default: fp16)",
    )
    misc_group.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Checkpoint file or directory to resume training from",
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
            optimizer=args.optimizer,
            image_size=args.image_size,
            num_workers=args.num_workers,
            enable_buckets=args.enable_buckets,
            bucket_min_dim=args.bucket_min_dim,
            bucket_max_dim=args.bucket_max_dim,
            scheduler=args.scheduler,
            sampler=args.sampler,
            cfg=args.cfg,
            sampler_steps=args.sampler_steps,
            sample_prompts=args.sample_prompts,
            sample_every=args.sample_every,
            samples_per_prompt=args.samples_per_prompt,
            sample_clip_skip=args.sample_clip_skip,
            adapter=args.adapter,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lycoris_algo=args.lycoris_algo,
            lycoris_dim=args.lycoris_dim,
            lycoris_alpha=args.lycoris_alpha,
            lycoris_factor=args.lycoris_factor,
            seed=args.seed,
            log_every=args.log_every,
            mixed_precision=args.mixed_precision,
            resume_from=args.resume_from,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print configuration summary
    print(config.print_summary())

    # Create bucket configuration
    bucket_config = None
    if config.enable_buckets:
        bucket_config = BucketConfig(
            enabled=True,
            min_dimension=config.bucket_min_dim,
            max_dimension=config.bucket_max_dim,
            base_pixel_count=config.bucket_base_pixels,
        )

    # Set random seed
    set_seed(config.seed)

    # Determine device
    if args.device is not None:
        device = args.device
        print(f"\nUsing device: {device}")
    elif torch.cuda.is_available():
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
    config_yaml = write_config_yaml(dirs["root"], config)
    print(f"  - Config:      {config_yaml}")

    # Initialize TensorBoard
    writer = init_tensorboard(dirs["tb"])
    log_hparams(writer, config)

    # Determine cache directory
    cache_dir = dirs["root"] / "cache"

    # Build dataloader
    if args.use_cached_data:
        # Check if cache exists, create if needed
        if not (cache_dir / "metadata.pt").exists():
            print("\nCache not found. Preprocessing dataset...")
            print("This is a one-time operation that will save significant VRAM during training.")

            # Determine dtype for preprocessing
            if config.mixed_precision == "fp16":
                dtype = torch.float16
            elif config.mixed_precision == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            preprocess_dataset(
                train_data=config.train_data,
                cache_dir=cache_dir,
                checkpoint=config.checkpoint,
                image_size=config.image_size,
                device=device,
                dtype=dtype,
                batch_size=4,  # Use batch size 4 for preprocessing
                bucket_config=bucket_config,
            )
            print("\n✓ Preprocessing complete!\n")

        print(f"\nLoading cached data from: {cache_dir}")
        dataloader = build_cached_dataloader(
            cache_dir=cache_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        print(f"Dataset size: {len(dataloader.dataset)} samples")
        print(f"Batches per epoch: {len(dataloader)}")
    else:
        print(f"\nLoading training data from: {config.train_data}")
        dataloader = build_dataloader(
            data_dir=config.train_data,
            batch_size=config.batch_size,
            image_size=config.image_size,
            num_workers=config.num_workers,
            bucket_config=bucket_config,
        )
        print(f"Dataset size: {len(dataloader.dataset)} images")
        print(f"Batches per epoch: {len(dataloader)}")

    # Determine dtype based on mixed precision setting
    if config.mixed_precision == "fp16":
        dtype = torch.float16
    elif config.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load model components
    print(f"\nLoading model from: {config.checkpoint}")

    # Import model loading functions
    from .model import load_sdxl_unet, load_text_encoders, load_vae, select_lora_params
    from .optim import build_optimizer

    model, unet_adapter = load_sdxl_unet(
        config.checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        adapter=config.adapter,
        lycoris_dim=config.lycoris_dim,
        lycoris_alpha=config.lycoris_alpha,
        lycoris_algo=config.lycoris_algo,
        lycoris_factor=config.lycoris_factor,
    )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled (saves VRAM, ~20% slower)")

    # Load VAE and text encoders based on caching mode
    if args.use_cached_data:
        print("\nUsing cached embeddings for training")
        print("Loading VAE and text encoders for sampling only (kept on CPU to save VRAM)...")
        vae = load_vae(config.checkpoint, device="cpu", dtype=dtype)
        vae.eval()

        te1_adapter = te2_adapter = None
        # Load text encoders without LoRA (just for sampling)
        text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, _ = load_text_encoders(
            config.checkpoint,
            device="cpu",
            dtype=dtype,
            lora_rank=None,  # No LoRA needed for sampling
            lora_alpha=None,
            adapter=config.adapter,
            lycoris_dim=config.lycoris_dim,
            lycoris_alpha=config.lycoris_alpha,
            lycoris_algo=config.lycoris_algo,
            lycoris_factor=config.lycoris_factor,
        )
    else:
        print("\nLoading VAE...")
        vae = load_vae(config.checkpoint, device=device, dtype=dtype)

        print("Loading text encoders...")
        (
            text_encoder_1,
            text_encoder_2,
            tokenizer_1,
            tokenizer_2,
            (te1_adapter, te2_adapter),
        ) = load_text_encoders(
            config.checkpoint,
            device=device,
            dtype=dtype,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            adapter=config.adapter,
            lycoris_dim=config.lycoris_dim,
            lycoris_alpha=config.lycoris_alpha,
            lycoris_algo=config.lycoris_algo,
            lycoris_factor=config.lycoris_factor,
        )

    # Collect adapter parameters - UNet only when using cached data
    trainable_params = []
    te1_params = te2_params = 0

    if config.adapter == "lycoris":
        if unet_adapter is None:
            raise RuntimeError("LyCORIS adapter for UNet was not created")
        unet_param_list = list(unet_adapter.parameters())
        trainable_params.extend(unet_param_list)
        unet_params = sum(p.numel() for p in unet_param_list)

        if not args.use_cached_data and text_encoder_1 is not None and text_encoder_2 is not None:
            te1_list = list(te1_adapter.parameters()) if te1_adapter is not None else []
            te2_list = list(te2_adapter.parameters()) if te2_adapter is not None else []
            trainable_params.extend(te1_list)
            trainable_params.extend(te2_list)
            te1_params = sum(p.numel() for p in te1_list)
            te2_params = sum(p.numel() for p in te2_list)
        else:
            te1_params = 0
            te2_params = 0
    else:
        trainable_params = list(select_lora_params(model))
        unet_params = sum(p.numel() for p in select_lora_params(model))

        if not args.use_cached_data and text_encoder_1 is not None and text_encoder_2 is not None:
            trainable_params.extend(select_lora_params(text_encoder_1))
            trainable_params.extend(select_lora_params(text_encoder_2))
            te1_params = sum(p.numel() for p in select_lora_params(text_encoder_1))
            te2_params = sum(p.numel() for p in select_lora_params(text_encoder_2))
        else:
            te1_params = 0
            te2_params = 0

    num_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable {config.adapter} parameters: {num_params:,}")
    print(f"  UNet: {unet_params:,}, TE1: {te1_params:,}, TE2: {te2_params:,}")

    # Create optimizer
    optimizer = build_optimizer(
        trainable_params,
        spec=config.optimizer,
        base_lr=config.learning_rate,
    )

    # Resume from checkpoint if requested
    resume_step = 0
    if config.resume_from is not None:
        resume_path = _resolve_resume_path(config.resume_from)
        print(f"\nResuming from checkpoint: {resume_path}")
        resume_step = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
        )
        print(f"  Loaded global_step={resume_step}")

        if resume_step >= config.steps:
            print(
                f"Checkpoint step ({resume_step}) >= target steps ({config.steps}). Nothing to do."
            )
            return

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
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        start_step=resume_step,
    )

    # Cleanup
    writer.close()
    print("\nTraining session complete.")


if __name__ == "__main__":
    main()
