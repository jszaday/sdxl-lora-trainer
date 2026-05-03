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
from .utils import resolve_adapter_spec, resolve_lr_scheduler_spec, set_seed


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
    optim_group.add_argument(
        "--lr_scheduler",
        "--lr_sched",
        "--lr-sched",
        type=str,
        default="constant",
        help="Learning rate scheduler spec, e.g. constant_with_warmup(warmup_steps=100)",
    )
    optim_group.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="Enable min-SNR loss weighting when set (e.g., 5.0)",
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
        "--num-buckets",
        type=int,
        default=0,
        help="Number of aspect ratio buckets: 0=auto/all, 1=fixed size, N=top N (default: 0)",
    )
    data_group.add_argument(
        "--train-width",
        type=int,
        default=1024,
        help="Training width when num-buckets=1 (fixed size mode) (default: 1024)",
    )
    data_group.add_argument(
        "--train-height",
        type=int,
        default=1024,
        help="Training height when num-buckets=1 (fixed size mode) (default: 1024)",
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
        choices=["simple", "normal", "karras", "exponential", "sgm_uniform"],
        help="Noise scheduler - simple, normal, karras, exponential, sgm_uniform (default: normal)",
    )
    sampling_group.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=[
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_2m_sde",
            "dpmpp_sde",
            "lms",
            "pndm",
            "ddim",
        ],
        help="Sampler algorithm (default: euler)",
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
    sampling_group.add_argument(
        "--enable_prompt_weighting",
        dest="enable_prompt_weighting",
        action="store_true",
        default=True,
        help="Enable A1111/ComfyUI-style prompt weighting: (text), ((text)), (text:1.5)",
    )
    sampling_group.add_argument(
        "--disable_prompt_weighting",
        dest="enable_prompt_weighting",
        action="store_false",
        help="Disable prompt weighting (treat parentheses as literal text)",
    )

    # Training weighting
    parser.add_argument(
        "--enable_training_prompt_weighting",
        action="store_true",
        default=False,
        help="Enable A1111/ComfyUI-style prompt weighting for training captions",
    )

    # LoRA arguments
    lora_group = parser.add_argument_group("LoRA arguments")
    lora_group.add_argument(
        "--adapter",
        type=str,
        default="lora",
        help="Adapter spec, e.g. lora(rank=16,alpha=16) or locon(rank=16,alpha=16)",
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
        "--torch_compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for the UNet (takes ~5-10m to compile, but faster training)",
    )
    misc_group.add_argument(
        "--tf32",
        action="store_true",
        default=True,
        help="Enable TF32 for faster matmuls on Ampere+ GPUs (default: True)",
    )
    misc_group.add_argument(
        "--no-tf32",
        action="store_false",
        dest="tf32",
        help="Disable TF32 and use full float32 for matmuls (slower, more precise)",
    )
    misc_group.add_argument(
        "--low_vram",
        action="store_true",
        default=False,
        help="Enable memory-saving bundle (grad checkpointing + 8-bit optimizer)",
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

    # Resolve spec-style arguments
    try:
        lr_scheduler, lr_warmup_steps, lr_num_cycles, lr_power = resolve_lr_scheduler_spec(
            args.lr_scheduler,
            warmup_steps=TrainingConfig.lr_warmup_steps,
            num_cycles=TrainingConfig.lr_num_cycles,
            power=TrainingConfig.lr_power,
        )
        adapter_params = resolve_adapter_spec(
            args.adapter,
            lora_rank=TrainingConfig.lora_rank,
            lora_alpha=TrainingConfig.lora_alpha,
            lycoris_algo=TrainingConfig.lycoris_algo,
            lycoris_dim=TrainingConfig.lycoris_dim,
            lycoris_alpha=TrainingConfig.lycoris_alpha,
            lycoris_factor=TrainingConfig.lycoris_factor,
            lycoris_dropout=TrainingConfig.lycoris_dropout,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    adapter = adapter_params.get("adapter", args.adapter)
    lora_rank = adapter_params.get("lora_rank", TrainingConfig.lora_rank)
    lora_alpha = adapter_params.get("lora_alpha", TrainingConfig.lora_alpha)
    lycoris_algo = adapter_params.get("lycoris_algo", TrainingConfig.lycoris_algo)
    lycoris_dim = adapter_params.get("lycoris_dim", TrainingConfig.lycoris_dim)
    lycoris_alpha = adapter_params.get("lycoris_alpha", TrainingConfig.lycoris_alpha)
    lycoris_factor = adapter_params.get("lycoris_factor", TrainingConfig.lycoris_factor)
    lycoris_dropout = adapter_params.get("lycoris_dropout", TrainingConfig.lycoris_dropout)

    # Bundle low-VRAM optimizations
    optimizer = args.optimizer
    gradient_checkpointing = args.gradient_checkpointing
    if args.low_vram:
        gradient_checkpointing = True
        if optimizer == "adamw":
            optimizer = "adamw8bit"
        print("\nLow-VRAM mode enabled:")
        print("  - Gradient checkpointing: ON")
        if optimizer == "adamw8bit":
            print("  - 8-bit optimizer: adamw8bit")

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
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_warmup_steps=lr_warmup_steps,
            lr_num_cycles=lr_num_cycles,
            lr_power=lr_power,
            image_size=args.image_size,
            num_workers=args.num_workers,
            num_buckets=args.num_buckets,
            bucket_min_dim=args.bucket_min_dim,
            bucket_max_dim=args.bucket_max_dim,
            train_width=args.train_width,
            train_height=args.train_height,
            scheduler=args.scheduler,
            sampler=args.sampler,
            cfg=args.cfg,
            sampler_steps=args.sampler_steps,
            sample_prompts=args.sample_prompts,
            sample_every=args.sample_every,
            samples_per_prompt=args.samples_per_prompt,
            sample_clip_skip=args.sample_clip_skip,
            enable_prompt_weighting=args.enable_prompt_weighting,
            enable_training_prompt_weighting=args.enable_training_prompt_weighting,
            adapter=adapter,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lycoris_algo=lycoris_algo,
            lycoris_dim=lycoris_dim,
            lycoris_alpha=lycoris_alpha,
            lycoris_factor=lycoris_factor,
            lycoris_dropout=lycoris_dropout,
            seed=args.seed,
            log_every=args.log_every,
            mixed_precision=args.mixed_precision,
            resume_from=args.resume_from,
            min_snr_gamma=args.min_snr_gamma,
            low_vram=args.low_vram,
            gradient_checkpointing=gradient_checkpointing,
            torch_compile=args.torch_compile,
            tf32=args.tf32,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print configuration summary
    print(config.print_summary())

    # Create bucket configuration (always enabled)
    bucket_config = BucketConfig(
        min_dimension=config.bucket_min_dim,
        max_dimension=config.bucket_max_dim,
        num_buckets=config.num_buckets,
        train_width=config.train_width,
        train_height=config.train_height,
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
                bucket_config=bucket_config,
                device=device,
                dtype=dtype,
                # TODO: Consider a separate --preprocess-batch-size flag if decoupling is useful.
                batch_size=config.batch_size,
                enable_weighting=config.enable_training_prompt_weighting,
            )
            print("\n✓ Preprocessing complete!\n")

        print(f"\nLoading cached data from: {cache_dir}")
        dataloader = build_cached_dataloader(
            cache_dir=cache_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )
        print(f"Dataset size: {len(dataloader.dataset)} samples")
        print(f"Batches per epoch: {len(dataloader)}")
    else:
        print(f"\nLoading training data from: {config.train_data}")
        dataloader = build_dataloader(
            data_dir=config.train_data,
            batch_size=config.batch_size,
            bucket_config=bucket_config,
            num_workers=config.num_workers,
            seed=config.seed,
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
    from .optim import build_lr_scheduler, build_optimizer

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
        lycoris_dropout=config.lycoris_dropout,
    )

    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        # TODO: enable gradient checkpointing for text encoders when training on-the-fly.
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
            lycoris_dropout=config.lycoris_dropout,
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
            lycoris_dropout=config.lycoris_dropout,
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
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        name=config.lr_scheduler,
        num_training_steps=config.steps,
        num_warmup_steps=config.lr_warmup_steps,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
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
            lr_scheduler=lr_scheduler,
            device=device,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            unet_adapter=unet_adapter,
            te1_adapter=te1_adapter,
            te2_adapter=te2_adapter,
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
        lr_scheduler=lr_scheduler,
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
        unet_adapter=unet_adapter,
        te1_adapter=te1_adapter,
        te2_adapter=te2_adapter,
        base_model=config.checkpoint,
        cached_data=args.use_cached_data,
    )

    # Cleanup
    writer.close()
    print("\nTraining session complete.")


if __name__ == "__main__":
    main()
