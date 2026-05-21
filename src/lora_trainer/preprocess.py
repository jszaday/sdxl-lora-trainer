"""Preprocessing utilities to cache latents and text embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from .bucketing import BucketConfig
from .data import ImageFolderWithCaptions
from .model import load_text_encoders, load_vae
from .train_loop import encode_prompts


def preprocess_dataset(
    train_data: Path,
    cache_dir: Path,
    checkpoint: str,
    bucket_config: BucketConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 1,
    enable_weighting: bool = False,
) -> None:
    """Preprocess a dataset by caching latents and text embeddings.

    Args:
        train_data: Directory containing training images and captions
        cache_dir: Directory to save cached tensors
        checkpoint: Path to base SDXL checkpoint or HuggingFace model ID
        device: Device to use for encoding
        dtype: Data type for models
        batch_size: Batch size for preprocessing (higher = faster but more VRAM)
        bucket_config: Bucket configuration (bucketing always enabled)
        enable_weighting: Whether to parse and apply prompt weights
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for cached data
    latents_dir = cache_dir / "latents"
    embeds_dir = cache_dir / "embeds"
    latents_dir.mkdir(exist_ok=True)
    embeds_dir.mkdir(exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {train_data}...")
    dataset = ImageFolderWithCaptions(
        data_dir=train_data,
        bucket_config=bucket_config,
    )
    print(f"Found {len(dataset)} images")
    if bucket_config.num_buckets == 1:
        print(
            f"Using fixed size training: {bucket_config.train_width}x{bucket_config.train_height}"
        )
    else:
        print(f"Using aspect-ratio bucketing with {len(bucket_config.buckets)} buckets")

    # Load VAE
    print(f"\nLoading VAE from {checkpoint}...")
    vae = load_vae(checkpoint, device=device, dtype=dtype)
    vae.eval()

    # Load text encoders
    print(f"Loading text encoders from {checkpoint}...")
    (
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        _,
    ) = load_text_encoders(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=None,  # No LoRA for preprocessing
        lora_alpha=None,
    )
    text_encoder_1.eval()
    text_encoder_2.eval()

    # When multiple buckets are in play (auto/all or top N), images can differ in size.
    # In that case we need batch_size=1 to avoid stacking mismatched tensors.
    if bucket_config.num_buckets != 1 and batch_size > 1:
        print(
            f"\nNote: Using {len(bucket_config.buckets)} buckets, processing images one at a time "
            f"(batch_size forced to 1 from {batch_size})"
        )
        batch_size = 1

    print(f"\nPreprocessing dataset (batch_size={batch_size})...")
    print(f"Cache directory: {cache_dir}")

    # Track latent shapes for efficient batching later
    latent_shapes = []

    # Process dataset
    with torch.no_grad():
        for idx in tqdm(range(0, len(dataset), batch_size), desc="Preprocessing"):
            # Collect batch
            batch_items = []
            batch_captions = []
            batch_time_ids = []
            batch_indices = []

            for i in range(idx, min(idx + batch_size, len(dataset))):
                item = dataset[i]
                batch_items.append(item["pixel_values"])
                batch_captions.append(item["caption"])
                batch_indices.append(i)
                # Extract time_ids if available (bucketing enabled)
                if "time_ids" in item:
                    batch_time_ids.append(item["time_ids"])

            # Stack pixel values (all same size when batch_size=1 or no bucketing)
            pixel_values = torch.stack(batch_items).to(device).to(dtype)

            # Encode images to latents
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Encode captions to embeddings
            prompt_embeds, pooled_embeds = encode_prompts(
                batch_captions,
                text_encoder_1,
                text_encoder_2,
                tokenizer_1,
                tokenizer_2,
                device,
                enable_weighting=enable_weighting,
            )

            # Save each item in the batch
            for i, batch_idx in enumerate(batch_indices):
                # Save latent (always as dict with time_ids)
                latent_path = latents_dir / f"{batch_idx:06d}.pt"
                torch.save(
                    {
                        "latent": latents[i].cpu(),
                        "time_ids": batch_time_ids[i],
                    },
                    latent_path,
                )
                # Track latent shape (H, W) for efficient batching
                latent_shapes.append((latents[i].shape[1], latents[i].shape[2]))

                # Save embeddings (both prompt and pooled)
                embed_path = embeds_dir / f"{batch_idx:06d}.pt"
                torch.save(
                    {
                        "prompt_embeds": prompt_embeds[i].cpu(),
                        "pooled_embeds": pooled_embeds[i].cpu(),
                    },
                    embed_path,
                )

    # Save metadata
    metadata_path = cache_dir / "metadata.pt"
    metadata = {
        "num_samples": len(dataset),
        "vae_scaling_factor": vae.config.scaling_factor,
        "checkpoint": checkpoint,
        "num_buckets": bucket_config.num_buckets,
        "latent_shapes": latent_shapes,
    }
    torch.save(metadata, metadata_path)

    print("\nPreprocessing complete!")
    print(f"  Latents: {latents_dir}")
    print(f"  Embeddings: {embeds_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Total samples: {len(dataset)}")

    # Clean up
    del vae, text_encoder_1, text_encoder_2
    torch.cuda.empty_cache()
