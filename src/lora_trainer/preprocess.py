"""Preprocessing utilities to cache latents and text embeddings."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from .data import ImageFolderWithCaptions
from .model import load_text_encoders, load_vae
from .train_loop import encode_prompts


def preprocess_dataset(
    train_data: Path,
    cache_dir: Path,
    checkpoint: str,
    image_size: int = 1024,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 1,
) -> None:
    """Preprocess a dataset by caching latents and text embeddings.

    Args:
        train_data: Directory containing training images and captions
        cache_dir: Directory to save cached tensors
        checkpoint: Path to base SDXL checkpoint or HuggingFace model ID
        image_size: Image size for preprocessing
        device: Device to use for encoding
        dtype: Data type for models
        batch_size: Batch size for preprocessing (higher = faster but more VRAM)
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
        image_size=image_size,
        center_crop=True,
    )
    print(f"Found {len(dataset)} images")

    # Load VAE
    print(f"\nLoading VAE from {checkpoint}...")
    vae = load_vae(checkpoint, device=device, dtype=dtype)
    vae.eval()

    # Load text encoders
    print(f"Loading text encoders from {checkpoint}...")
    text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2 = load_text_encoders(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=None,  # No LoRA for preprocessing
        lora_alpha=None,
    )
    text_encoder_1.eval()
    text_encoder_2.eval()

    print(f"\nPreprocessing dataset (batch_size={batch_size})...")
    print(f"Cache directory: {cache_dir}")

    # Process dataset
    with torch.no_grad():
        for idx in tqdm(range(0, len(dataset), batch_size), desc="Preprocessing"):
            # Collect batch
            batch_items = []
            batch_captions = []
            batch_indices = []

            for i in range(idx, min(idx + batch_size, len(dataset))):
                item = dataset[i]
                batch_items.append(item["pixel_values"])
                batch_captions.append(item["caption"])
                batch_indices.append(i)

            # Stack pixel values
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
            )

            # Save each item in the batch
            for i, batch_idx in enumerate(batch_indices):
                # Save latent
                latent_path = latents_dir / f"{batch_idx:06d}.pt"
                torch.save(latents[i].cpu(), latent_path)

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
        "image_size": image_size,
        "vae_scaling_factor": vae.config.scaling_factor,
        "checkpoint": checkpoint,
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
