"""Dataset and DataLoader utilities for training."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from .bucketing import (
    BucketConfig,
    calculate_time_ids,
    get_bucket_for_image,
    resize_and_crop_to_bucket,
)

logger = logging.getLogger(__name__)


class ImageFolderWithCaptions(Dataset):
    """Dataset for images with optional text captions.

    Expects a directory structure:
        /path/to/images/
            image1.jpg
            image1.txt  (optional caption)
            image2.png
            image2.txt  (optional caption)
            ...

    If a .txt file with the same basename exists, it's loaded as the caption.
    Otherwise, the caption defaults to an empty string.
    """

    def __init__(
        self,
        data_dir: Path,
        image_size: int = 1024,
        center_crop: bool = True,
        bucket_config: BucketConfig | None = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing image files
            image_size: Target size for images (will be resized and/or cropped)
            center_crop: Whether to center crop images to square
            bucket_config: Optional bucket configuration for aspect-ratio bucketing
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.center_crop = center_crop
        self.bucket_config = bucket_config

        # Collect all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.image_paths: list[Path] = []
        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() in image_extensions:
                self.image_paths.append(path)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Initialize bucketing if enabled
        self.use_bucketing = bucket_config is not None and bucket_config.enabled
        if self.use_bucketing:
            self._initialize_bucketing()
        else:
            # Build traditional transforms for fixed-size training
            transform_list = []
            if center_crop:
                transform_list.append(transforms.CenterCrop(min(image_size, image_size)))
            transform_list.extend(
                [
                    transforms.Resize(
                        image_size, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
                ]
            )
            self.transform = transforms.Compose(transform_list)

    def _initialize_bucketing(self) -> None:
        """Scan all images and assign to aspect ratio buckets.

        Stores metadata for each image including bucket assignment, original size,
        and crop coordinates for efficient loading during training.
        """
        logger.info(f"Initializing bucketing for {len(self.image_paths)} images...")
        self.image_metadata = []
        bucket_counts = defaultdict(int)

        for path in self.image_paths:
            # Open image to get dimensions (without loading full data)
            with Image.open(path) as img:
                bucket = get_bucket_for_image(img, self.bucket_config.buckets)
                self.image_metadata.append(
                    {
                        "path": path,
                        "original_size": img.size,  # (width, height)
                        "bucket": bucket,
                    }
                )
                bucket_counts[f"{bucket.width}x{bucket.height}"] += 1

        # Log bucket distribution
        logger.info("Bucket distribution:")
        for bucket_size, count in sorted(bucket_counts.items()):
            logger.info(f"  {bucket_size}: {count} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, any]:
        """Get a single item from the dataset.

        Returns:
            Dictionary with:
            - 'pixel_values': Normalized image tensor [C, H, W]
            - 'caption': Text caption string
            - 'time_ids': SDXL time_ids (if bucketing enabled)
            - 'bucket': Bucket info (if bucketing enabled)
        """
        if self.use_bucketing:
            # Bucketing path: use bucket-aware loading
            metadata = self.image_metadata[idx]
            image_path = metadata["path"]

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Resize and crop to bucket dimensions
            bucket = metadata["bucket"]
            image, crop_coords = resize_and_crop_to_bucket(image, bucket)

            # Apply minimal transforms (normalize only)
            pixel_values = transforms.ToTensor()(image)
            pixel_values = transforms.Normalize([0.5], [0.5])(pixel_values)

            # Calculate SDXL time_ids
            time_ids = calculate_time_ids(
                original_size=metadata["original_size"],
                crop_coords=crop_coords,
                target_size=(bucket.height, bucket.width),
            )

            # Load caption
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text().strip()
            else:
                caption = ""

            return {
                "pixel_values": pixel_values,
                "caption": caption,
                "time_ids": time_ids,
                "bucket": bucket,
            }
        else:
            # Traditional path: fixed-size loading
            image_path = self.image_paths[idx]

            # Load image
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.transform(image)

            # Load caption if exists
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text().strip()
            else:
                caption = ""

            return {
                "pixel_values": pixel_values,
                "caption": caption,
            }


class BucketBatchSampler(Sampler):
    """Sampler that groups samples by aspect ratio bucket for efficient batching.

    This sampler ensures that all images in a batch come from the same bucket,
    which guarantees they have the same dimensions and can be stacked without padding.
    """

    def __init__(
        self,
        dataset: ImageFolderWithCaptions,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """Initialize the bucket batch sampler.

        Args:
            dataset: Dataset with bucketing enabled
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within buckets
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if not dataset.use_bucketing:
            raise ValueError("BucketBatchSampler requires a dataset with bucketing enabled")

        # Group indices by bucket
        self.bucket_indices = defaultdict(list)
        for idx in range(len(dataset)):
            bucket = dataset.image_metadata[idx]["bucket"]
            bucket_key = (bucket.width, bucket.height)
            self.bucket_indices[bucket_key].append(idx)

        # Calculate total number of batches
        self.num_batches = 0
        for indices in self.bucket_indices.values():
            if self.drop_last:
                self.num_batches += len(indices) // self.batch_size
            else:
                self.num_batches += (len(indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches grouped by bucket."""
        # Create batches for each bucket
        all_batches = []
        for _bucket_key, indices in self.bucket_indices.items():
            # Shuffle indices within bucket if requested
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            # Create batches of batch_size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batch order if requested
        if self.shuffle:
            random.shuffle(all_batches)

        # Yield batches
        for batch in all_batches:
            yield batch

    def __len__(self):
        """Return the number of batches."""
        return self.num_batches


def bucket_aware_collate(batch: list[dict]) -> dict:
    """Custom collate function for bucketed batches.

    All items in the batch should have the same dimensions (guaranteed by BucketBatchSampler).
    This function stacks the tensors and returns the batch dictionary.

    Args:
        batch: List of dataset items (each a dict)

    Returns:
        Batch dictionary with stacked tensors
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")

    # Verify all items in batch have same dimensions (defensive check)
    first_shape = batch[0]["pixel_values"].shape
    for item in batch:
        if item["pixel_values"].shape != first_shape:
            raise ValueError(
                f"Mixed bucket sizes in batch: {first_shape} vs {item['pixel_values'].shape}. "
                "This indicates a bug in the BucketBatchSampler."
            )

    # Stack tensors
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]
    time_ids = torch.tensor([item["time_ids"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "caption": captions,
        "time_ids": time_ids,
    }


class CachedLatentsDataset(Dataset):
    """Dataset for pre-cached latents and text embeddings.

    Expects a cache directory structure created by preprocess.py:
        /path/to/cache/
            metadata.pt
            latents/
                000000.pt
                000001.pt
                ...
            embeds/
                000000.pt
                000001.pt
                ...
    """

    def __init__(self, cache_dir: Path):
        """Initialize dataset from cached tensors.

        Args:
            cache_dir: Directory containing cached latents and embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.latents_dir = self.cache_dir / "latents"
        self.embeds_dir = self.cache_dir / "embeds"

        # Load metadata
        metadata_path = self.cache_dir / "metadata.pt"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        self.metadata = torch.load(metadata_path)
        self.num_samples = self.metadata["num_samples"]

        # Verify cache exists
        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory not found: {self.latents_dir}")
        if not self.embeds_dir.exists():
            raise ValueError(f"Embeddings directory not found: {self.embeds_dir}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item from the cached dataset.

        Returns:
            Dictionary with:
            - 'latents': Pre-encoded VAE latents [C, H, W]
            - 'prompt_embeds': Text encoder embeddings
            - 'pooled_embeds': Pooled text embeddings
            - 'time_ids': SDXL time_ids (if available, else default)
        """
        # Load latent
        latent_path = self.latents_dir / f"{idx:06d}.pt"
        latent_data = torch.load(latent_path)

        # Handle both old format (just tensor) and new format (dict with time_ids)
        if isinstance(latent_data, dict):
            latents = latent_data["latent"]
            time_ids = latent_data.get("time_ids", [1024, 1024, 0, 0, 1024, 1024])
        else:
            # Backward compatibility: old caches only have tensor
            latents = latent_data
            time_ids = [1024, 1024, 0, 0, 1024, 1024]

        # Load embeddings
        embed_path = self.embeds_dir / f"{idx:06d}.pt"
        embeds = torch.load(embed_path)

        return {
            "latents": latents,
            "prompt_embeds": embeds["prompt_embeds"],
            "pooled_embeds": embeds["pooled_embeds"],
            "time_ids": torch.tensor(time_ids),
        }


def build_dataloader(
    data_dir: Path,
    batch_size: int,
    image_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
    center_crop: bool = True,
    bucket_config: BucketConfig | None = None,
) -> DataLoader:
    """Build a DataLoader for training.

    Args:
        data_dir: Directory containing training images
        batch_size: Batch size
        image_size: Target image size (used when bucketing is disabled)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the dataset
        center_crop: Whether to center crop images (used when bucketing is disabled)
        bucket_config: Optional bucket configuration for aspect-ratio bucketing

    Returns:
        Configured DataLoader
    """
    dataset = ImageFolderWithCaptions(
        data_dir=data_dir,
        image_size=image_size,
        center_crop=center_crop,
        bucket_config=bucket_config,
    )

    loader_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if torch.cuda.is_available() and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    # Use custom sampler and collate if bucketing enabled
    if bucket_config and bucket_config.enabled:
        sampler = BucketBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
        loader_kwargs["batch_sampler"] = sampler
        loader_kwargs["collate_fn"] = bucket_aware_collate
    else:
        # Traditional fixed-size batching
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
        loader_kwargs["drop_last"] = True

    dataloader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    return dataloader


def build_cached_dataloader(
    cache_dir: Path,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader for cached latents and embeddings.

    Args:
        cache_dir: Directory containing cached tensors
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the dataset

    Returns:
        Configured DataLoader
    """
    dataset = CachedLatentsDataset(cache_dir=cache_dir)

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,  # Drop incomplete batches for consistent batch sizes
    }
    if torch.cuda.is_available() and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    return dataloader
