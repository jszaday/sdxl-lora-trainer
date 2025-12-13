"""Dataset and DataLoader utilities for training."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing image files
            image_size: Target size for images (will be resized and/or cropped)
            center_crop: Whether to center crop images to square
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.center_crop = center_crop

        # Collect all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.image_paths: list[Path] = []
        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() in image_extensions:
                self.image_paths.append(path)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Build transforms
        transform_list = []
        if center_crop:
            transform_list.append(transforms.CenterCrop(min(image_size, image_size)))
        transform_list.extend(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, any]:
        """Get a single item from the dataset.

        Returns:
            Dictionary with:
            - 'pixel_values': Normalized image tensor [C, H, W]
            - 'caption': Text caption string
        """
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
        """
        # Load latent
        latent_path = self.latents_dir / f"{idx:06d}.pt"
        latents = torch.load(latent_path)

        # Load embeddings
        embed_path = self.embeds_dir / f"{idx:06d}.pt"
        embeds = torch.load(embed_path)

        return {
            "latents": latents,
            "prompt_embeds": embeds["prompt_embeds"],
            "pooled_embeds": embeds["pooled_embeds"],
        }


def build_dataloader(
    data_dir: Path,
    batch_size: int,
    image_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
    center_crop: bool = True,
) -> DataLoader:
    """Build a DataLoader for training.

    Args:
        data_dir: Directory containing training images
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the dataset
        center_crop: Whether to center crop images

    Returns:
        Configured DataLoader
    """
    dataset = ImageFolderWithCaptions(
        data_dir=data_dir,
        image_size=image_size,
        center_crop=center_crop,
    )

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
