"""Dataset and DataLoader utilities for training."""

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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Drop incomplete batches for consistent batch sizes
    )

    return dataloader
