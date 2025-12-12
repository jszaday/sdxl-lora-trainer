"""Tests for dataset and dataloader functionality."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from lora_trainer.data import ImageFolderWithCaptions, build_dataloader


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with test images and captions."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create some dummy RGB images
    for i in range(5):
        img = Image.new("RGB", (256, 256), color=(i * 50, i * 50, i * 50))
        img.save(temp_dir / f"image_{i}.jpg")

        # Add captions for some images
        if i % 2 == 0:
            (temp_dir / f"image_{i}.txt").write_text(f"Caption for image {i}")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


def test_dataset_size(temp_image_dir):
    """Test that dataset size matches number of images."""
    dataset = ImageFolderWithCaptions(temp_image_dir, image_size=128)
    assert len(dataset) == 5


def test_dataset_getitem_returns_correct_keys(temp_image_dir):
    """Test that dataset returns expected dictionary keys."""
    dataset = ImageFolderWithCaptions(temp_image_dir, image_size=128)
    sample = dataset[0]

    assert "pixel_values" in sample
    assert "caption" in sample


def test_dataset_pixel_values_shape(temp_image_dir):
    """Test that pixel_values tensor has correct shape."""
    image_size = 128
    dataset = ImageFolderWithCaptions(temp_image_dir, image_size=image_size)
    sample = dataset[0]

    pixel_values = sample["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (3, image_size, image_size)  # [C, H, W]


def test_dataset_pixel_values_normalized(temp_image_dir):
    """Test that pixel values are normalized to [-1, 1]."""
    dataset = ImageFolderWithCaptions(temp_image_dir, image_size=128)
    sample = dataset[0]

    pixel_values = sample["pixel_values"]
    assert pixel_values.min() >= -1.0
    assert pixel_values.max() <= 1.0


def test_dataset_caption_with_text_file(temp_image_dir):
    """Test that captions are loaded from .txt files."""
    dataset = ImageFolderWithCaptions(temp_image_dir, image_size=128)

    # Image 0 should have a caption file
    sample = dataset[0]
    assert sample["caption"] == "Caption for image 0"


def test_dataset_caption_without_text_file(temp_image_dir):
    """Test that missing captions default to empty string."""
    dataset = ImageFolderWithCaptions(temp_image_dir, image_size=128)

    # Image 1 doesn't have a caption file
    sample = dataset[1]
    assert sample["caption"] == ""


def test_dataset_empty_directory():
    """Test that empty directory raises ValueError."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        with pytest.raises(ValueError, match="No images found"):
            ImageFolderWithCaptions(temp_dir, image_size=128)
    finally:
        shutil.rmtree(temp_dir)


def test_dataloader_batch_size(temp_image_dir):
    """Test that dataloader produces correct batch sizes."""
    batch_size = 2
    dataloader = build_dataloader(
        data_dir=temp_image_dir,
        batch_size=batch_size,
        image_size=128,
        num_workers=0,  # Use 0 for testing
    )

    batch = next(iter(dataloader))
    assert batch["pixel_values"].shape[0] == batch_size


def test_dataloader_iterations(temp_image_dir):
    """Test that dataloader produces expected number of batches."""
    batch_size = 2
    dataloader = build_dataloader(
        data_dir=temp_image_dir,
        batch_size=batch_size,
        image_size=128,
        num_workers=0,
        shuffle=False,
    )

    # 5 images with batch_size=2 and drop_last=True should give 2 batches
    batches = list(dataloader)
    assert len(batches) == 2


def test_dataloader_pin_memory_with_cuda(temp_image_dir):
    """Test that pin_memory is set when CUDA is available."""
    dataloader = build_dataloader(
        data_dir=temp_image_dir,
        batch_size=2,
        image_size=128,
        num_workers=0,
    )

    # pin_memory should be True if CUDA is available
    if torch.cuda.is_available():
        assert dataloader.pin_memory is True
    else:
        assert dataloader.pin_memory is False
