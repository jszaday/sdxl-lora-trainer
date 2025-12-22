"""Aspect-ratio bucketing for efficient training on variable-sized images."""

import logging
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class AspectBucket:
    """Represents an aspect ratio bucket for training images.

    Attributes:
        width: Target width in pixels (must be divisible by 8)
        height: Target height in pixels (must be divisible by 8)
        aspect_ratio: Width / height ratio
        name: Human-readable name (e.g., "1:1", "3:4", "16:9")
    """

    width: int
    height: int
    aspect_ratio: float
    name: str

    def __post_init__(self):
        """Validate bucket dimensions."""
        if self.width % 8 != 0:
            raise ValueError(f"Bucket width {self.width} must be divisible by 8")
        if self.height % 8 != 0:
            raise ValueError(f"Bucket height {self.height} must be divisible by 8")


@dataclass
class BucketConfig:
    """Configuration for aspect ratio bucketing system.

    Attributes:
        enabled: Whether bucketing is enabled
        buckets: List of available aspect ratio buckets
        min_dimension: Minimum dimension for any bucket (default: 512)
        max_dimension: Maximum dimension for any bucket (default: 2048)
        base_pixel_count: Target pixel count per image (default: 1024 * 1024)
    """

    enabled: bool = True
    buckets: list[AspectBucket] = field(default_factory=list)
    min_dimension: int = 512
    max_dimension: int = 2048
    base_pixel_count: int = 1024 * 1024

    def __post_init__(self):
        """Generate default buckets if not provided."""
        if not self.buckets:
            self.buckets = generate_buckets(
                base_pixels=self.base_pixel_count,
                min_dim=self.min_dimension,
                max_dim=self.max_dimension,
            )


def generate_buckets(
    base_pixels: int = 1024 * 1024,
    min_dim: int = 512,
    max_dim: int = 2048,
) -> list[AspectBucket]:
    """Generate a list of aspect ratio buckets with constant pixel area.

    Args:
        base_pixels: Target pixel count per bucket (default: 1024 * 1024)
        min_dim: Minimum dimension in pixels (default: 512)
        max_dim: Maximum dimension in pixels (default: 2048)

    Returns:
        List of AspectBucket objects covering common aspect ratios
    """
    buckets = []

    # Predefined bucket dimensions (all divisible by 8, ~1M pixels each)
    # Format: (width, height, ratio_name)
    bucket_specs = [
        # Very tall (1:4 to 1:3)
        (512, 2048, "1:4"),
        (512, 1984, "~1:4"),
        (576, 1792, "~1:3"),
        (576, 1728, "1:3"),
        # Tall (1:2 to 2:3)
        (640, 1536, "~2:5"),
        (704, 1408, "1:2"),
        (704, 1344, "~1:2"),
        (768, 1344, "~9:16"),
        (768, 1280, "3:5"),
        (832, 1216, "~2:3"),
        # Portrait (3:4 to 15:16)
        (832, 1152, "~3:4"),
        (896, 1152, "~4:5"),
        (896, 1088, "~5:6"),
        (960, 1088, "~8:9"),
        (960, 1024, "~15:16"),
        # Square
        (1024, 1024, "1:1"),
        # Landscape (16:15 to 4:3)
        (1024, 960, "~16:15"),
        (1088, 960, "~9:8"),
        (1088, 896, "~6:5"),
        (1152, 896, "~5:4"),
        (1152, 832, "~4:3"),
        # Wide (3:2 to 2:1)
        (1216, 832, "~3:2"),
        (1280, 768, "5:3"),
        (1344, 768, "~16:9"),
        (1344, 704, "~2:1"),
        (1408, 704, "2:1"),
        # Very wide (3:1 to 4:1)
        (1536, 640, "~5:2"),
        (1728, 576, "3:1"),
        (1792, 576, "~3:1"),
        (1984, 512, "~4:1"),
        (2048, 512, "4:1"),
    ]

    # Filter buckets by min/max dimensions
    for width, height, name in bucket_specs:
        if min_dim <= width <= max_dim and min_dim <= height <= max_dim:
            aspect_ratio = width / height
            buckets.append(
                AspectBucket(
                    width=width,
                    height=height,
                    aspect_ratio=aspect_ratio,
                    name=name,
                )
            )

    logger.info(f"Generated {len(buckets)} aspect ratio buckets")
    return buckets


def get_bucket_for_image(
    image: Image.Image,
    buckets: list[AspectBucket],
) -> AspectBucket:
    """Determine the best bucket for an image based on aspect ratio.

    Args:
        image: PIL Image to assign to a bucket
        buckets: List of available buckets

    Returns:
        The AspectBucket with the closest aspect ratio to the image
    """
    if not buckets:
        raise ValueError("No buckets available")

    # Calculate image aspect ratio
    img_width, img_height = image.size
    img_aspect = img_width / img_height

    # Find bucket with closest aspect ratio
    best_bucket = min(buckets, key=lambda b: abs(b.aspect_ratio - img_aspect))

    return best_bucket


def resize_and_crop_to_bucket(
    image: Image.Image,
    bucket: AspectBucket,
) -> tuple[Image.Image, tuple[int, int]]:
    """Resize and center-crop an image to fit a bucket.

    Strategy:
    1. Resize the image maintaining aspect ratio until one dimension matches the bucket
    2. Center crop the other dimension to exact bucket size

    Args:
        image: PIL Image to resize and crop
        bucket: Target AspectBucket

    Returns:
        Tuple of (cropped_image, crop_coordinates)
        crop_coordinates is (top, left) offset for SDXL time_ids
    """
    img_width, img_height = image.size
    target_width, target_height = bucket.width, bucket.height

    # Calculate resize dimensions (maintain aspect ratio)
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height

    if img_aspect > target_aspect:
        # Image is wider than target - match height, crop width
        resize_height = target_height
        resize_width = int(resize_height * img_aspect)
    else:
        # Image is taller than target - match width, crop height
        resize_width = target_width
        resize_height = int(resize_width / img_aspect)

    # Ensure dimensions are at least as large as target
    resize_width = max(resize_width, target_width)
    resize_height = max(resize_height, target_height)

    # Resize image
    resized = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)

    # Calculate center crop coordinates
    crop_left = (resize_width - target_width) // 2
    crop_top = (resize_height - target_height) // 2
    crop_right = crop_left + target_width
    crop_bottom = crop_top + target_height

    # Crop to exact bucket size
    cropped = resized.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Return cropped image and crop coordinates (for time_ids)
    return cropped, (crop_top, crop_left)


def calculate_time_ids(
    original_size: tuple[int, int],
    crop_coords: tuple[int, int],
    target_size: tuple[int, int],
) -> list[int]:
    """Calculate SDXL time_ids conditioning vector.

    SDXL uses time_ids to condition the model on image resolution and cropping.
    Format: [original_height, original_width, crop_top, crop_left, target_height, target_width]

    Args:
        original_size: Original image size as (width, height)
        crop_coords: Crop coordinates as (top, left)
        target_size: Target bucket size as (height, width)

    Returns:
        List of 6 integers for SDXL time_ids
    """
    orig_width, orig_height = original_size
    crop_top, crop_left = crop_coords
    target_height, target_width = target_size

    time_ids = [
        orig_height,
        orig_width,
        crop_top,
        crop_left,
        target_height,
        target_width,
    ]

    return time_ids
