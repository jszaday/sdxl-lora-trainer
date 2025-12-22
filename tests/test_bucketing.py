"""Unit tests for aspect-ratio bucketing functionality."""

import pytest
import torch
from PIL import Image

from lora_trainer.bucketing import (
    AspectBucket,
    BucketConfig,
    calculate_time_ids,
    generate_buckets,
    get_bucket_for_image,
    resize_and_crop_to_bucket,
)
from lora_trainer.data import BucketBatchSampler, bucket_aware_collate


class TestAspectBucket:
    """Test AspectBucket dataclass."""

    def test_valid_bucket_creation(self):
        """Test creating a valid bucket."""
        bucket = AspectBucket(width=1024, height=1024, aspect_ratio=1.0, name="1:1")
        assert bucket.width == 1024
        assert bucket.height == 1024
        assert bucket.aspect_ratio == 1.0
        assert bucket.name == "1:1"

    def test_invalid_width_not_divisible_by_8(self):
        """Test that bucket width must be divisible by 8."""
        with pytest.raises(ValueError, match="width.*divisible by 8"):
            AspectBucket(width=1023, height=1024, aspect_ratio=1.0, name="invalid")

    def test_invalid_height_not_divisible_by_8(self):
        """Test that bucket height must be divisible by 8."""
        with pytest.raises(ValueError, match="height.*divisible by 8"):
            AspectBucket(width=1024, height=1023, aspect_ratio=1.0, name="invalid")


class TestBucketConfig:
    """Test BucketConfig dataclass."""

    def test_default_initialization(self):
        """Test default BucketConfig initialization."""
        config = BucketConfig()
        assert config.enabled is True
        assert len(config.buckets) > 0  # Should auto-generate buckets
        assert config.min_dimension == 512
        assert config.max_dimension == 2048
        assert config.base_pixel_count == 1024 * 1024

    def test_custom_bucket_range(self):
        """Test custom bucket range."""
        config = BucketConfig(min_dimension=768, max_dimension=1536)
        assert len(config.buckets) > 0
        # All buckets should be within range
        for bucket in config.buckets:
            assert 768 <= bucket.width <= 1536
            assert 768 <= bucket.height <= 1536

    def test_disabled_bucketing(self):
        """Test disabled bucketing."""
        config = BucketConfig(enabled=False)
        assert config.enabled is False


class TestGenerateBuckets:
    """Test bucket generation."""

    def test_generates_buckets(self):
        """Test that buckets are generated."""
        buckets = generate_buckets()
        assert len(buckets) > 0
        assert all(isinstance(b, AspectBucket) for b in buckets)

    def test_all_dimensions_divisible_by_8(self):
        """Test that all bucket dimensions are divisible by 8."""
        buckets = generate_buckets()
        for bucket in buckets:
            assert bucket.width % 8 == 0, f"Width {bucket.width} not divisible by 8"
            assert bucket.height % 8 == 0, f"Height {bucket.height} not divisible by 8"

    def test_buckets_within_range(self):
        """Test that buckets respect min/max dimensions."""
        buckets = generate_buckets(min_dim=768, max_dim=1536)
        for bucket in buckets:
            assert 768 <= bucket.width <= 1536
            assert 768 <= bucket.height <= 1536

    def test_aspect_ratios_are_correct(self):
        """Test that aspect ratios are calculated correctly."""
        buckets = generate_buckets()
        for bucket in buckets:
            expected_ratio = bucket.width / bucket.height
            assert abs(bucket.aspect_ratio - expected_ratio) < 0.0001

    def test_pixel_count_roughly_constant(self):
        """Test that all buckets have roughly the same pixel count."""
        buckets = generate_buckets(base_pixels=1024 * 1024)
        target = 1024 * 1024
        for bucket in buckets:
            pixel_count = bucket.width * bucket.height
            # Allow 20% variance
            assert 0.8 * target <= pixel_count <= 1.2 * target


class TestGetBucketForImage:
    """Test bucket assignment for images."""

    def test_square_image_gets_square_bucket(self):
        """Test that square images get 1:1 bucket."""
        buckets = generate_buckets()
        img = Image.new("RGB", (1000, 1000))
        bucket = get_bucket_for_image(img, buckets)
        # Should be close to 1:1 ratio
        assert abs(bucket.aspect_ratio - 1.0) < 0.1

    def test_portrait_image_gets_portrait_bucket(self):
        """Test that portrait images get tall bucket."""
        buckets = generate_buckets()
        img = Image.new("RGB", (800, 1200))  # 2:3 portrait
        bucket = get_bucket_for_image(img, buckets)
        # Should be a portrait bucket (height > width)
        assert bucket.height > bucket.width

    def test_landscape_image_gets_landscape_bucket(self):
        """Test that landscape images get wide bucket."""
        buckets = generate_buckets()
        img = Image.new("RGB", (1600, 900))  # 16:9 landscape
        bucket = get_bucket_for_image(img, buckets)
        # Should be a landscape bucket (width > height)
        assert bucket.width > bucket.height

    def test_closest_aspect_ratio_is_selected(self):
        """Test that the closest aspect ratio bucket is selected."""
        buckets = generate_buckets()
        # Create image with known aspect ratio
        img = Image.new("RGB", (1920, 1080))  # 16:9 = 1.778
        bucket = get_bucket_for_image(img, buckets)
        # Should get a bucket close to 16:9
        img_ratio = 1920 / 1080
        ratio_diff = abs(bucket.aspect_ratio - img_ratio)
        # Should be the closest match
        for b in buckets:
            other_diff = abs(b.aspect_ratio - img_ratio)
            assert ratio_diff <= other_diff

    def test_empty_buckets_raises_error(self):
        """Test that empty bucket list raises error."""
        img = Image.new("RGB", (1000, 1000))
        with pytest.raises(ValueError, match="No buckets available"):
            get_bucket_for_image(img, [])


class TestResizeAndCropToBucket:
    """Test image resizing and cropping."""

    def test_output_matches_bucket_dimensions(self):
        """Test that output image has exact bucket dimensions."""
        bucket = AspectBucket(width=1024, height=768, aspect_ratio=1.333, name="4:3")
        img = Image.new("RGB", (2000, 1500))
        cropped, _ = resize_and_crop_to_bucket(img, bucket)
        assert cropped.size == (1024, 768)

    def test_crop_coordinates_are_returned(self):
        """Test that crop coordinates are returned."""
        bucket = AspectBucket(width=1024, height=1024, aspect_ratio=1.0, name="1:1")
        img = Image.new("RGB", (2000, 1500))
        _, crop_coords = resize_and_crop_to_bucket(img, bucket)
        assert isinstance(crop_coords, tuple)
        assert len(crop_coords) == 2
        assert all(isinstance(c, int) for c in crop_coords)

    def test_square_image_to_square_bucket(self):
        """Test square image to square bucket (minimal cropping)."""
        bucket = AspectBucket(width=1024, height=1024, aspect_ratio=1.0, name="1:1")
        img = Image.new("RGB", (2000, 2000))
        cropped, crop_coords = resize_and_crop_to_bucket(img, bucket)
        assert cropped.size == (1024, 1024)
        # Should be minimal crop (close to 0,0)
        assert crop_coords == (0, 0)

    def test_landscape_to_portrait_crops_width(self):
        """Test landscape image to portrait bucket crops width."""
        bucket = AspectBucket(width=768, height=1024, aspect_ratio=0.75, name="3:4")
        img = Image.new("RGB", (2000, 1000))  # Wide landscape
        cropped, crop_coords = resize_and_crop_to_bucket(img, bucket)
        assert cropped.size == (768, 1024)
        # Should crop horizontally (left coord > 0)
        crop_top, crop_left = crop_coords
        assert crop_left > 0

    def test_portrait_to_landscape_crops_height(self):
        """Test portrait image to landscape bucket crops height."""
        bucket = AspectBucket(width=1024, height=768, aspect_ratio=1.333, name="4:3")
        img = Image.new("RGB", (1000, 2000))  # Tall portrait
        cropped, crop_coords = resize_and_crop_to_bucket(img, bucket)
        assert cropped.size == (1024, 768)
        # Should crop vertically (top coord > 0)
        crop_top, crop_left = crop_coords
        assert crop_top > 0


class TestCalculateTimeIds:
    """Test SDXL time_ids calculation."""

    def test_time_ids_format(self):
        """Test that time_ids has correct format."""
        time_ids = calculate_time_ids(
            original_size=(2000, 1500),
            crop_coords=(100, 50),
            target_size=(1024, 768),
        )
        assert len(time_ids) == 6
        assert all(isinstance(x, int) for x in time_ids)

    def test_time_ids_values(self):
        """Test that time_ids contains correct values."""
        orig_size = (2000, 1500)  # width, height
        crop_coords = (100, 50)  # top, left
        target_size = (1024, 768)  # height, width

        time_ids = calculate_time_ids(orig_size, crop_coords, target_size)

        # Format: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        assert time_ids[0] == 1500  # original height
        assert time_ids[1] == 2000  # original width
        assert time_ids[2] == 100  # crop top
        assert time_ids[3] == 50  # crop left
        assert time_ids[4] == 1024  # target height
        assert time_ids[5] == 768  # target width

    def test_no_crop_time_ids(self):
        """Test time_ids when no cropping occurs."""
        time_ids = calculate_time_ids(
            original_size=(1024, 1024),
            crop_coords=(0, 0),
            target_size=(1024, 1024),
        )
        # Format: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        assert time_ids == [1024, 1024, 0, 0, 1024, 1024]


class TestBucketBatchSampler:
    """Test BucketBatchSampler (integration with ImageFolderWithCaptions)."""

    def test_sampler_requires_bucketing_enabled(self, tmp_path):
        """Test that sampler requires bucketing to be enabled."""
        from lora_trainer.data import ImageFolderWithCaptions

        # Create dummy images
        for i in range(10):
            img = Image.new("RGB", (1024, 1024))
            img.save(tmp_path / f"img_{i}.png")

        # Create dataset without bucketing
        dataset = ImageFolderWithCaptions(tmp_path, bucket_config=None)

        with pytest.raises(
            ValueError, match="BucketBatchSampler requires.*bucketing enabled"
        ):
            BucketBatchSampler(dataset, batch_size=2)

    def test_sampler_groups_by_bucket(self, tmp_path):
        """Test that sampler groups samples by bucket."""
        from lora_trainer.data import ImageFolderWithCaptions

        # Create images with different aspect ratios
        # Square images
        for i in range(5):
            img = Image.new("RGB", (1024, 1024))
            img.save(tmp_path / f"square_{i}.png")

        # Landscape images
        for i in range(5):
            img = Image.new("RGB", (1600, 900))
            img.save(tmp_path / f"landscape_{i}.png")

        # Create dataset with bucketing
        bucket_config = BucketConfig(enabled=True)
        dataset = ImageFolderWithCaptions(tmp_path, bucket_config=bucket_config)

        sampler = BucketBatchSampler(dataset, batch_size=2, shuffle=False)

        # Collect batches
        batches = list(sampler)
        assert len(batches) > 0

        # Verify each batch contains indices from same bucket
        for batch_indices in batches:
            buckets_in_batch = set()
            for idx in batch_indices:
                bucket = dataset.image_metadata[idx]["bucket"]
                buckets_in_batch.add((bucket.width, bucket.height))
            # All items in batch should have same bucket
            assert len(buckets_in_batch) == 1

    def test_sampler_respects_batch_size(self, tmp_path):
        """Test that sampler creates batches of correct size."""
        from lora_trainer.data import ImageFolderWithCaptions

        # Create 10 square images
        for i in range(10):
            img = Image.new("RGB", (1024, 1024))
            img.save(tmp_path / f"img_{i}.png")

        bucket_config = BucketConfig(enabled=True)
        dataset = ImageFolderWithCaptions(tmp_path, bucket_config=bucket_config)

        sampler = BucketBatchSampler(dataset, batch_size=3, drop_last=True)

        batches = list(sampler)
        for batch in batches:
            assert len(batch) == 3  # All batches should be size 3

    def test_sampler_drop_last(self, tmp_path):
        """Test that sampler drops incomplete batches when drop_last=True."""
        from lora_trainer.data import ImageFolderWithCaptions

        # Create 10 images (won't divide evenly by batch_size=3)
        for i in range(10):
            img = Image.new("RGB", (1024, 1024))
            img.save(tmp_path / f"img_{i}.png")

        bucket_config = BucketConfig(enabled=True)
        dataset = ImageFolderWithCaptions(tmp_path, bucket_config=bucket_config)

        sampler_drop = BucketBatchSampler(dataset, batch_size=3, drop_last=True)
        sampler_keep = BucketBatchSampler(dataset, batch_size=3, drop_last=False)

        batches_drop = list(sampler_drop)
        batches_keep = list(sampler_keep)

        # drop_last should have fewer batches
        assert len(batches_drop) <= len(batches_keep)


class TestBucketAwareCollate:
    """Test bucket-aware collate function."""

    def test_collate_stacks_tensors(self):
        """Test that collate function stacks tensors correctly."""
        batch = [
            {
                "pixel_values": torch.randn(3, 1024, 768),
                "caption": "test 1",
                "time_ids": [1024, 768, 0, 0, 1024, 768],
            },
            {
                "pixel_values": torch.randn(3, 1024, 768),
                "caption": "test 2",
                "time_ids": [1024, 768, 0, 0, 1024, 768],
            },
        ]

        collated = bucket_aware_collate(batch)

        assert "pixel_values" in collated
        assert "caption" in collated
        assert "time_ids" in collated

        assert collated["pixel_values"].shape == (2, 3, 1024, 768)
        assert isinstance(collated["caption"], list)
        assert len(collated["caption"]) == 2
        assert collated["time_ids"].shape == (2, 6)

    def test_collate_rejects_mixed_shapes(self):
        """Test that collate rejects batches with mixed shapes."""
        batch = [
            {
                "pixel_values": torch.randn(3, 1024, 768),
                "caption": "test 1",
                "time_ids": [1024, 768, 0, 0, 1024, 768],
            },
            {
                "pixel_values": torch.randn(3, 768, 1024),  # Different shape!
                "caption": "test 2",
                "time_ids": [768, 1024, 0, 0, 768, 1024],
            },
        ]

        with pytest.raises(ValueError, match="Mixed bucket sizes"):
            bucket_aware_collate(batch)

    def test_collate_empty_batch_raises_error(self):
        """Test that collate raises error for empty batch."""
        with pytest.raises(ValueError, match="Cannot collate empty batch"):
            bucket_aware_collate([])


class TestEndToEndBucketing:
    """End-to-end integration tests for bucketing."""

    def test_dataset_with_bucketing_loads_correctly(self, tmp_path):
        """Test that dataset with bucketing loads images correctly."""
        from lora_trainer.data import ImageFolderWithCaptions

        # Create test images with different aspect ratios
        img_square = Image.new("RGB", (1024, 1024), color="red")
        img_square.save(tmp_path / "square.png")

        img_landscape = Image.new("RGB", (1600, 900), color="green")
        img_landscape.save(tmp_path / "landscape.png")

        img_portrait = Image.new("RGB", (900, 1600), color="blue")
        img_portrait.save(tmp_path / "portrait.png")

        # Create dataset with bucketing
        bucket_config = BucketConfig(enabled=True)
        dataset = ImageFolderWithCaptions(tmp_path, bucket_config=bucket_config)

        assert len(dataset) == 3

        # Load each item
        for idx in range(len(dataset)):
            item = dataset[idx]
            assert "pixel_values" in item
            assert "caption" in item
            assert "time_ids" in item
            assert "bucket" in item

            # Verify time_ids format
            assert len(item["time_ids"]) == 6

    def test_bucketing_preserves_aspect_ratios(self, tmp_path):
        """Test that bucketing uses appropriate buckets for different images."""
        from lora_trainer.data import ImageFolderWithCaptions

        # Create images (files are sorted alphabetically)
        img_square = Image.new("RGB", (1000, 1000))
        img_square.save(tmp_path / "1_square.png")

        img_tall = Image.new("RGB", (1000, 2000))
        img_tall.save(tmp_path / "2_tall.png")

        img_wide = Image.new("RGB", (2000, 1000))
        img_wide.save(tmp_path / "3_wide.png")

        bucket_config = BucketConfig(enabled=True)
        dataset = ImageFolderWithCaptions(tmp_path, bucket_config=bucket_config)

        # Check bucket assignments (sorted order: 1_square, 2_tall, 3_wide)
        buckets = [dataset[i]["bucket"] for i in range(len(dataset))]

        # Index 0: Square should get square-ish bucket
        assert abs(buckets[0].aspect_ratio - 1.0) < 0.3

        # Index 1: Tall should get portrait bucket
        assert buckets[1].height > buckets[1].width

        # Index 2: Wide should get landscape bucket
        assert buckets[2].width > buckets[2].height
