"""Tests for configuration parsing and validation."""

import shutil
import tempfile
from pathlib import Path

import pytest

from lora_trainer.config import TrainingConfig


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with dummy image files."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create some dummy image files
    for i in range(10):
        (temp_dir / f"image_{i}.jpg").touch()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_prompts_file():
    """Create a temporary prompts file."""
    temp_file = Path(tempfile.mktemp(suffix=".txt"))
    temp_file.write_text("a photo of a cat\na photo of a dog\n")

    yield temp_file

    # Cleanup
    temp_file.unlink()


def test_valid_config_basic(temp_data_dir):
    """Test that a valid config computes expected values."""
    config = TrainingConfig(
        checkpoint="dummy_checkpoint.safetensors",
        train_data=temp_data_dir,
        steps=100,
        batch_size=4,
        workspace=Path("/tmp/test_workspace"),
    )

    assert config.steps == 100
    assert config.batch_size == 4
    assert config.effective_batch_size == 4  # batch_size * grad_accum (1)
    assert config.num_images == 10
    assert config.steps_per_epoch == 2  # 10 images / 4 batch_size = 2.5 -> 2
    assert config.num_epochs >= 1


def test_config_with_grad_accum(temp_data_dir):
    """Test effective batch size computation with gradient accumulation."""
    config = TrainingConfig(
        checkpoint="dummy_checkpoint.safetensors",
        train_data=temp_data_dir,
        steps=100,
        batch_size=2,
        workspace=Path("/tmp/test_workspace"),
        grad_accum=4,
    )

    assert config.effective_batch_size == 8  # 2 * 4


def test_config_computes_epochs(temp_data_dir):
    """Test that num_epochs is computed correctly."""
    config = TrainingConfig(
        checkpoint="dummy_checkpoint.safetensors",
        train_data=temp_data_dir,
        steps=1000,
        batch_size=2,
        workspace=Path("/tmp/test_workspace"),
    )

    # 10 images / 2 batch_size = 5 steps per epoch
    # 1000 steps / 5 = 200 epochs
    assert config.steps_per_epoch == 5
    assert config.num_epochs == 200


def test_invalid_negative_steps(temp_data_dir):
    """Test that negative steps raise ValueError."""
    with pytest.raises(ValueError, match="steps must be positive"):
        TrainingConfig(
            checkpoint="dummy.safetensors",
            train_data=temp_data_dir,
            steps=-1,
            batch_size=4,
            workspace=Path("/tmp/test"),
        )


def test_invalid_negative_batch_size(temp_data_dir):
    """Test that negative batch_size raises ValueError."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(
            checkpoint="dummy.safetensors",
            train_data=temp_data_dir,
            steps=100,
            batch_size=-1,
            workspace=Path("/tmp/test"),
        )


def test_invalid_train_data_not_exists():
    """Test that non-existent train_data raises ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        TrainingConfig(
            checkpoint="dummy.safetensors",
            train_data=Path("/nonexistent/path"),
            steps=100,
            batch_size=4,
            workspace=Path("/tmp/test"),
        )


def test_invalid_cfg_negative(temp_data_dir):
    """Test that negative CFG raises ValueError."""
    with pytest.raises(ValueError, match="cfg must be non-negative"):
        TrainingConfig(
            checkpoint="dummy.safetensors",
            train_data=temp_data_dir,
            steps=100,
            batch_size=4,
            workspace=Path("/tmp/test"),
            cfg=-1.0,
        )


def test_invalid_lora_rank(temp_data_dir):
    """Test that invalid LoRA rank raises ValueError."""
    with pytest.raises(ValueError, match="lora_rank must be positive"):
        TrainingConfig(
            checkpoint="dummy.safetensors",
            train_data=temp_data_dir,
            steps=100,
            batch_size=4,
            workspace=Path("/tmp/test"),
            lora_rank=0,
        )


def test_config_with_sample_prompts(temp_data_dir, temp_prompts_file):
    """Test config with sample prompts file."""
    config = TrainingConfig(
        checkpoint="dummy.safetensors",
        train_data=temp_data_dir,
        steps=100,
        batch_size=4,
        workspace=Path("/tmp/test"),
        sample_prompts=temp_prompts_file,
    )

    assert config.sample_prompts == temp_prompts_file
    assert config.sample_prompts.exists()


def test_config_print_summary(temp_data_dir):
    """Test that print_summary returns a string."""
    config = TrainingConfig(
        checkpoint="dummy.safetensors",
        train_data=temp_data_dir,
        steps=100,
        batch_size=4,
        workspace=Path("/tmp/test"),
    )

    summary = config.print_summary()
    assert isinstance(summary, str)
    assert "Training Configuration" in summary
    assert "100" in summary  # steps
    assert "4" in summary  # batch_size
