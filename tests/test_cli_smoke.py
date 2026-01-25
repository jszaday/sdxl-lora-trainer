"""Smoke tests for CLI end-to-end functionality."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def temp_training_setup():
    """Create a complete temporary training setup."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create train data directory with images
    train_data = temp_dir / "train_data"
    train_data.mkdir()

    # Create 5 small test images
    for i in range(5):
        img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
        img.save(train_data / f"image_{i}.jpg")

    # Create workspace directory
    workspace = temp_dir / "workspace"

    yield {
        "train_data": train_data,
        "workspace": workspace,
        "root": temp_dir,
    }

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.skip(reason="Requires real SDXL model - Phase 2 loads from HuggingFace")
def test_cli_runs_successfully(temp_training_setup):
    """Test that CLI runs end-to-end without crashing."""
    # Run the CLI as a subprocess
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lora_trainer.cli",
            "--checkpoint",
            "dummy_checkpoint.safetensors",
            "--train_data",
            str(temp_training_setup["train_data"]),
            "--steps",
            "3",
            "--batch_size",
            "2",
            "--workspace",
            str(temp_training_setup["workspace"]),
            "--num_workers",
            "0",  # Use 0 workers for testing
        ],
        capture_output=True,
        text=True,
        timeout=60,  # 60 second timeout
    )

    # Check that it completed successfully
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"

    # Check that workspace was created
    assert temp_training_setup["workspace"].exists()

    # Check that subdirectories were created
    assert (temp_training_setup["workspace"] / "checkpoints").exists()
    assert (temp_training_setup["workspace"] / "tb").exists()
    assert (temp_training_setup["workspace"] / "samples").exists()

    # Check that final checkpoint was saved
    final_checkpoint = temp_training_setup["workspace"] / "checkpoints" / "final.pt"
    assert final_checkpoint.exists()


def test_cli_with_invalid_args_fails():
    """Test that CLI fails gracefully with invalid arguments."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lora_trainer.cli",
            "--checkpoint",
            "dummy.safetensors",
            "--train_data",
            "/nonexistent/path",
            "--steps",
            "10",
            "--batch_size",
            "4",
            "--workspace",
            "/tmp/test",
        ],
        capture_output=True,
        text=True,
    )

    # Should fail with non-zero exit code
    assert result.returncode != 0


def test_cli_missing_required_args_fails():
    """Test that CLI fails when required arguments are missing."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lora_trainer.cli",
            "--checkpoint",
            "dummy.safetensors",
            # Missing other required args
        ],
        capture_output=True,
        text=True,
    )

    # Should fail with non-zero exit code
    assert result.returncode != 0


def test_cli_help_message():
    """Test that --help returns usage information."""
    result = subprocess.run(
        [sys.executable, "-m", "lora_trainer.cli", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "SDXL LoRA Trainer" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--train_data" in result.stdout
    assert "--steps" in result.stdout
