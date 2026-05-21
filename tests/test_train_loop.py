"""Tests for training loop functionality."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from lora_trainer.config import TrainingConfig
from lora_trainer.logging import create_run_dirs
from lora_trainer.model import LoRALayer
from lora_trainer.train_loop import save_checkpoint, train


class DummyUNet(nn.Module):
    """Minimal UNet for testing that returns diffusers-compatible output."""

    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # Return a simple output with .sample attribute like diffusers
        output = self.conv(sample)
        return type("Output", (), {"sample": output})()


def select_lora_params_dummy(model):
    """Select params for dummy model (all params since no LoRA yet)."""
    # For dummy model without LoRA, return all params
    for module in model.modules():
        if isinstance(module, LoRALayer):
            yield from module.lora_down.parameters()
            yield from module.lora_up.parameters()
    # If no LoRA layers, return all parameters
    if not any(isinstance(m, LoRALayer) for m in model.modules()):
        yield from model.parameters()


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with dummy image files."""
    temp_dir = Path(tempfile.mkdtemp())
    for i in range(10):
        (temp_dir / f"image_{i}.jpg").touch()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_dataloader():
    """Create a simple in-memory dataloader for testing."""
    # Create fake cached data: [batch_size, channels, height, width]
    latents = torch.randn(20, 4, 64, 64)
    prompt_embeds = torch.randn(20, 77, 2048)
    pooled_embeds = torch.randn(20, 1280)
    time_ids = torch.zeros(20, 6)

    dataset = TensorDataset(latents, prompt_embeds, pooled_embeds, time_ids)

    def collate_fn(batch):
        stacked = {
            "latents": torch.stack([item[0] for item in batch]),
            "prompt_embeds": torch.stack([item[1] for item in batch]),
            "pooled_embeds": torch.stack([item[2] for item in batch]),
            "time_ids": torch.stack([item[3] for item in batch]),
        }
        return stacked

    return DataLoader(dataset, batch_size=2, collate_fn=collate_fn, drop_last=True)


def test_train_completes_requested_steps(temp_workspace, temp_data_dir):
    """Test that training runs for the requested number of steps."""
    # Create minimal config
    config = TrainingConfig(
        checkpoint="dummy.safetensors",
        train_data=temp_data_dir,
        steps=5,
        batch_size=2,
        workspace=temp_workspace,
        grad_accum=1,
        image_size=512,  # Match our 64x64 latent data (64 * 8 = 512)
    )

    # Setup
    dirs = create_run_dirs(config.workspace)
    writer = SummaryWriter(log_dir=str(dirs["tb"]))
    model = DummyUNet()
    optimizer = torch.optim.AdamW(select_lora_params_dummy(model), lr=1e-4)

    # Create simple cached dataloader
    latents = torch.randn(20, 4, 64, 64)
    prompt_embeds = torch.randn(20, 77, 2048)
    pooled_embeds = torch.randn(20, 1280)
    time_ids = torch.zeros(20, 6)
    dataset = TensorDataset(latents, prompt_embeds, pooled_embeds, time_ids)

    def collate_fn(batch):
        return {
            "latents": torch.stack([item[0] for item in batch]),
            "prompt_embeds": torch.stack([item[1] for item in batch]),
            "pooled_embeds": torch.stack([item[2] for item in batch]),
            "time_ids": torch.stack([item[3] for item in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, drop_last=True)

    # Run training
    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        config=config,
        dirs=dirs,
        writer=writer,
        device="cpu",
        cached_data=True,
    )


class DummyWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def close(self):
        pass


def test_train_logs_perf_metrics(temp_workspace, temp_data_dir):
    """Ensure perf metrics are logged during training."""
    config = TrainingConfig(
        checkpoint="dummy.safetensors",
        train_data=temp_data_dir,
        steps=2,
        batch_size=2,
        workspace=temp_workspace,
        grad_accum=1,
        image_size=512,
        sample_every=10,  # avoid periodic checkpoints
        log_every=1,  # log every step for testing
    )

    dirs = create_run_dirs(config.workspace)
    writer = DummyWriter()
    model = DummyUNet(in_channels=4, out_channels=4)
    optimizer = torch.optim.AdamW(select_lora_params_dummy(model), lr=1e-4)

    # Simple dataloader matching dummy latent shape
    latents = torch.randn(8, 4, 64, 64)
    prompt_embeds = torch.randn(8, 77, 2048)
    pooled_embeds = torch.randn(8, 1280)
    time_ids = torch.zeros(8, 6)
    dataset = TensorDataset(latents, prompt_embeds, pooled_embeds, time_ids)

    def collate_fn(batch):
        return {
            "latents": torch.stack([item[0] for item in batch]),
            "prompt_embeds": torch.stack([item[1] for item in batch]),
            "pooled_embeds": torch.stack([item[2] for item in batch]),
            "time_ids": torch.stack([item[3] for item in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, drop_last=True)

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        config=config,
        dirs=dirs,
        writer=writer,
        device="cpu",
        cached_data=True,
    )

    tags = {tag for tag, _, _ in writer.scalars}
    assert "perf/step_time_sec" in tags
    assert "perf/images_per_sec" in tags
    assert "perf/checkpoint_time_sec" in tags
    writer.close()

    # Check that final checkpoint exists
    final_checkpoint = dirs["checkpoints"] / "final.pt"
    assert final_checkpoint.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP smoke test")
def test_train_amp_fp16_smoke(temp_workspace, temp_data_dir):
    """Smoke test AMP/GradScaler path on CUDA."""
    config = TrainingConfig(
        checkpoint="dummy.safetensors",
        train_data=temp_data_dir,
        steps=1,
        batch_size=2,
        workspace=temp_workspace,
        grad_accum=1,
        image_size=512,
        sample_every=1000,
        log_every=1,
        mixed_precision="fp16",
    )

    dirs = create_run_dirs(config.workspace)
    writer = DummyWriter()
    model = DummyUNet(in_channels=4, out_channels=4).to("cuda")
    optimizer = torch.optim.AdamW(select_lora_params_dummy(model), lr=1e-4)

    latents = torch.randn(8, 4, 64, 64)
    prompt_embeds = torch.randn(8, 77, 2048)
    pooled_embeds = torch.randn(8, 1280)
    time_ids = torch.zeros(8, 6)
    dataset = TensorDataset(latents, prompt_embeds, pooled_embeds, time_ids)

    def collate_fn(batch):
        return {
            "latents": torch.stack([item[0] for item in batch]),
            "prompt_embeds": torch.stack([item[1] for item in batch]),
            "pooled_embeds": torch.stack([item[2] for item in batch]),
            "time_ids": torch.stack([item[3] for item in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, drop_last=True)

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        config=config,
        dirs=dirs,
        writer=writer,
        device="cuda",
        cached_data=True,
    )

    final_checkpoint = dirs["checkpoints"] / "final.pt"
    assert final_checkpoint.exists()

    writer.close()


def test_save_checkpoint_creates_file(temp_workspace):
    """Test that save_checkpoint creates a checkpoint file."""
    model = DummyUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    checkpoint_dir = temp_workspace / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=100,
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint_path = checkpoint_dir / "step_000100.pt"
    assert checkpoint_path.exists()


def test_save_checkpoint_final(temp_workspace):
    """Test that final checkpoint has correct name."""
    model = DummyUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    checkpoint_dir = temp_workspace / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=1000,
        checkpoint_dir=checkpoint_dir,
        is_final=True,
    )

    final_path = checkpoint_dir / "final.pt"
    assert final_path.exists()


def test_checkpoint_contains_required_keys(temp_workspace):
    """Test that checkpoint contains model and optimizer state."""
    model = DummyUNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    checkpoint_dir = temp_workspace / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=50,
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint_path = checkpoint_dir / "step_000050.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "global_step" in checkpoint
    assert checkpoint["global_step"] == 50


def test_model_parameters_update_during_training(temp_workspace, temp_data_dir):
    """Test that model parameters actually change during training."""
    config = TrainingConfig(
        checkpoint="dummy.safetensors",
        train_data=temp_data_dir,
        steps=3,
        batch_size=2,
        workspace=temp_workspace,
        image_size=512,  # Match our 64x64 latent data (64 * 8 = 512)
    )

    dirs = create_run_dirs(config.workspace)
    writer = SummaryWriter(log_dir=str(dirs["tb"]))
    model = DummyUNet()

    # Save initial parameters
    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    optimizer = torch.optim.AdamW(select_lora_params_dummy(model), lr=1e-3)

    # Create dataloader (use 4 channels like latent space)
    latents = torch.randn(20, 4, 64, 64)
    prompt_embeds = torch.randn(20, 77, 2048)
    pooled_embeds = torch.randn(20, 1280)
    time_ids = torch.zeros(20, 6)
    dataset = TensorDataset(latents, prompt_embeds, pooled_embeds, time_ids)

    def collate_fn(batch):
        return {
            "latents": torch.stack([item[0] for item in batch]),
            "prompt_embeds": torch.stack([item[1] for item in batch]),
            "pooled_embeds": torch.stack([item[2] for item in batch]),
            "time_ids": torch.stack([item[3] for item in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, drop_last=True)

    # Run training
    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        config=config,
        dirs=dirs,
        writer=writer,
        device="cpu",
        cached_data=True,
    )

    # Check that at least some parameters changed
    params_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            params_changed = True
            break

    assert params_changed, "Model parameters should change during training"

    writer.close()
