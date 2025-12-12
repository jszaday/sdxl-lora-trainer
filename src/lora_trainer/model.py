"""Model loading and LoRA parameter selection.

Phase 1: Contains a DummyUNet stub for testing the training infrastructure.
Phase 2: Will be replaced with real SDXL UNet + LoRA injection.
"""

from collections.abc import Iterator

import torch
import torch.nn as nn


class DummyUNet(nn.Module):
    """Placeholder UNet for Phase 1 testing.

    This is a minimal model that accepts image tensors and produces
    a simple loss for testing the training loop infrastructure.
    Will be replaced with real SDXL UNet in Phase 2.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        # Minimal architecture just to have trainable parameters
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a simple output for loss computation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, 1, 1] - simplified for dummy training
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.pool(x)  # Global pooling to fixed size for simple loss
        return x


def load_model(checkpoint: str, device: str = "cpu") -> nn.Module:
    """Load model from checkpoint.

    Phase 1: Returns DummyUNet (checkpoint path is ignored for now).
    Phase 2: Will load real SDXL UNet from checkpoint/HF hub.

    Args:
        checkpoint: Path to checkpoint or HuggingFace model ID
        device: Device to load model on

    Returns:
        Model instance
    """
    # Phase 1: Just return a dummy model
    model = DummyUNet()
    model = model.to(device)
    return model


def select_lora_params(model: nn.Module) -> Iterator[nn.Parameter]:
    """Select LoRA parameters for training.

    Phase 1: Returns all parameters (no LoRA yet).
    Phase 2: Will return only LoRA parameters after injection.

    Args:
        model: Model instance

    Returns:
        Iterator of parameters to train
    """
    # Phase 1: Train all parameters
    return model.parameters()
