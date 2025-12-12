"""Tests for model loading and LoRA injection."""

import torch
import torch.nn as nn

from lora_trainer.model import LoRALayer, inject_lora_into_unet, select_lora_params


class DummyAttentionBlock(nn.Module):
    """Minimal attention block for testing LoRA injection."""

    def __init__(self, dim=64):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim))

    def forward(self, x):
        # Simple forward - just use one path for testing
        v = self.to_v(x)
        return self.to_out[0](v)


class DummyUNet(nn.Module):
    """Minimal UNet-like model for testing."""

    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([DummyAttentionBlock() for _ in range(2)])
        self.up_blocks = nn.ModuleList([DummyAttentionBlock() for _ in range(2)])

    def forward(self, x, timesteps, encoder_hidden_states):
        for block in self.down_blocks:
            x = block(x)
        for block in self.up_blocks:
            x = block(x)
        return type("Output", (), {"sample": x})()


def test_lora_layer_initialization():
    """Test that LoRALayer initializes correctly."""
    original = nn.Linear(64, 128)
    lora = LoRALayer(original, rank=8, alpha=8.0)

    assert lora.rank == 8
    assert lora.alpha == 8.0
    assert lora.lora_down.in_features == 64
    assert lora.lora_down.out_features == 8
    assert lora.lora_up.in_features == 8
    assert lora.lora_up.out_features == 128


def test_lora_layer_freezes_original():
    """Test that LoRA freezes the original layer parameters."""
    original = nn.Linear(64, 128)
    lora = LoRALayer(original, rank=8, alpha=8.0)

    # Original layer should be frozen
    assert not lora.original_layer.weight.requires_grad
    if lora.original_layer.bias is not None:
        assert not lora.original_layer.bias.requires_grad

    # LoRA layers should be trainable
    assert lora.lora_down.weight.requires_grad
    assert lora.lora_up.weight.requires_grad


def test_lora_layer_forward():
    """Test that LoRALayer forward pass works correctly."""
    original = nn.Linear(64, 128)
    lora = LoRALayer(original, rank=8, alpha=8.0)

    x = torch.randn(2, 64)
    output = lora(x)

    assert output.shape == (2, 128)


def test_lora_layer_initialization_weights():
    """Test that LoRA weights are initialized correctly."""
    original = nn.Linear(64, 128)
    lora = LoRALayer(original, rank=8, alpha=8.0)

    # lora_up should be initialized to zeros
    assert torch.allclose(lora.lora_up.weight, torch.zeros_like(lora.lora_up.weight))

    # lora_down should NOT be all zeros (kaiming init)
    assert not torch.allclose(lora.lora_down.weight, torch.zeros_like(lora.lora_down.weight))


def test_inject_lora_into_dummy_unet():
    """Test that LoRA injection works on a dummy UNet."""
    model = DummyUNet()

    # Inject LoRA
    model = inject_lora_into_unet(model, rank=4, alpha=4.0)

    # Count LoRA layers after injection
    lora_count = sum(1 for _ in model.modules() if isinstance(_, LoRALayer))

    # Should have injected LoRA into to_q, to_k, to_v, to_out.0 in each attention block
    # 4 layers per block * 4 blocks = 16 LoRA layers
    assert lora_count == 16


def test_inject_lora_preserves_functionality():
    """Test that model still works after LoRA injection."""
    model = DummyUNet()
    model = inject_lora_into_unet(model, rank=4, alpha=4.0)

    x = torch.randn(2, 64)
    timesteps = torch.zeros(2).long()
    encoder_hidden_states = torch.randn(2, 77, 64)

    output = model(x, timesteps, encoder_hidden_states)
    assert output.sample.shape == (2, 64)


def test_select_lora_params_returns_only_lora():
    """Test that select_lora_params returns only LoRA parameters."""
    model = DummyUNet()
    model = inject_lora_into_unet(model, rank=4, alpha=4.0)

    lora_params = list(select_lora_params(model))

    # Should have LoRA parameters
    assert len(lora_params) > 0

    # All params should be from lora_down or lora_up
    for param in lora_params:
        assert param.requires_grad

    # Count expected: 16 LoRA layers * 2 (down + up) = 32 parameter tensors
    assert len(lora_params) == 32


def test_select_lora_params_excludes_original():
    """Test that select_lora_params excludes original layer parameters."""
    model = DummyUNet()

    # Count all parameters before LoRA
    all_params_before = list(model.parameters())

    model = inject_lora_into_unet(model, rank=4, alpha=4.0)

    lora_params = list(select_lora_params(model))

    # LoRA params should be a subset of all params
    assert len(lora_params) < len(list(model.parameters()))

    # Original layer params should not be in LoRA params
    lora_param_ids = {id(p) for p in lora_params}
    original_param_ids = {id(p) for p in all_params_before}

    # No overlap between original and LoRA params
    assert len(lora_param_ids & original_param_ids) == 0


def test_lora_injection_with_custom_targets():
    """Test LoRA injection with custom target modules."""
    model = DummyUNet()

    # Only inject into to_q layers
    model = inject_lora_into_unet(model, rank=4, alpha=4.0, target_modules=["to_q"])

    lora_count = sum(1 for _ in model.modules() if isinstance(_, LoRALayer))

    # Should only have 4 LoRA layers (one to_q per attention block)
    assert lora_count == 4


def test_lora_params_are_trainable():
    """Test that LoRA parameters are trainable while original params are frozen."""
    model = DummyUNet()
    model = inject_lora_into_unet(model, rank=4, alpha=4.0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]

    # Should have both trainable (LoRA) and frozen (original) params
    assert len(trainable_params) > 0
    assert len(frozen_params) > 0

    # All LoRA params should be trainable
    lora_params = list(select_lora_params(model))
    trainable_param_ids = {id(p) for p in trainable_params}
    for param in lora_params:
        assert id(param) in trainable_param_ids
