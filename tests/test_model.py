"""Tests for model loading and LoRA injection."""

import torch
import torch.nn as nn

from lora_trainer.model import (
    LoRALayer,
    inject_lora_into_unet,
    load_lora_weights,
    select_lora_params,
)


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


def test_load_lora_weights_applies_to_unet_and_text_encoders(tmp_path):
    """Ensure Comfy-style keys load into UNet and both text encoders."""

    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = LoRALayer(nn.Linear(2, 2))

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DummyAttn()

    class DummyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = DummyBlock()

    class DummySelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LoRALayer(nn.Linear(2, 2))

    class DummyTE(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummySelfAttn()

    unet = DummyUNet()
    te1 = DummyTE()
    te2 = DummyTE()

    state = {
        "lora_unet_block_attn_to_q.lora_down.weight": torch.ones_like(
            unet.block.attn.to_q.lora_down.weight
        ),
        "lora_unet_block_attn_to_q.lora_up.weight": torch.full_like(
            unet.block.attn.to_q.lora_up.weight, 2.0
        ),
        "lora_te1_self_attn_q_proj.lora_down.weight": torch.full_like(
            te1.self_attn.q_proj.lora_down.weight, 3.0
        ),
        "lora_te1_self_attn_q_proj.lora_up.weight": torch.full_like(
            te1.self_attn.q_proj.lora_up.weight, 4.0
        ),
        "lora_te2_self_attn_q_proj.lora_down.weight": torch.full_like(
            te2.self_attn.q_proj.lora_down.weight, 5.0
        ),
        "lora_te2_self_attn_q_proj.lora_up.weight": torch.full_like(
            te2.self_attn.q_proj.lora_up.weight, 6.0
        ),
    }

    lora_path = tmp_path / "dummy_lora.pt"
    torch.save(state, lora_path)

    load_lora_weights(lora_path, unet=unet, text_encoder_1=te1, text_encoder_2=te2)

    assert torch.allclose(
        unet.block.attn.to_q.lora_down.weight,
        state["lora_unet_block_attn_to_q.lora_down.weight"],
    )
    assert torch.allclose(
        unet.block.attn.to_q.lora_up.weight,
        state["lora_unet_block_attn_to_q.lora_up.weight"],
    )
    assert torch.allclose(
        te1.self_attn.q_proj.lora_down.weight,
        state["lora_te1_self_attn_q_proj.lora_down.weight"],
    )
    assert torch.allclose(
        te1.self_attn.q_proj.lora_up.weight,
        state["lora_te1_self_attn_q_proj.lora_up.weight"],
    )
    assert torch.allclose(
        te2.self_attn.q_proj.lora_down.weight,
        state["lora_te2_self_attn_q_proj.lora_down.weight"],
    )
    assert torch.allclose(
        te2.self_attn.q_proj.lora_up.weight,
        state["lora_te2_self_attn_q_proj.lora_up.weight"],
    )


def test_lora_round_trip_export_and_import(tmp_path):
    """Test complete round-trip: extract -> convert -> save -> load -> verify."""
    from safetensors.torch import save_file

    from lora_converter.converter import convert_lora_state
    from lora_trainer.model import LoRAConv2d, extract_lora_state_dict

    # Create models with LoRA (alpha=32.0)
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = LoRALayer(nn.Linear(64, 64), rank=8, alpha=32.0)
            self.to_k = LoRALayer(nn.Linear(64, 64), rank=8, alpha=32.0)

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DummyAttn()
            self.conv1 = LoRAConv2d(nn.Conv2d(3, 16, 3, padding=1), rank=8, alpha=32.0)

    class DummyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = DummyBlock()

    class DummySelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LoRALayer(nn.Linear(64, 64), rank=8, alpha=32.0)

    class DummyTE(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummySelfAttn()

    # Step 1: Create original models with alpha=32.0
    unet1 = DummyUNet()
    te1_1 = DummyTE()

    # Step 2: Extract LoRA state (should include alphas)
    unet_lora_state = extract_lora_state_dict(unet1)
    te1_lora_state = extract_lora_state_dict(te1_1)

    # Verify extracted state has alphas
    assert "block.attn.to_q.alpha" in unet_lora_state
    assert "block.attn.to_k.alpha" in unet_lora_state
    assert "block.conv1.alpha" in unet_lora_state
    assert "self_attn.q_proj.alpha" in te1_lora_state

    # Verify alpha values
    assert unet_lora_state["block.attn.to_q.alpha"].item() == 32.0
    assert te1_lora_state["self_attn.q_proj.alpha"].item() == 32.0

    # Step 3: Convert to ComfyUI format
    # Note: We need to manually handle text encoder state since the converter
    # expects "text_model" in the key to identify TE keys. For UNet, we just
    # pass through as-is.
    converted = convert_lora_state(unet_lora_state)

    # For text encoder, manually convert with te1 prefix
    for key, tensor in te1_lora_state.items():
        # Convert self_attn.q_proj.lora_down.weight -> lora_te1_self_attn_q_proj.lora_down.weight
        if ".lora_down.weight" in key:
            base = key[: -len(".lora_down.weight")]
            suffix = "lora_down.weight"
        elif ".lora_up.weight" in key:
            base = key[: -len(".lora_up.weight")]
            suffix = "lora_up.weight"
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            suffix = "alpha"
        else:
            continue
        comfy_key = f"lora_te1_{base.replace('.', '_')}.{suffix}"
        converted[comfy_key] = tensor.detach().cpu()

    # Verify keys are in ComfyUI format
    assert "lora_unet_block_attn_to_q.lora_down.weight" in converted
    assert "lora_unet_block_attn_to_q.lora_up.weight" in converted
    assert "lora_unet_block_attn_to_q.alpha" in converted
    assert "lora_unet_block_conv1.lora_down.weight" in converted
    assert "lora_te1_self_attn_q_proj.alpha" in converted

    # Step 4: Save to safetensors
    lora_path = tmp_path / "test_lora.safetensors"
    metadata = {"network_dim": "8", "network_alpha": "32.0"}
    save_file(converted, str(lora_path), metadata=metadata)

    # Step 5: Create fresh models with different alpha (16.0)
    unet2 = DummyUNet()
    te1_2 = DummyTE()

    # Manually set alpha to 16.0 to verify it gets overwritten
    unet2.block.attn.to_q.alpha = 16.0
    unet2.block.attn.to_k.alpha = 16.0
    unet2.block.conv1.alpha = 16.0
    te1_2.self_attn.q_proj.alpha = 16.0

    # Step 6: Load LoRA weights (should update alphas to 32.0)
    load_lora_weights(lora_path, unet=unet2, text_encoder_1=te1_2)

    # Step 7: Verify weights match
    assert torch.allclose(
        unet2.block.attn.to_q.lora_down.weight,
        unet1.block.attn.to_q.lora_down.weight,
    )
    assert torch.allclose(
        unet2.block.attn.to_q.lora_up.weight, unet1.block.attn.to_q.lora_up.weight
    )
    assert torch.allclose(unet2.block.conv1.lora_down.weight, unet1.block.conv1.lora_down.weight)
    assert torch.allclose(
        te1_2.self_attn.q_proj.lora_down.weight,
        te1_1.self_attn.q_proj.lora_down.weight,
    )

    # Step 8: Verify alphas were updated to 32.0
    assert unet2.block.attn.to_q.alpha == 32.0
    assert unet2.block.attn.to_k.alpha == 32.0
    assert unet2.block.conv1.alpha == 32.0
    assert te1_2.self_attn.q_proj.alpha == 32.0

    # Step 9: Verify all LoRA modules have correct alpha
    for name, module in unet2.named_modules():
        if isinstance(module, (LoRALayer, LoRAConv2d)):
            assert module.alpha == 32.0, f"Module {name} has alpha={module.alpha}, expected 32.0"

    for name, module in te1_2.named_modules():
        if isinstance(module, LoRALayer):
            assert module.alpha == 32.0, f"Module {name} has alpha={module.alpha}, expected 32.0"
