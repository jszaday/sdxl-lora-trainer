import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from lora_converter.converter import convert_checkpoint, convert_lora_state


def test_convert_lora_state_to_comfy_keys():
    lora_state = {
        "down_blocks.0.attn.to_q.lora_down.weight": torch.randn(4, 3),
        "down_blocks.0.attn.to_q.lora_up.weight": torch.randn(3, 4),
    }

    converted = convert_lora_state(lora_state)
    assert set(converted) == {
        "lora_unet_down_blocks_0_attn_to_q.lora_down.weight",
        "lora_unet_down_blocks_0_attn_to_q.lora_up.weight",
    }


def test_convert_checkpoint_writes_safetensors(tmp_path):
    input_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "converted.safetensors"

    state = {
        "model_state_dict": {
            "up_blocks.1.attn.to_k.lora_down.weight": torch.randn(2, 2),
            "up_blocks.1.attn.to_k.lora_up.weight": torch.randn(2, 2),
            "some_other.weight": torch.randn(1, 1),
        }
    }
    torch.save(state, input_path)

    convert_checkpoint(input_path, output_path)
    tensors = load_file(output_path)

    assert set(tensors) == {
        "lora_unet_up_blocks_1_attn_to_k.lora_down.weight",
        "lora_unet_up_blocks_1_attn_to_k.lora_up.weight",
    }
    assert tensors["lora_unet_up_blocks_1_attn_to_k.lora_down.weight"].shape == (2, 2)
    with safe_open(output_path, framework="pt", device="cpu") as f:
        meta = f.metadata()
    assert meta.get("format") == "pt"
    assert meta.get("network_dim") == "2"


def test_convert_safetensors_input(tmp_path):
    input_path = tmp_path / "input.safetensors"
    output_path = tmp_path / "converted_out.safetensors"

    lora_state = {
        "mid_block.attn.lora_down.weight": torch.randn(4, 4),
        "mid_block.attn.lora_up.weight": torch.randn(4, 4),
    }
    save_file(lora_state, str(input_path))

    convert_checkpoint(input_path, output_path)
    tensors = load_file(output_path)

    assert set(tensors) == {
        "lora_unet_mid_block_attn.lora_down.weight",
        "lora_unet_mid_block_attn.lora_up.weight",
    }
    with safe_open(output_path, framework="pt", device="cpu") as f:
        meta = f.metadata()
    assert meta.get("format") == "pt"
