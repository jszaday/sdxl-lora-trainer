"""Tests for prompt weighting during training."""

from unittest.mock import MagicMock

import torch

from lora_trainer.train_loop import encode_prompts


class MockTokenizer:
    def __init__(self, max_length=77):
        self.model_max_length = max_length

    def __call__(self, text, **kwargs):
        # Return dummy input_ids
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
        return MagicMock(
            input_ids=torch.zeros((batch_size, self.model_max_length), dtype=torch.long)
        )


class MockEncoderOutput:
    def __init__(self, batch_size, hidden_dim=768, seq_len=77):
        self.hidden_states = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(13)]
        self.pooler_output = torch.randn(batch_size, hidden_dim)

    def __getitem__(self, idx):
        if idx == 0:
            return self.pooler_output
        return None


class MockTextEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, output_hidden_states=False):
        batch_size = input_ids.shape[0]
        return MockEncoderOutput(batch_size, self.hidden_dim)


def test_encode_prompts_with_weighting_enabled():
    """Verify that encode_prompts handles weighting when enabled."""
    tokenizer_1 = MockTokenizer(77)
    tokenizer_2 = MockTokenizer(77)
    text_encoder_1 = MockTextEncoder(768)
    text_encoder_2 = MockTextEncoder(1280)
    device = "cpu"

    captions = ["a photo of a (cat:1.5)"]

    # 1. Without weighting
    embeds_no_weight, pooled_no_weight = encode_prompts(
        captions,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        device,
        enable_weighting=False,
    )

    # 2. With weighting
    embeds_weight, pooled_weight = encode_prompts(
        captions,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        device,
        enable_weighting=True,
    )

    # Since we use random mock encoders, we can't easily check for specific values,
    # but we can check that they are different (or at least that the weighted path was taken).
    # Wait, in our MockEncoder, forward returns different random values each time.
    # To be deterministic, we should fix the seed or mock more carefully.

    assert embeds_weight.shape == (1, 77, 768 + 1280)
    assert pooled_weight.shape == (1, 1280)


def test_encode_prompts_weighting_consistency():
    """Verify that weighting changes the embeddings."""
    hidden_dim_1 = 768
    hidden_dim_2 = 1280
    device = "cpu"

    class BetterMockTokenizer:
        def __init__(self, max_length=77):
            self.model_max_length = max_length

        def __call__(self, text, **kwargs):
            if isinstance(text, list):
                batch_size = len(text)
                input_ids = torch.zeros((batch_size, self.model_max_length), dtype=torch.long)
                for i, t in enumerate(text):
                    if t:
                        input_ids[i, :] = 1  # Non-empty
                res = MagicMock()
                res.input_ids = input_ids
                return res
            # Single string case
            input_ids = (
                torch.ones((1, self.model_max_length), dtype=torch.long)
                if text
                else torch.zeros((1, self.model_max_length), dtype=torch.long)
            )
            res = MagicMock()
            res.input_ids = input_ids
            return res

    class VariableTextEncoder(torch.nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, input_ids, output_hidden_states=False):
            batch_size = input_ids.shape[0]
            # If all zeros (empty prompt), return 0.0, else 1.0
            is_empty = (input_ids == 0).all()
            val = 0.0 if is_empty else 1.0
            out = MagicMock()
            out.hidden_states = [
                torch.full((batch_size, 77, self.hidden_dim), val) for _ in range(13)
            ]
            out.__getitem__.return_value = torch.full((batch_size, self.hidden_dim), val)
            return out

    tokenizer_1 = BetterMockTokenizer(77)
    tokenizer_2 = BetterMockTokenizer(77)
    te1 = VariableTextEncoder(hidden_dim_1)
    te2 = VariableTextEncoder(hidden_dim_2)

    captions = ["(cat:2.0)"]

    # Without weighting
    embeds_no, _ = encode_prompts(
        captions, te1, te2, tokenizer_1, tokenizer_2, device, enable_weighting=False
    )
    assert torch.allclose(embeds_no, torch.tensor(1.0))

    # With weighting
    embeds_yes, _ = encode_prompts(
        captions, te1, te2, tokenizer_1, tokenizer_2, device, enable_weighting=True
    )

    # Check that it's greater than 1.0
    assert torch.any(embeds_yes > 1.0)
    # Specifically, it should be 2.0 for some tokens
    assert torch.allclose(embeds_yes.max(), torch.tensor(2.0))
