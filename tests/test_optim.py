"""Tests for optimizer spec parsing."""

import pytest

from lora_trainer.optim import parse_optimizer_spec


def test_parse_optimizer_spec_supports_tuples_and_lists():
    """Tuple/list literal args should parse via AST."""
    name, kwargs = parse_optimizer_spec(
        "prodigy(decouple=True,use_bias_correction=True,"
        "weight_decay=0.01,safeguard_warmup=True,betas=(0.9,0.99),"
        "layers=[1,2,3])"
    )

    assert name == "prodigy"
    assert kwargs["decouple"] is True
    assert kwargs["use_bias_correction"] is True
    assert kwargs["weight_decay"] == 0.01
    assert kwargs["safeguard_warmup"] is True
    assert kwargs["betas"] == (0.9, 0.99)
    assert kwargs["layers"] == [1, 2, 3]


def test_parse_optimizer_spec_rejects_positional_args():
    """Positional args should not be accepted."""
    with pytest.raises(ValueError, match="Positional arguments"):
        parse_optimizer_spec("adamw(1e-4)")
