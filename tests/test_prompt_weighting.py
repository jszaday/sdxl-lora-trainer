"""Tests for prompt weighting functionality."""

import pytest
import torch

from lora_trainer.prompt_weighting import (
    WeightedSegment,
    apply_prompt_weights,
    get_token_positions,
    parse_parentheses,
    parse_weighted_prompt,
    token_weights,
)


class TestParseParentheses:
    """Tests for parentheses parsing."""

    def test_no_parentheses(self):
        result = parse_parentheses("a simple prompt")
        assert result == ["a simple prompt"]

    def test_single_group(self):
        result = parse_parentheses("a (cat) image")
        assert result == ["a ", "(cat)", " image"]

    def test_multiple_groups(self):
        result = parse_parentheses("a (big) (fluffy) cat")
        assert result == ["a ", "(big)", " ", "(fluffy)", " cat"]

    def test_nested_groups(self):
        result = parse_parentheses("a ((very)) cute cat")
        assert result == ["a ", "((very))", " cute cat"]

    def test_only_parentheses(self):
        result = parse_parentheses("(cat)")
        assert result == ["(cat)"]


class TestTokenWeights:
    """Tests for token weight parsing."""

    def test_no_weights(self):
        result = token_weights("a simple prompt", 1.0)
        assert result == [("a simple prompt", 1.0)]

    def test_single_parenthesis(self):
        result = token_weights("a (cat)", 1.0)
        assert len(result) == 2
        assert result[0] == ("a ", 1.0)
        assert result[1] == ("cat", 1.1)

    def test_nested_parentheses(self):
        result = token_weights("a ((very)) cute cat", 1.0)
        assert len(result) == 3
        assert result[0] == ("a ", 1.0)
        assert result[1][0] == "very"
        assert abs(result[1][1] - 1.21) < 0.001  # 1.1 * 1.1
        assert result[2] == (" cute cat", 1.0)

    def test_explicit_weight(self):
        result = token_weights("a (cat:1.5)", 1.0)
        assert len(result) == 2
        assert result[0] == ("a ", 1.0)
        assert result[1] == ("cat", 1.5)

    def test_explicit_weight_in_nested(self):
        result = token_weights("((cat:2.0))", 1.0)
        assert len(result) == 1
        # Outer parens multiply by 1.1, inner has explicit weight 2.0
        # Since explicit weight overrides, result should be 2.0
        assert result[0] == ("cat", 2.0)

    def test_colon_without_weight(self):
        # Colon at the beginning should be treated as literal text
        result = token_weights("(:text)", 1.0)
        # Should try to parse empty string before colon as weight, fail, use default
        assert len(result) == 1


class TestParseWeightedPrompt:
    """Tests for the main parsing function."""

    def test_parse_simple_parentheses(self):
        """Test (text) -> weight 1.1"""
        segments = parse_weighted_prompt("a (cat)")
        assert len(segments) == 2
        assert segments[0] == WeightedSegment(text="a ", weight=1.0)
        assert segments[1] == WeightedSegment(text="cat", weight=1.1)

    def test_parse_nested_parentheses(self):
        """Test ((text)) -> weight 1.21"""
        segments = parse_weighted_prompt("a ((very)) cute cat")
        assert len(segments) == 3
        assert segments[0] == WeightedSegment(text="a ", weight=1.0)
        assert abs(segments[1].weight - 1.21) < 0.001  # 1.1 * 1.1
        assert segments[1].text == "very"
        assert segments[2] == WeightedSegment(text=" cute cat", weight=1.0)

    def test_parse_explicit_weight(self):
        """Test (text:1.5) -> weight 1.5"""
        segments = parse_weighted_prompt("a (cat:1.5)")
        assert len(segments) == 2
        assert segments[0] == WeightedSegment(text="a ", weight=1.0)
        assert segments[1] == WeightedSegment(text="cat", weight=1.5)

    def test_parse_mixed_syntax(self):
        """Test combination of syntaxes"""
        segments = parse_weighted_prompt("a (big:1.3) ((fluffy)) cat")
        assert len(segments) == 5
        assert segments[0] == WeightedSegment(text="a ", weight=1.0)
        assert segments[1] == WeightedSegment(text="big", weight=1.3)
        assert segments[2] == WeightedSegment(text=" ", weight=1.0)
        assert abs(segments[3].weight - 1.21) < 0.001
        assert segments[3].text == "fluffy"
        assert segments[4] == WeightedSegment(text=" cat", weight=1.0)

    def test_empty_prompt(self):
        """Test empty string"""
        segments = parse_weighted_prompt("")
        # Empty prompt returns empty list
        assert len(segments) == 0

    def test_only_weighted(self):
        """Test prompt that's entirely weighted"""
        segments = parse_weighted_prompt("(cat:2.0)")
        assert len(segments) == 1
        assert segments[0] == WeightedSegment(text="cat", weight=2.0)

    def test_triple_nested(self):
        """Test (((text))) -> weight 1.1^3 = 1.331"""
        segments = parse_weighted_prompt("(((cat)))")
        assert len(segments) == 1
        assert segments[0].text == "cat"
        expected_weight = 1.1 * 1.1 * 1.1
        assert abs(segments[0].weight - expected_weight) < 0.001

    def test_edge_cases(self):
        """Test edge cases"""
        # Unbalanced parentheses (missing closing)
        segments = parse_weighted_prompt("a (cat image")
        # Should parse as: "a " and "(cat image" (no closing paren)
        # Since there's no closing paren, it's treated as literal text
        assert len(segments) > 0

        # Multiple weights in sequence
        segments = parse_weighted_prompt("(a:1.5) (b:2.0)")
        assert any(s.weight == 1.5 for s in segments)
        assert any(s.weight == 2.0 for s in segments)


class TestGetTokenPositions:
    """Tests for token position mapping."""

    @pytest.fixture()
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""

        class MockTokenizer:
            def __call__(self, text, add_special_tokens=True, return_tensors=None):
                # Simple mock: each word becomes one token
                words = text.split() if isinstance(text, str) else text
                if isinstance(text, str):
                    tokens = list(range(len(words)))
                else:
                    tokens = [list(range(len(t.split()))) for t in text]

                if return_tensors == "pt":
                    if isinstance(tokens[0], list):
                        return {"input_ids": torch.tensor(tokens)}
                    return {"input_ids": torch.tensor([tokens])}
                return {"input_ids": tokens}

        return MockTokenizer()

    def test_single_segment(self, mock_tokenizer):
        segments = ["hello world"]
        positions = get_token_positions(segments, mock_tokenizer)
        assert len(positions) == 1
        assert positions[0] == (0, 2)  # 2 tokens

    def test_multiple_segments(self, mock_tokenizer):
        segments = ["hello world", "foo bar"]
        positions = get_token_positions(segments, mock_tokenizer)
        assert len(positions) == 2
        assert positions[0] == (0, 2)  # "hello world" = 2 tokens
        assert positions[1] == (2, 4)  # "foo bar" = 2 tokens

    def test_empty_segment(self, mock_tokenizer):
        segments = ["hello", "", "world"]
        positions = get_token_positions(segments, mock_tokenizer)
        assert len(positions) == 3
        # Empty segment should have zero range
        assert positions[1] == (1, 1)


class TestApplyPromptWeights:
    """Tests for weight application."""

    def test_apply_weights_formula(self):
        """Test linear interpolation formula: weighted = (orig - empty) * w + empty"""
        # Create dummy embeddings
        orig_embeds = torch.randn(10, 768)
        empty_embeds = torch.randn(10, 768)

        segments = [WeightedSegment("test", 1.5)]
        positions = [(0, 10)]

        weighted = apply_prompt_weights(orig_embeds, segments, empty_embeds, positions)

        # Verify formula
        expected = (orig_embeds - empty_embeds) * 1.5 + empty_embeds
        assert torch.allclose(weighted, expected)

    def test_weight_one_no_change(self):
        """Test that weight=1.0 produces no change"""
        orig_embeds = torch.randn(10, 768)
        empty_embeds = torch.randn(10, 768)

        segments = [WeightedSegment("test", 1.0)]
        positions = [(0, 10)]

        weighted = apply_prompt_weights(orig_embeds, segments, empty_embeds, positions)

        # Should be unchanged
        assert torch.allclose(weighted, orig_embeds)

    def test_partial_segment_weighting(self):
        """Test weighting only part of the sequence"""
        orig_embeds = torch.randn(10, 768)
        empty_embeds = torch.randn(10, 768)

        segments = [
            WeightedSegment("unweighted", 1.0),
            WeightedSegment("weighted", 1.5),
        ]
        positions = [(0, 5), (5, 10)]

        weighted = apply_prompt_weights(orig_embeds, segments, empty_embeds, positions)

        # First half should be unchanged
        assert torch.allclose(weighted[0:5], orig_embeds[0:5])

        # Second half should be weighted
        expected_second = (orig_embeds[5:10] - empty_embeds[5:10]) * 1.5 + empty_embeds[5:10]
        assert torch.allclose(weighted[5:10], expected_second)

    def test_weight_application_preserves_shape(self):
        """Test that shape is preserved"""
        orig_embeds = torch.randn(77, 2048)
        empty_embeds = torch.randn(77, 2048)

        segments = [WeightedSegment("test", 1.5)]
        positions = [(0, 77)]

        weighted = apply_prompt_weights(orig_embeds, segments, empty_embeds, positions)

        assert weighted.shape == orig_embeds.shape

    def test_mismatched_segments_positions_error(self):
        """Test that mismatched segments and positions raises error"""
        orig_embeds = torch.randn(10, 768)
        empty_embeds = torch.randn(10, 768)

        segments = [WeightedSegment("test", 1.5)]
        positions = [(0, 5), (5, 10)]  # 2 positions but 1 segment

        with pytest.raises(ValueError, match="Mismatched segments"):
            apply_prompt_weights(orig_embeds, segments, empty_embeds, positions)


class TestBackwardCompatibility:
    """Test backward compatibility with non-weighted prompts."""

    def test_plain_prompt_unchanged(self):
        """Test that prompts without weights are parsed correctly"""
        segments = parse_weighted_prompt("a simple cat image")
        assert len(segments) == 1
        assert segments[0].text == "a simple cat image"
        assert segments[0].weight == 1.0

    def test_no_false_positives(self):
        """Test that colons in regular text don't trigger weighting"""
        # This should work, but might parse oddly depending on implementation
        segments = parse_weighted_prompt("image created at 3:00 PM")
        # All segments should have weight 1.0 since there are no parentheses
        assert all(s.weight == 1.0 for s in segments)
