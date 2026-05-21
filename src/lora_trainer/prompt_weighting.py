"""Prompt weighting support for A1111/ComfyUI-style syntax.

Implements parsing and application of prompt weights like:
- (text) -> weight * 1.1
- ((text)) -> weight * 1.1 * 1.1 = 1.21
- (text:1.5) -> weight = 1.5

Based on ComfyUI's implementation in comfy/sd1_clip.py.
"""

from dataclasses import dataclass

import torch


@dataclass
class WeightedSegment:
    """Represents a text segment with its weight multiplier."""

    text: str
    weight: float


def parse_parentheses(string: str) -> list[str]:
    """Parse string into top-level parentheses groups.

    Examples:
        "a (b) c" -> ["a ", "(b)", " c"]
        "a ((b)) c" -> ["a ", "((b))", " c"]
        "(a:1.5)" -> ["(a:1.5)"]

    Args:
        string: Input string to parse

    Returns:
        List of string segments, with parentheses groups intact
    """
    result = []
    current = []
    depth = 0

    for char in string:
        if char == "(":
            if depth == 0 and current:
                # Save accumulated text before this group
                result.append("".join(current))
                current = []
            current.append(char)
            depth += 1
        elif char == ")":
            current.append(char)
            depth -= 1
            if depth == 0:
                # Complete top-level group
                result.append("".join(current))
                current = []
        else:
            current.append(char)

    if current:
        result.append("".join(current))

    return result


def token_weights(string: str, current_weight: float) -> list[tuple[str, float]]:
    """Recursively parse prompt weights from parentheses syntax.

    Supports:
    - (text) -> multiply weight by 1.1
    - ((text)) -> nested multiplication: 1.1 * 1.1 = 1.21
    - (text:1.5) -> explicit weight of 1.5

    Args:
        string: Prompt string to parse
        current_weight: Current accumulated weight from outer scopes

    Returns:
        List of (text, weight) tuples
    """
    segments = parse_parentheses(string)
    result = []

    for segment in segments:
        weight = current_weight

        # Check if this is a parenthesized group
        if len(segment) >= 2 and segment[0] == "(" and segment[-1] == ")":
            # Strip outer parentheses
            inner = segment[1:-1]

            # Default: multiply by 1.1
            weight *= 1.1

            # Check for explicit weight syntax (text:1.5)
            colon_idx = inner.rfind(":")
            if colon_idx > 0:
                try:
                    weight = float(inner[colon_idx + 1 :])
                    inner = inner[:colon_idx]
                except ValueError:
                    # Not a valid weight, treat the colon as literal text
                    pass

            # Recursively process the inner content
            result.extend(token_weights(inner, weight))
        else:
            # Plain text segment
            result.append((segment, current_weight))

    return result


def parse_weighted_prompt(prompt: str) -> list[WeightedSegment]:
    """Parse a prompt with ComfyUI-style weighting syntax.

    Supports:
    - (text) -> weight *= 1.1
    - ((text)) -> weight *= 1.1 * 1.1 = 1.21
    - (text:1.5) -> weight = 1.5
    - Nested parentheses with compounding

    Args:
        prompt: Input prompt string

    Returns:
        List of WeightedSegment objects with text and weight

    Examples:
        >>> parse_weighted_prompt("a (cat)")
        [WeightedSegment(text='a ', weight=1.0), WeightedSegment(text='cat', weight=1.1)]

        >>> parse_weighted_prompt("a ((very)) cute cat")
        [WeightedSegment(text='a ', weight=1.0),
         WeightedSegment(text='very', weight=1.21),
         WeightedSegment(text=' cute cat', weight=1.0)]

        >>> parse_weighted_prompt("a (cat:1.5)")
        [WeightedSegment(text='a ', weight=1.0), WeightedSegment(text='cat', weight=1.5)]
    """
    # Parse into (text, weight) tuples
    tuples = token_weights(prompt, 1.0)

    # Convert to WeightedSegment objects
    segments = [WeightedSegment(text=text, weight=weight) for text, weight in tuples]

    return segments


def apply_prompt_weights(
    prompt_embeds: torch.Tensor,
    weighted_segments: list[WeightedSegment],
    empty_embeds: torch.Tensor,
    token_positions: list[tuple[int, int]],
) -> torch.Tensor:
    """Apply weights to prompt embeddings using linear interpolation.

    Uses ComfyUI's formula:
        weighted_embedding = (embedding - empty_embedding) * weight + empty_embedding

    This interpolates between the empty embedding (baseline) and the actual embedding:
    - weight=0.0: Result is the empty embedding
    - weight=1.0: Result is the original embedding
    - weight>1.0: Emphasizes the difference (amplifies semantic meaning)
    - weight<1.0: De-emphasizes (brings closer to neutral)

    Args:
        prompt_embeds: Token embeddings [seq_len, hidden_dim]
        weighted_segments: List of WeightedSegment with text and weight
        empty_embeds: Embeddings for empty prompt [seq_len, hidden_dim]
        token_positions: List of (start_idx, end_idx) for each segment

    Returns:
        Weighted embeddings [seq_len, hidden_dim]
    """
    if len(weighted_segments) != len(token_positions):
        raise ValueError(
            f"Mismatched segments ({len(weighted_segments)}) and positions ({len(token_positions)})"
        )

    # Clone to avoid modifying original
    result = prompt_embeds.clone()

    # Apply weights to each segment
    for segment, (start_idx, end_idx) in zip(weighted_segments, token_positions, strict=True):
        if segment.weight == 1.0:
            # No modification needed
            continue

        # Apply linear interpolation formula
        # weighted = (orig - empty) * weight + empty
        orig = prompt_embeds[start_idx:end_idx]
        empty = empty_embeds[start_idx:end_idx]
        result[start_idx:end_idx] = (orig - empty) * segment.weight + empty

    return result


def get_token_positions(
    segments: list[str],
    tokenizer,
    max_length: int = 77,
) -> list[tuple[int, int]]:
    """Map text segments to token position ranges.

    Tokenizes each segment and tracks where its tokens appear in the
    final padded sequence.

    Args:
        segments: List of text strings (from WeightedSegment.text)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default: 77 for CLIP)

    Returns:
        List of (start_idx, end_idx) tuples for each segment

    Note:
        Special tokens (BOS, EOS, PAD) are handled by the tokenizer.
        Token positions may not exactly match text boundaries due to
        subword tokenization.
    """
    positions = []
    current_pos = 0

    # Tokenize each segment to get token counts
    for segment in segments:
        if not segment:
            # Empty segment, no tokens
            positions.append((current_pos, current_pos))
            continue

        # Tokenize this segment (without padding or truncation yet)
        result = tokenizer(
            segment,
            add_special_tokens=False,  # Don't add BOS/EOS here
            return_tensors="pt",
        )
        # Handle both dict and object return types
        tokens = result["input_ids"][0] if isinstance(result, dict) else result.input_ids[0]

        segment_len = len(tokens)
        end_pos = current_pos + segment_len

        # Check for truncation
        if end_pos > max_length:
            # Truncate this segment
            end_pos = max_length
            segment_len = max_length - current_pos

        positions.append((current_pos, end_pos))
        current_pos = end_pos

        # Stop if we've hit the max length
        if current_pos >= max_length:
            break

    # Handle remaining segments if we truncated
    while len(positions) < len(segments):
        # Any remaining segments get zero range
        positions.append((max_length, max_length))

    return positions
