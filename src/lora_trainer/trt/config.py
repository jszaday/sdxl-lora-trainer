"""Fixed SDXL inference shapes for TensorRT-friendly sampling."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResolutionSpec:
    """A supported SDXL image resolution and its latent dimensions."""

    name: str
    width: int
    height: int

    @property
    def latent_width(self) -> int:
        return self.width // 8

    @property
    def latent_height(self) -> int:
        return self.height // 8

    @property
    def latent_shape(self) -> tuple[int, int]:
        return self.latent_height, self.latent_width


_RESOLUTION_PAIRS = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
    (896, 1152),
    (832, 1216),
    (768, 1344),
    (640, 1536),
]

SDXL_RESOLUTIONS: dict[str, ResolutionSpec] = {
    f"{width}x{height}": ResolutionSpec(
        name=f"{width}x{height}",
        width=width,
        height=height,
    )
    for width, height in _RESOLUTION_PAIRS
}


def parse_resolution(value: str) -> ResolutionSpec:
    """Parse and validate a fixed SDXL resolution string like 1216x832."""
    normalized = value.lower().replace(" ", "")
    if normalized not in SDXL_RESOLUTIONS:
        supported = ", ".join(SDXL_RESOLUTIONS)
        raise ValueError(f"Unsupported SDXL resolution '{value}'. Supported: {supported}")
    return SDXL_RESOLUTIONS[normalized]


def parse_resolution_free(value: str, default: str | None = None) -> ResolutionSpec:
    """Parse any WxH resolution string, not limited to the fixed SDXL list.

    Used for hires-fix where output resolutions are arbitrary (e.g. 2048x2048).
    Pass default to fall back when value is None or empty.
    """
    s = value or default
    if not s:
        raise ValueError("Resolution must be specified as WxH, e.g. '2048x2048'")
    try:
        w, h = s.lower().replace(" ", "").split("x")
        return ResolutionSpec(name=s, width=int(w), height=int(h))
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Resolution must be WxH, e.g. '1024x1024' or '2048x2048'. Got: {s!r}"
        ) from exc


def get_resolution(width: int, height: int) -> ResolutionSpec:
    """Return the supported resolution matching width and height."""
    return parse_resolution(f"{width}x{height}")


def infer_resolution_from_latents(latents: torch.Tensor) -> ResolutionSpec:
    """Infer an SDXL resolution from latent tensor shape."""
    if latents.ndim != 4:
        raise ValueError("Latents must have shape [batch, 4, latent_h, latent_w]")
    if latents.shape[1] != 4:
        raise ValueError("SDXL latents must have 4 channels")

    latent_h = int(latents.shape[2])
    latent_w = int(latents.shape[3])
    for spec in SDXL_RESOLUTIONS.values():
        if spec.latent_shape == (latent_h, latent_w):
            return spec

    supported = ", ".join(
        f"{spec.name} -> {spec.latent_height}x{spec.latent_width}"
        for spec in SDXL_RESOLUTIONS.values()
    )
    raise ValueError(
        f"Unsupported latent spatial shape {latent_h}x{latent_w}. Supported: {supported}"
    )


def validate_latents_shape(
    latents: torch.Tensor,
    resolution: ResolutionSpec,
    *,
    batch_size: int = 1,
) -> None:
    """Validate latents against the selected SDXL resolution."""
    expected = (batch_size, 4, resolution.latent_height, resolution.latent_width)
    actual = tuple(int(dim) for dim in latents.shape)
    if actual != expected:
        raise ValueError(f"Expected latents shape {expected} for {resolution.name}, got {actual}")
