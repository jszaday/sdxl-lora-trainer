"""Content-addressed TensorRT artifact cache paths."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ResolutionSpec

TRT_CACHE_VERSION = "sdxl-unet-trt-v1"


@dataclass(frozen=True)
class EngineCacheKey:
    """Identity for a frozen SDXL UNet TensorRT engine."""

    version: str
    checkpoint: str
    checkpoint_sha256: str
    lora: str | None
    lora_sha256: str | None
    resolution: str
    precision: str
    opset: int
    effective_batch: int = 2

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EngineArtifacts:
    """Paths for cached TensorRT build artifacts."""

    key: EngineCacheKey
    cache_dir: Path
    onnx_path: Path
    engine_path: Path
    metadata_path: Path


def file_sha256(path: Path) -> str:
    """Hash a local file with SHA256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(value: str | Path) -> tuple[str, str]:
    """Return stable identity and content hash for a local file or model id."""
    path = Path(value).expanduser()
    if path.exists() and path.is_file():
        resolved = path.resolve()
        return str(resolved), file_sha256(resolved)
    return str(value), f"model-id:{value}"


def build_engine_cache_key(
    checkpoint: str,
    resolution: ResolutionSpec,
    *,
    precision: str,
    lora_checkpoint: Path | None = None,
    opset: int = 17,
) -> EngineCacheKey:
    """Build the cache key for a frozen SDXL UNet engine."""
    checkpoint_id, checkpoint_hash = artifact_identity(checkpoint)
    lora_id = None
    lora_hash = None
    if lora_checkpoint is not None:
        lora_id, lora_hash = artifact_identity(lora_checkpoint)

    return EngineCacheKey(
        version=TRT_CACHE_VERSION,
        checkpoint=checkpoint_id,
        checkpoint_sha256=checkpoint_hash,
        lora=lora_id,
        lora_sha256=lora_hash,
        resolution=resolution.name,
        precision=precision,
        opset=opset,
    )


def resolve_engine_artifacts(
    engine_dir: Path,
    onnx_dir: Path,
    key: EngineCacheKey,
) -> EngineArtifacts:
    """Resolve content-addressed artifact paths for an engine key."""
    slug = f"unet_{key.precision}_{key.resolution}_{key.digest}"
    cache_dir = Path(engine_dir) / slug
    onnx_cache_dir = Path(onnx_dir) / slug
    return EngineArtifacts(
        key=key,
        cache_dir=cache_dir,
        onnx_path=onnx_cache_dir / "model.onnx",
        engine_path=cache_dir / "model.plan",
        metadata_path=cache_dir / "metadata.json",
    )


def write_metadata(artifacts: EngineArtifacts) -> None:
    """Write cache metadata next to a built engine."""
    artifacts.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": asdict(artifacts.key),
        "digest": artifacts.key.digest,
        "engine_path": str(artifacts.engine_path),
        "onnx_path": str(artifacts.onnx_path),
    }
    artifacts.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
