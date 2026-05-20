"""Build fixed-shape TensorRT engines for frozen SDXL components."""

from pathlib import Path

import torch
import torch.nn as nn

from lora_trainer.model import (
    clear_single_file_cache,
    load_lora_weights,
    load_sdxl_unet,
    merge_lora_layers,
)

from .cache import (
    EngineArtifacts,
    build_engine_cache_key,
    resolve_engine_artifacts,
    write_metadata,
)
from .config import ResolutionSpec


class SdxlUnetExportWrapper(nn.Module):
    """Expose SDXL UNet conditioning tensors as plain ONNX inputs."""

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample


def dtype_from_precision(precision: str) -> torch.dtype:
    """Return torch dtype for an inference precision string."""
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_frozen_unet(
    checkpoint: str,
    *,
    device: str,
    dtype: torch.dtype,
    lora_checkpoint: Path | None = None,
    lora_rank: int = 16,
) -> nn.Module:
    """Load a plain or LoRA-merged SDXL UNet for export."""
    rank = lora_rank if lora_checkpoint is not None else None
    unet, _ = load_sdxl_unet(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=rank,
        adapter="lora",
    )
    if lora_checkpoint is not None:
        load_lora_weights(lora_checkpoint, unet=unet)
        merge_lora_layers(unet)
    unet.requires_grad_(False)
    return unet.eval()


def export_unet_onnx(
    unet: nn.Module,
    onnx_path: Path,
    resolution: ResolutionSpec,
    *,
    device: str,
    dtype: torch.dtype,
    opset: int = 17,
) -> None:
    """Export a fixed-shape CFG-batched SDXL UNet to ONNX."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    batch = 2
    wrapper = SdxlUnetExportWrapper(unet).to(device=device, dtype=dtype).eval()
    sample = torch.randn(
        batch,
        4,
        resolution.latent_height,
        resolution.latent_width,
        device=device,
        dtype=dtype,
    )
    timestep = torch.full((batch,), 999, device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch, 77, 2048, device=device, dtype=dtype)
    text_embeds = torch.randn(batch, 1280, device=device, dtype=dtype)
    time_ids = torch.tensor(
        [[resolution.height, resolution.width, 0, 0, resolution.height, resolution.width]],
        device=device,
        dtype=dtype,
    ).repeat(batch, 1)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
            str(onnx_path),
            input_names=[
                "sample",
                "timestep",
                "encoder_hidden_states",
                "text_embeds",
                "time_ids",
            ],
            output_names=["noise_pred"],
            opset_version=opset,
            do_constant_folding=True,
            external_data=True,
            dynamo=False,
        )
    consolidate_external_data(onnx_path)


def consolidate_external_data(onnx_path: Path) -> None:
    """Rewrite ONNX external tensors into a single sidecar data file.

    PyTorch's legacy exporter may emit one file per initializer. TensorRT's
    parser is more reliable with one colocated external-data file.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("ONNX is required to consolidate exported weights") from exc

    onnx_path = Path(onnx_path)
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )


def build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    *,
    precision: str,
    workspace_gb: float = 8.0,
) -> None:
    """Build a TensorRT plan file from an ONNX graph."""
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("TensorRT Python bindings are not installed") from exc

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX graph:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * 1024**3),
    )
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not report fast fp16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the serialized engine")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def build_unet_engine(
    checkpoint: str,
    resolution: ResolutionSpec,
    *,
    engine_dir: Path,
    onnx_dir: Path,
    precision: str = "fp16",
    device: str = "cuda",
    lora_checkpoint: Path | None = None,
    lora_rank: int = 16,
    opset: int = 17,
    workspace_gb: float = 8.0,
    force: bool = False,
) -> EngineArtifacts:
    """Export ONNX and build the cached fixed-shape UNet engine."""
    dtype = dtype_from_precision(precision)
    key = build_engine_cache_key(
        checkpoint,
        resolution,
        precision=precision,
        lora_checkpoint=lora_checkpoint,
        opset=opset,
    )
    artifacts = resolve_engine_artifacts(engine_dir, onnx_dir, key)

    if artifacts.engine_path.exists() and artifacts.metadata_path.exists() and not force:
        return artifacts

    unet = None
    try:
        unet = load_frozen_unet(
            checkpoint,
            device=device,
            dtype=dtype,
            lora_checkpoint=lora_checkpoint,
            lora_rank=lora_rank,
        )
        if force or not artifacts.onnx_path.exists():
            export_unet_onnx(
                unet,
                artifacts.onnx_path,
                resolution,
                device=device,
                dtype=dtype,
                opset=opset,
            )
        else:
            consolidate_external_data(artifacts.onnx_path)
        build_engine_from_onnx(
            artifacts.onnx_path,
            artifacts.engine_path,
            precision=precision,
            workspace_gb=workspace_gb,
        )
        write_metadata(artifacts)
    finally:
        del unet
        clear_single_file_cache()
    return artifacts
