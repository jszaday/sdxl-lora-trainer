"""TensorRT-accelerated tiled image upscaling via spandrel models."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def _load_spandrel(model_path: Path) -> tuple[nn.Module, int]:
    """Load a spandrel upscale model. Returns (inner_module, scale)."""
    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as exc:
        raise RuntimeError("spandrel is required: pip install spandrel") from exc

    path = Path(model_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(path))
    else:
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = {k.removeprefix("module."): v for k, v in sd.items()}

    desc = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(desc, ImageModelDescriptor):
        raise RuntimeError(f"{path.name} is not a single-image upscale model")
    return desc.model, int(desc.scale)


def export_upscaler_onnx(
    model_path: Path,
    onnx_path: Path,
    *,
    tile_size: int = 512,
    device: str = "cuda",
    opset: int = 17,
) -> int:
    """Export an upscale model to fixed-shape ONNX. Returns scale factor."""
    model, scale = _load_spandrel(Path(model_path))
    model = model.to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)

    dummy = torch.zeros(1, 3, tile_size, tile_size, device=device, dtype=torch.float32)

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["image"],
            output_names=["upscaled"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return scale


def build_upscaler_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    tile_size: int,
    scale: int,
    precision: str = "fp16",
    workspace_gb: float = 4.0,
) -> None:
    """Build a TensorRT plan from an upscaler ONNX and write a metadata sidecar."""
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are not installed") from exc

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.NONE
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("GPU does not support fast fp16")
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    meta = {"tile_size": tile_size, "scale": scale, "precision": precision}
    engine_path.with_suffix(".json").write_text(json.dumps(meta))


class TRTUpscalerTile:
    """Runs fixed-shape upscaler tiles through a TensorRT engine."""

    def __init__(self, engine_path: Path):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT not installed") from exc

        engine_path = Path(engine_path)
        meta_path = engine_path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.scale: int = meta["scale"]
            self.tile_size: int = meta["tile_size"]
        else:
            raise RuntimeError(f"Missing engine metadata: {meta_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self._runtime = runtime

    @torch.inference_mode()
    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        """Upscale a single [1, 3, H, W] tile. Input must be fp16 CUDA."""
        b, c, h, w = tile.shape
        out = torch.empty(
            b, c, h * self.scale, w * self.scale, dtype=tile.dtype, device=tile.device
        )
        t = tile.contiguous()
        self.context.set_tensor_address("image", t.data_ptr())
        self.context.set_tensor_address("upscaled", out.data_ptr())
        # Myelin (DAT attention) uses internal streams; sync after execute to ensure
        # output is ready before PyTorch reads it on the same or other streams.
        stream = torch.cuda.current_stream(tile.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT upscaler execution failed")
        stream.synchronize()
        return out


def tiled_upscale(
    image: torch.Tensor,
    tile_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile_size: int = 512,
    overlap: int = 32,
    scale: int = 4,
) -> torch.Tensor:
    """Tile-stitch upscale of a [1, 3, H, W] image tensor.

    Tiles overlap by `overlap` pixels; overlapping regions are averaged.
    This matches ComfyUI's tiled_scale approach without Gaussian blending.
    """
    _, c, h, w = image.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, c, out_h, out_w, dtype=image.dtype, device=image.device)
    counts = torch.zeros(1, 1, out_h, out_w, dtype=torch.float32, device=image.device)

    step = tile_size - overlap

    def _tile_starts(dim: int) -> list[int]:
        starts = list(range(0, dim - tile_size + 1, step))
        if not starts or starts[-1] + tile_size < dim:
            starts.append(max(0, dim - tile_size))
        return sorted(set(starts))

    for y0 in _tile_starts(h):
        y1 = min(y0 + tile_size, h)
        for x0 in _tile_starts(w):
            x1 = min(x0 + tile_size, w)
            tile = image[:, :, y0:y1, x0:x1]
            ph, pw = tile_size - (y1 - y0), tile_size - (x1 - x0)
            if ph > 0 or pw > 0:
                tile = F.pad(tile, (0, pw, 0, ph), mode="reflect")
            upscaled = tile_fn(tile)
            uh, uw = (y1 - y0) * scale, (x1 - x0) * scale
            upscaled = upscaled[:, :, :uh, :uw]
            oy0, ox0 = y0 * scale, x0 * scale
            output[:, :, oy0 : oy0 + uh, ox0 : ox0 + uw] += upscaled
            counts[:, :, oy0 : oy0 + uh, ox0 : ox0 + uw] += 1.0

    return (output / counts.clamp(min=1e-6)).clamp(0.0, 1.0)


class CompiledUpscalerBackend:
    """torch.compile-accelerated upscaler. Always runs fp32: DAT and similar attention-based
    models overflow in fp16 (same reason ComfyUI forces .float() before every upscale call).
    """

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda",
        tile_size: int = 512,
        compile_mode: str = "reduce-overhead",
    ):
        model, self.scale = _load_spandrel(Path(model_path))
        model = model.to(device=device, dtype=torch.float32).eval()

        # Pre-warm with a fp32 forward so any lazy dtype promotions (DAT's
        # self.mean = self.mean.type_as(x)) settle before torch.compile captures CUDA graphs.
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, tile_size, tile_size, device=device, dtype=torch.float32)
            model(dummy)

        self._model = torch.compile(model, mode=compile_mode, fullgraph=False)
        self._device = device

    @torch.inference_mode()
    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        return self._model(tile.to(device=self._device, dtype=torch.float32))


def benchmark_upscaler(
    model_path: Path,
    *,
    engine_dir: Path,
    onnx_dir: Path,
    image_size: int = 1024,
    tile_size: int = 512,
    overlap: int = 32,
    precision: str = "fp16",
    device: str = "cuda",
    opset: int = 17,
    workspace_gb: float = 4.0,
    force: bool = False,
    warmup: int = 2,
    runs: int = 5,
    skip_trt: bool = False,
) -> dict[str, float]:
    """Benchmark PyTorch fp32 vs torch.compile fp16 vs TRT fp16 for a tiled upscale pass.

    Returns dict with keys: 'torch_fp32_s', 'compiled_fp32_s', 'trt_fp16_s' (if built).
    """
    model_path = Path(model_path)

    image = torch.rand(1, 3, image_size, image_size, device=device, dtype=torch.float32)
    results: dict[str, float] = {}

    # --- PyTorch fp32 baseline (matches ComfyUI's .float() cast) ---
    torch_model, scale = _load_spandrel(model_path)
    torch_model = torch_model.to(device=device, dtype=torch.float32).eval()

    def torch_tile_fn(tile: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return torch_model(tile.float())  # noqa: F821

    print(f"Warming up PyTorch fp32 ({warmup} passes) …")
    for _ in range(warmup):
        tiled_upscale(image, torch_tile_fn, tile_size=tile_size, overlap=overlap, scale=scale)
    torch.cuda.synchronize()

    print(f"Timing PyTorch fp32 ({runs} runs) …")
    t0 = time.perf_counter()
    for _ in range(runs):
        tiled_upscale(image, torch_tile_fn, tile_size=tile_size, overlap=overlap, scale=scale)
        torch.cuda.synchronize()
    results["torch_fp32_s"] = (time.perf_counter() - t0) / runs
    del torch_model

    # --- torch.compile fp32 ---
    compiled_backend = CompiledUpscalerBackend(model_path, device=device, tile_size=tile_size)

    print(f"\nWarming up torch.compile fp32 ({warmup} passes, first triggers JIT) …")
    for _ in range(warmup):
        tiled_upscale(image, compiled_backend, tile_size=tile_size, overlap=overlap, scale=scale)
    torch.cuda.synchronize()

    print(f"Timing torch.compile fp32 ({runs} runs) …")
    t0 = time.perf_counter()
    for _ in range(runs):
        tiled_upscale(image, compiled_backend, tile_size=tile_size, overlap=overlap, scale=scale)
        torch.cuda.synchronize()
    results["compiled_fp32_s"] = (time.perf_counter() - t0) / runs

    # --- TRT fp16 (optional — known to have Myelin instability on some Blackwell GPUs) ---
    if not skip_trt:
        stem = model_path.stem
        onnx_path = onnx_dir / f"{stem}_tile{tile_size}.onnx"
        engine_path = engine_dir / f"{stem}_tile{tile_size}_{precision}.plan"

        if force or not engine_path.exists():
            print("\nExporting ONNX …")
            export_upscaler_onnx(
                model_path, onnx_path, tile_size=tile_size, device=device, opset=opset
            )
            print("Building TRT engine (takes a few minutes) …")
            build_upscaler_engine(
                onnx_path,
                engine_path,
                tile_size=tile_size,
                scale=scale,
                precision=precision,
                workspace_gb=workspace_gb,
            )
            print(f"Engine: {engine_path}")
        else:
            print(f"\nReusing TRT engine: {engine_path}")

        try:
            backend = TRTUpscalerTile(engine_path)
            image_fp16 = image.half()

            print(f"Warming up TRT fp16 ({warmup} passes) …")
            for _ in range(warmup):
                tiled_upscale(
                    image_fp16, backend, tile_size=tile_size, overlap=overlap, scale=scale
                )  # noqa: E501
            torch.cuda.synchronize()

            print(f"Timing TRT fp16 ({runs} runs) …")
            t0 = time.perf_counter()
            for _ in range(runs):
                tiled_upscale(
                    image_fp16, backend, tile_size=tile_size, overlap=overlap, scale=scale
                )  # noqa: E501
                torch.cuda.synchronize()
            results["trt_fp16_s"] = (time.perf_counter() - t0) / runs
        except Exception as exc:
            print(f"TRT benchmark failed: {exc}")
            results["trt_fp16_s"] = float("nan")

    return results
