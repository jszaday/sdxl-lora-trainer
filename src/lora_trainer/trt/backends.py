"""UNet backends for frozen SDXL inference."""

from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn


class UnetBackend(Protocol):
    """Callable UNet backend used inside the denoise loop."""

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        added_cond_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return predicted noise tensor."""


class TorchUnetBackend:
    """Plain PyTorch UNet backend, optionally torch.compile'd."""

    def __init__(self, unet: nn.Module, *, compile_unet: bool = False):
        self.unet = unet.eval()
        if compile_unet:
            self.unet = torch.compile(self.unet, mode="reduce-overhead")

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        added_cond_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample


class TensorRTUnavailableError(RuntimeError):
    """Raised when a requested TensorRT backend cannot run."""


def unet_engine_path(engine_dir: Path, resolution, precision: str) -> Path:
    """Return the conventional UNet engine path for a resolution."""
    return Path(engine_dir) / f"unet_{precision}_{resolution.name}.plan"


class TensorRTUnetBackend:
    """TensorRT plan-backed UNet execution using torch CUDA tensor buffers."""

    def __init__(
        self,
        engine_path: Path | None = None,
        *,
        engine_dir: Path | None = None,
        resolution=None,
        precision: str = "fp16",
    ):
        if engine_path is None:
            if engine_dir is None or resolution is None:
                raise TensorRTUnavailableError(
                    "TensorRTUnetBackend requires engine_path or engine_dir + resolution"
                )
            engine_path = unet_engine_path(engine_dir, resolution, precision)
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise TensorRTUnavailableError(
                f"Missing TensorRT UNet engine: {self.engine_path}. "
                "Run with --backend torch or build this engine first."
            )

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise TensorRTUnavailableError(
                "TensorRT Python bindings are not installed. "
                "Install TensorRT for this CUDA environment or use --backend torch."
            ) from exc

        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if engine is None:
            raise TensorRTUnavailableError(
                f"Failed to deserialize TensorRT engine: {self.engine_path}"
            )

        self.runtime = runtime
        self.engine = engine
        self.context = engine.create_execution_context()
        self.input_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        expected_inputs = {"sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"}
        missing = expected_inputs.difference(self.input_names)
        if missing:
            raise TensorRTUnavailableError(
                f"TensorRT engine is missing expected inputs: {', '.join(sorted(missing))}"
            )
        if self.output_names != ["noise_pred"]:
            raise TensorRTUnavailableError(
                f"TensorRT engine must expose one 'noise_pred' output, got {self.output_names}"
            )
        self.stream: torch.cuda.Stream | None = None
        self.stream_device: torch.device | None = None

    def _execution_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Return a non-default CUDA stream for TensorRT execution."""
        if self.stream is None or self.stream_device != device:
            self.stream = torch.cuda.Stream(device=device)
            self.stream_device = device
        return self.stream

    def __call__(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        added_cond_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not sample.is_cuda:
            raise TensorRTUnavailableError("TensorRT UNet inputs must be CUDA tensors")

        inputs = {
            "sample": sample.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
            "text_embeds": added_cond_kwargs["text_embeds"].contiguous(),
            "time_ids": added_cond_kwargs["time_ids"].contiguous(),
        }
        output = torch.empty_like(inputs["sample"])

        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.context.set_tensor_address("noise_pred", output.data_ptr())

        current_stream = torch.cuda.current_stream(sample.device)
        execution_stream = self._execution_stream(sample.device)
        execution_stream.wait_stream(current_stream)
        ok = self.context.execute_async_v3(stream_handle=execution_stream.cuda_stream)
        if not ok:
            raise TensorRTUnavailableError("TensorRT UNet execution failed")
        current_stream.wait_stream(execution_stream)
        return output
