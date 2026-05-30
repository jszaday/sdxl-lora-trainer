"""GPU-resident app server: loads models once, serves inference requests over a Unix socket."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import torch

from .hires import InferencePipeline
from .pipeline import dtype_from_precision, load_inference_models, load_torch_unet_backend
from .trt.config import ResolutionSpec, parse_resolution, parse_resolution_free
from .trt.upscaler import CompiledUpscalerBackend
from .utils import save_images

_SOCKET_DEFAULT = "/tmp/lora_hires.sock"
_LEN_FMT = ">I"
_HEARTBEAT_INTERVAL = 5.0  # seconds between server pings
_HEARTBEAT_TIMEOUT = 15.0  # seconds to wait for client pong before shutdown


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(precision: str, device: str) -> torch.dtype:
    _ = device
    return dtype_from_precision(precision)


def _send_msg(conn: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack(_LEN_FMT, len(data)) + data)


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> dict | None:
    header = _recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack(_LEN_FMT, header)
    body = _recv_exact(conn, length)
    if body is None:
        return None
    return json.loads(body)


def _warmup(pipeline: InferencePipeline) -> None:
    """Single-step dummy pass to trigger torch.compile before the first real request."""
    from .sampling import encode_prompts_for_sampling
    from .trt.inference import sample_frozen_sdxl

    print("Warming up UNet (torch.compile) …", flush=True)
    dummy_res = ResolutionSpec(name="1024x1024", width=1024, height=1024)
    pos, pos_p = encode_prompts_for_sampling(
        ["warmup"],
        pipeline.te1,
        pipeline.te2,
        pipeline.tok1,
        pipeline.tok2,
        pipeline.device,
        clip_skip=1,
        enable_prompt_weighting=False,
    )
    neg, neg_p = encode_prompts_for_sampling(
        [""],
        pipeline.te1,
        pipeline.te2,
        pipeline.tok1,
        pipeline.tok2,
        pipeline.device,
        clip_skip=1,
        enable_prompt_weighting=False,
    )
    sample_frozen_sdxl(
        pipeline.unet_backend,
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        pooled_prompt_embeds=pos_p,
        pooled_negative_prompt_embeds=neg_p,
        resolution=dummy_res,
        sampler="euler",
        scheduler_name="karras",
        num_inference_steps=1,
        guidance_scale=1.0,
        device=pipeline.device,
        dtype=pipeline.dtype,
        seed=0,
        progress=False,
    )
    print("Warm-up done.", flush=True)


def _run_health_monitor(conn: socket.socket) -> None:
    """Ping the client every N seconds; shut down if it goes silent or disconnects."""
    print("Health monitor: client connected.", flush=True)
    try:
        while True:
            time.sleep(_HEARTBEAT_INTERVAL)
            try:
                _send_msg(conn, {"type": "ping"})
            except OSError:
                break
            conn.settimeout(_HEARTBEAT_TIMEOUT)
            try:
                msg = _recv_msg(conn)
            except OSError:
                msg = None
            finally:
                conn.settimeout(None)
            if msg is None or msg.get("type") != "pong":
                break
    finally:
        conn.close()
    print("Health monitor: client gone — shutting down.", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)


def _handle(req: dict, *, pipeline: InferencePipeline) -> dict:
    req_type = req.get("type", "simple")

    from .trt.backends import TensorRTUnetBackend

    if req_type == "hires" and isinstance(pipeline.unet_backend, TensorRTUnetBackend):
        return {
            "status": "error",
            "message": (
                "TRT backend does not support hires mode (engines are resolution-specific). "
                "Restart the server with --backend torch for hires-fix."
            ),
        }

    resolution = parse_resolution_free(req.get("resolution"), default="1024x1024")
    seed_raw = str(req.get("seed", 42))
    seeds = [int(s.strip()) for s in seed_raw.split(",")]
    seed = seeds[0]

    if pipeline.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    with torch.inference_mode():
        if req_type == "simple":
            images, _ = pipeline(
                req.get("prompt", ""),
                req.get("negative", ""),
                mode="simple",
                resolution=resolution,
                steps=int(req.get("steps", 30)),
                denoise=float(req.get("denoise", 1.0)),
                cfg=float(req.get("cfg", 7.0)),
                sampler=req.get("sampler", "euler"),
                scheduler_name=req.get("scheduler", "karras"),
                clip_skip=int(req.get("clip_skip", 1)),
                seed=seed,
                enable_prompt_weighting=bool(req.get("enable_prompt_weighting", True)),
                progress=bool(req.get("progress", True)),
                face_denoise=float(req.get("face_denoise", 0.4)),
                face_steps=int(req.get("face_steps", 20)),
                face_size=int(req.get("face_size", 512)),
                face_padding=int(req.get("face_padding", 32)),
                face_feather=int(req.get("face_feather", 16)),
            )
            output_path = Path(req.get("output", "output.png"))
            save_images(images, output_path)
            resp: dict = {"status": "ok", "output": str(output_path)}
        else:
            hires_res = (
                parse_resolution_free(req["hires_resolution"])
                if req.get("hires_resolution")
                else resolution
            )
            first_images, images = pipeline(
                req.get("prompt", ""),
                req.get("negative", ""),
                mode="hires",
                resolution=resolution,
                hires_resolution=hires_res,
                steps=int(req.get("first_steps", 30)),
                denoise=float(req.get("first_denoise", 1.0)),
                second_steps=int(req.get("second_steps", 20)),
                second_denoise=float(req.get("second_denoise", 0.65)),
                cfg=float(req.get("cfg", 7.0)),
                sampler=req.get("sampler", "euler"),
                scheduler_name=req.get("scheduler", "karras"),
                clip_skip=int(req.get("clip_skip", 1)),
                seed=seed,
                enable_prompt_weighting=bool(req.get("enable_prompt_weighting", True)),
                progress=bool(req.get("progress", True)),
                face_denoise=float(req.get("face_denoise", 0.4)),
                face_steps=int(req.get("face_steps", 20)),
                face_size=int(req.get("face_size", 512)),
                face_padding=int(req.get("face_padding", 32)),
                face_feather=int(req.get("face_feather", 16)),
            )
            output_path = Path(req.get("output", "output.png"))
            save_images(images, output_path)
            resp = {"status": "ok", "output": str(output_path)}
            if req.get("save_first"):
                first_path = Path(req["save_first"])
                save_images(first_images, first_path)
                resp["save_first"] = str(first_path)

    resp["elapsed"] = round(time.perf_counter() - t0, 3)
    if pipeline.device == "cuda":
        resp["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        resp["reserved_vram_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    return resp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "GPU-resident inference server. Loads models once at startup and serves both "
            "plain SDXL (type=simple) and hires-fix (type=hires) requests over a Unix socket."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="SDXL checkpoint path or HF model ID.")
    p.add_argument(
        "--upscale_model",
        type=Path,
        default=None,
        help="Spandrel upscale model (.safetensors or .pth). Required for hires requests.",
    )
    p.add_argument(
        "--face_model",
        type=Path,
        default=None,
        help="YOLO face model (.pt). Enables ADetailer when set.",
    )
    p.add_argument("--lora_checkpoint", type=Path, default=None)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--backend",
        choices=["torch", "trt"],
        default="torch",
        help="UNet backend. TRT only supports simple (non-hires) requests.",
    )
    p.add_argument(
        "--resolution",
        default="1024x1024",
        help="TRT engine resolution (WxH). Only used when --backend trt.",
    )
    p.add_argument(
        "--engine_dir",
        type=Path,
        default=Path("engines"),
        help="Directory for TRT engine files.",
    )
    p.add_argument(
        "--onnx_dir",
        type=Path,
        default=Path("engines/onnx"),
        help="Directory for ONNX export files.",
    )
    p.add_argument(
        "--compile_unet",
        action="store_true",
        help="Apply torch.compile(reduce-overhead) to the UNet (torch backend only).",
    )
    p.add_argument(
        "--upscale_tile_size",
        type=int,
        default=512,
        help="Tile size for the upscale model.",
    )
    p.add_argument("--upscale_overlap", type=int, default=32)
    p.add_argument("--socket", default=_SOCKET_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or _default_device()
    if args.backend == "trt" and not device.startswith("cuda"):
        raise ValueError("TensorRT backend requires a CUDA device; use --backend torch.")
    dtype = _dtype_for_device(args.precision, device)

    print(f"Device: {device} ({args.precision})")
    print(f"Loading SDXL: {args.checkpoint}")
    vae, te1, te2, tok1, tok2 = load_inference_models(
        args.checkpoint,
        device=device,
        dtype=dtype,
        lora_checkpoint=args.lora_checkpoint,
        lora_rank=args.lora_rank,
    )

    if args.backend == "trt":
        from .pipeline import prepare_trt_engine
        from .trt.backends import TensorRTUnetBackend

        trt_res = parse_resolution(args.resolution)
        engine_path = prepare_trt_engine(
            args.checkpoint,
            trt_res,
            engine_dir=args.engine_dir,
            onnx_dir=args.onnx_dir,
            precision=args.precision,
            device=device,
            lora_checkpoint=args.lora_checkpoint,
            lora_rank=args.lora_rank,
        )
        unet_backend = TensorRTUnetBackend(engine_path)
        print(f"TRT engine loaded: {engine_path.name}")
    else:
        unet_backend = load_torch_unet_backend(
            args.checkpoint,
            device=device,
            dtype=dtype,
            lora_checkpoint=args.lora_checkpoint,
            lora_rank=args.lora_rank,
            compile_unet=args.compile_unet,
            flash_attention=True,
        )

    upscaler = None
    if args.upscale_model is not None:
        print(f"Loading upscale model: {args.upscale_model.name}")
        upscaler = CompiledUpscalerBackend(
            args.upscale_model, device=device, tile_size=args.upscale_tile_size
        )

    face_detector = None
    if args.face_model is not None:
        from ultralytics import YOLO

        print(f"Loading face model: {args.face_model.name}")
        face_detector = YOLO(str(args.face_model))

    pipeline = InferencePipeline(
        vae,
        te1,
        te2,
        tok1,
        tok2,
        unet_backend,
        upscaler=upscaler,
        face_detector=face_detector,
        device=device,
        dtype=dtype,
        upscale_tile_size=args.upscale_tile_size,
        upscale_overlap=args.upscale_overlap,
    )

    if args.compile_unet and args.backend == "torch":
        _warmup(pipeline)

    sock_path = Path(args.socket)
    if sock_path.exists():
        sock_path.unlink()

    def _cleanup(sig, frame):  # noqa: ARG001
        print("\nShutting down …")
        if sock_path.exists():
            sock_path.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
        srv.bind(str(sock_path))
        srv.listen(1)
        print(f"\nReady. Listening on {sock_path}")
        print("Ctrl-C to stop.\n", flush=True)

        while True:
            conn, _ = srv.accept()
            try:
                req = _recv_msg(conn)
            except Exception:
                conn.close()
                continue
            if req is None:
                conn.close()
                continue

            if req.get("type") == "health":
                # Persistent connection — hand off to monitor thread, do NOT close here.
                threading.Thread(target=_run_health_monitor, args=(conn,), daemon=True).start()
                continue

            # Inference request — handle and close.
            req_type = req.get("type", "simple")
            print(f"→ [{req_type}] {req.get('prompt', '')[:60]!r}", flush=True)
            try:
                resp = _handle(req, pipeline=pipeline)
                print(
                    f"  done in {resp.get('elapsed', 0):.2f}s  →  {resp.get('output', '')}",
                    flush=True,
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                resp = {"status": "error", "message": str(exc)}
            try:
                _send_msg(conn, resp)
            except Exception:
                pass
            conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
