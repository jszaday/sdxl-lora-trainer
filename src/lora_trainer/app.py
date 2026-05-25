"""Streamlit inference app — SDXL with optional hires-fix, client/server architecture."""

from __future__ import annotations

import argparse
import atexit
import json
import random
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
import weakref
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal

import streamlit as st
from PIL import Image

from lora_trainer.pipeline import SAMPLERS, SCHEDULERS
from lora_trainer.trt.config import SDXL_RESOLUTIONS

_NO_MODEL = "— None —"
_LEN_FMT = ">I"
_SETTINGS_PATH = Path.home() / ".config" / "lora-trainer" / "app_settings.json"

_managed_procs: list[subprocess.Popen] = []


@atexit.register
def _cleanup_procs() -> None:
    for proc in _managed_procs:
        if proc.poll() is None:
            proc.terminate()


# --- settings ---


@dataclass
class ModelSettings:
    checkpoint: str = ""
    upscale_model: str = _NO_MODEL
    face_model: str = _NO_MODEL
    lora: str = _NO_MODEL
    lora_rank: int = 16


@dataclass
class SamplingSettings:
    sampler: str = "euler"
    scheduler: str = "normal"
    cfg: float = 7.0
    clip_skip: int = 1


@dataclass
class SimpleSettings:
    steps: int = 25
    denoise: float = 1.0


@dataclass
class HiresSettings:
    hires_resolution: str = "1024x1024"
    first_steps: int = 25
    first_denoise: float = 1.0
    second_steps: int = 25
    second_denoise: float = 0.7


@dataclass
class SeedSettings:
    random_seed: bool = True
    seed: int = 42


@dataclass
class BackendSettings:
    backend: Literal["torch", "trt"] = "torch"
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"
    compile_unet: bool = False


@dataclass
class ADetailerSettings:
    face_denoise: float = 0.4
    face_steps: int = 20
    face_size: int = 512
    face_padding: int = 32
    face_feather: int = 16


@dataclass
class PromptSettings:
    prompt: str = ""
    negative: str = ""


@dataclass
class AppSettings:
    mode: Literal["simple", "hires"] = "simple"
    resolution: str = "1024x1024"
    models: ModelSettings = field(default_factory=ModelSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    simple: SimpleSettings = field(default_factory=SimpleSettings)
    hires: HiresSettings = field(default_factory=HiresSettings)
    seed: SeedSettings = field(default_factory=SeedSettings)
    backend: BackendSettings = field(default_factory=BackendSettings)
    adetailer: ADetailerSettings = field(default_factory=ADetailerSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)


def _dataclass_from_dict(cls, data: object):
    if not isinstance(data, dict):
        return cls()
    kwargs = {}
    for item in fields(cls):
        default = getattr(cls(), item.name)
        value = data.get(item.name, default)
        if is_dataclass(default):
            value = _dataclass_from_dict(type(default), value)
        kwargs[item.name] = value
    return cls(**kwargs)


def _load_settings() -> AppSettings:
    try:
        return _dataclass_from_dict(AppSettings, json.loads(_SETTINGS_PATH.read_text()))
    except Exception:
        return AppSettings()


def _reset_defaults() -> None:
    st.session_state["settings"] = AppSettings()
    for key in list(st.session_state.keys()):
        if str(key).startswith("ui."):
            st.session_state.pop(key, None)
    _SETTINGS_PATH.unlink(missing_ok=True)


def _save_settings(settings: AppSettings) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2, sort_keys=True))


def _settings() -> AppSettings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = _load_settings()
    return st.session_state["settings"]


def _choice_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _valid_choice(value: str, options: list[str], default: str) -> str:
    return value if value in options else default


# --- model scanning ---


def _scan(directory: Path | None, *globs: str) -> list[Path]:
    if not directory or not directory.exists():
        return []
    results: list[Path] = []
    for g in globs:
        results.extend(directory.glob(g))
    return sorted(set(results))


# --- socket protocol ---


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
    return json.loads(body) if body else None


def _request(sock_path: str, req: dict, timeout: float = 600.0) -> dict | None:
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(timeout)
        conn.connect(sock_path)
    except Exception:
        conn.close()
        raise
    with conn:
        _send_msg(conn, req)
        return _recv_msg(conn)


# --- health monitor ---


def _start_health_monitor(sock_path: str) -> socket.socket:
    """Open a persistent health connection and start a daemon pong thread.

    The socket is returned and must be stored in session_state.  The pong thread
    holds only a weakref so the socket's lifetime — and therefore the server's
    lifetime — is tied to the session: when the tab is closed and the session is
    garbage-collected, the socket closes and the server detects EOF within one
    heartbeat interval and calls SIGTERM on itself.
    """
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)
    _send_msg(conn, {"type": "health"})
    conn.settimeout(2.0)  # Short timeout so we can periodically check the weakref

    conn_ref = weakref.ref(conn)

    def _pong_loop() -> None:
        while True:
            c = conn_ref()
            if c is None:
                return  # Session GC'd the socket — server will detect EOF
            try:
                msg = _recv_msg(c)
                timed_out = False
            except TimeoutError:
                timed_out = True
                msg = None
            except OSError:
                return  # Connection gone
            c = None  # Release strong ref so GC can close the socket when needed

            if timed_out:
                continue  # No ping yet; loop back and re-check weakref
            if msg is None:
                return  # Server closed connection (server exited)
            if msg.get("type") != "ping":
                continue

            c = conn_ref()
            if c is None:
                return
            try:
                _send_msg(c, {"type": "pong"})
            except OSError:
                return
            c = None  # Release

    threading.Thread(target=_pong_loop, daemon=True).start()
    return conn


# --- server lifecycle ---


def _server_cfg_key(
    checkpoint: str,
    upscale_model: str,
    face_model: str,
    lora: str,
    lora_rank: int,
    precision: str,
    backend: str,
    compile_unet: bool,
    trt_resolution: str,
) -> tuple:
    return (
        checkpoint,
        upscale_model,
        face_model,
        lora,
        lora_rank,
        precision,
        backend,
        compile_unet,
        trt_resolution,
    )


def _server_alive() -> bool:
    proc = st.session_state.get("_server_proc")
    return proc is not None and proc.poll() is None


def _kill_server() -> None:
    # Close health connection first so the server can begin its own cleanup.
    health_conn = st.session_state.pop("_health_conn", None)
    if health_conn is not None:
        try:
            health_conn.close()
        except OSError:
            pass
    proc = st.session_state.pop("_server_proc", None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc in _managed_procs:
            _managed_procs.remove(proc)
    sock = st.session_state.get("_sock_path", "")
    if sock:
        Path(sock).unlink(missing_ok=True)
    st.session_state.pop("_server_cfg", None)


def _launch_server(
    sock_path: str,
    checkpoint: str,
    upscale_model: str | None,
    face_model: str | None,
    lora: str | None,
    lora_rank: int,
    precision: str,
    backend: str,
    compile_unet: bool,
    trt_resolution: str,
    engine_dir: Path,
    onnx_dir: Path,
    upscale_tile_size: int = 512,
    upscale_overlap: int = 32,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "lora_trainer.appsrv",
        "--checkpoint",
        checkpoint,
        "--precision",
        precision,
        "--backend",
        backend,
        "--socket",
        sock_path,
        "--engine_dir",
        str(engine_dir),
        "--onnx_dir",
        str(onnx_dir),
        "--upscale_tile_size",
        str(upscale_tile_size),
        "--upscale_overlap",
        str(upscale_overlap),
    ]
    if upscale_model:
        cmd += ["--upscale_model", upscale_model]
    if face_model:
        cmd += ["--face_model", face_model]
    if lora:
        cmd += ["--lora_checkpoint", lora, "--lora_rank", str(lora_rank)]
    if compile_unet:
        cmd.append("--compile_unet")
    if backend == "trt":
        cmd += ["--resolution", trt_resolution]
    proc = subprocess.Popen(cmd)
    _managed_procs.append(proc)
    return proc


def _wait_for_socket(
    sock_path: str, timeout: int = 300, proc: subprocess.Popen | None = None
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # server crashed during startup
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(1.0)
            conn.connect(sock_path)
            conn.close()
            return True
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


# --- startup arg parsing ---


def _parse_startup_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--upscale_model_dir", type=Path, default=None)
    p.add_argument("--face_model_dir", type=Path, default=None)
    p.add_argument("--lora_dir", type=Path, default=None)
    p.add_argument("--engine_dir", type=Path, default=Path("engines"))
    p.add_argument("--onnx_dir", type=Path, default=Path("engines/onnx"))
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


# --- main ---


def main() -> None:
    st.set_page_config(page_title="SDXL Inference", layout="wide")
    startup = _parse_startup_args()

    checkpoints = _scan(startup.checkpoint_dir, "*.safetensors")
    upscale_models = _scan(startup.upscale_model_dir, "*.safetensors", "*.pth")
    face_models = _scan(startup.face_model_dir, "*.pt")
    loras = _scan(startup.lora_dir, "*.safetensors")

    if not checkpoints:
        st.error(f"No .safetensors checkpoints found in {startup.checkpoint_dir}")
        st.stop()

    checkpoint_paths = [str(p) for p in checkpoints]
    upscale_paths = [_NO_MODEL] + [str(p) for p in upscale_models]
    face_paths = [_NO_MODEL] + [str(p) for p in face_models]
    lora_paths = [_NO_MODEL] + [str(p) for p in loras]
    has_upscalers = len(upscale_models) > 0

    # Per-session paths — stable across reruns
    if "_sock_path" not in st.session_state:
        sid = uuid.uuid4().hex[:8]
        st.session_state["_sock_path"] = f"/tmp/lora_app_{sid}.sock"
        st.session_state["_out_path"] = Path(f"/tmp/lora_app_{sid}.png")
        st.session_state["_out_first_path"] = Path(f"/tmp/lora_app_{sid}_first.png")

    settings = _settings()
    if "pending_seed" in st.session_state:
        settings.seed.seed = int(st.session_state.pop("pending_seed"))

    resolution_options = list(SDXL_RESOLUTIONS)
    settings.resolution = _valid_choice(settings.resolution, resolution_options, "1024x1024")
    settings.hires.hires_resolution = _valid_choice(
        settings.hires.hires_resolution, resolution_options, settings.resolution
    )
    settings.sampling.sampler = _valid_choice(settings.sampling.sampler, SAMPLERS, "euler")
    settings.sampling.scheduler = _valid_choice(settings.sampling.scheduler, SCHEDULERS, "normal")
    settings.backend.precision = _valid_choice(
        settings.backend.precision, ["fp16", "bf16", "fp32"], "fp16"
    )
    settings.backend.backend = _valid_choice(settings.backend.backend, ["torch", "trt"], "torch")
    settings.models.checkpoint = _valid_choice(
        settings.models.checkpoint, checkpoint_paths, checkpoint_paths[0]
    )
    settings.models.upscale_model = _valid_choice(
        settings.models.upscale_model, upscale_paths, _NO_MODEL
    )
    settings.models.face_model = _valid_choice(settings.models.face_model, face_paths, _NO_MODEL)
    settings.models.lora = _valid_choice(settings.models.lora, lora_paths, _NO_MODEL)
    if settings.models.upscale_model == _NO_MODEL:
        settings.mode = "simple"

    # --- sidebar ---
    with st.sidebar:
        st.header("Model")
        settings.models.checkpoint = st.selectbox(
            "Checkpoint",
            checkpoint_paths,
            format_func=lambda p: Path(p).stem,
            index=_choice_index(checkpoint_paths, settings.models.checkpoint),
            key="ui.models.checkpoint",
        )
        settings.models.upscale_model = st.selectbox(
            "Upscale model",
            upscale_paths,
            format_func=lambda p: _NO_MODEL if p == _NO_MODEL else Path(p).stem,
            index=_choice_index(upscale_paths, settings.models.upscale_model),
            key="ui.models.upscale_model",
            disabled=not has_upscalers,
            help=(
                "Spandrel upscale model for hires-fix. Add --upscale_model_dir to enable."
                if not has_upscalers
                else None
            ),
        )
        upscale_model = (
            None if settings.models.upscale_model == _NO_MODEL else settings.models.upscale_model
        )

        settings.models.face_model = st.selectbox(
            "Face model (ADetailer)",
            face_paths,
            format_func=lambda p: _NO_MODEL if p == _NO_MODEL else Path(p).stem,
            index=_choice_index(face_paths, settings.models.face_model),
            key="ui.models.face_model",
            disabled=not face_models,
            help=(
                "YOLO face model for ADetailer. Add --face_model_dir to enable."
                if not face_models
                else None
            ),
        )
        face_model = None if settings.models.face_model == _NO_MODEL else settings.models.face_model

        settings.models.lora = st.selectbox(
            "LoRA",
            lora_paths,
            format_func=lambda p: _NO_MODEL if p == _NO_MODEL else Path(p).stem,
            index=_choice_index(lora_paths, settings.models.lora),
            key="ui.models.lora",
        )
        lora = None if settings.models.lora == _NO_MODEL else settings.models.lora
        settings.models.lora_rank = int(
            st.number_input(
                "LoRA rank",
                min_value=1,
                value=int(settings.models.lora_rank),
                key="ui.models.lora_rank",
                disabled=lora is None,
            )
        )

        # --- Hires-fix ---
        if has_upscalers:
            st.header("Hires-fix")
            hires_enabled = st.checkbox(
                "Enable hires-fix",
                value=settings.mode == "hires",
                key="ui.mode.hires_enabled",
                disabled=upscale_model is None,
                help="Runs SDXL → upscale → downscale → SDXL for sharper high-res results.",
            )
        else:
            hires_enabled = False
        settings.mode = "hires" if hires_enabled and upscale_model is not None else "simple"

        # --- Resolution ---
        st.header("Resolution")
        settings.resolution = st.selectbox(
            "Base resolution",
            resolution_options,
            index=_choice_index(resolution_options, settings.resolution),
            key="ui.resolution",
        )
        resolution = settings.resolution
        if hires_enabled:
            settings.hires.hires_resolution = st.selectbox(
                "Hires resolution",
                resolution_options,
                index=_choice_index(resolution_options, settings.hires.hires_resolution),
                key="ui.hires.hires_resolution",
                help="Target resolution for the second SDXL pass.",
            )
        else:
            settings.hires.hires_resolution = _valid_choice(
                settings.hires.hires_resolution, resolution_options, resolution
            )
        hires_resolution = settings.hires.hires_resolution if hires_enabled else resolution

        # --- Sampling ---
        st.header("Sampling")
        settings.sampling.sampler = st.selectbox(
            "Sampler",
            SAMPLERS,
            index=_choice_index(SAMPLERS, settings.sampling.sampler),
            key="ui.sampling.sampler",
        )
        settings.sampling.scheduler = st.selectbox(
            "Scheduler",
            SCHEDULERS,
            index=_choice_index(SCHEDULERS, settings.sampling.scheduler),
            key="ui.sampling.scheduler",
        )
        settings.sampling.cfg = float(
            st.slider(
                "CFG",
                min_value=1.0,
                max_value=15.0,
                step=0.5,
                value=float(settings.sampling.cfg),
                key="ui.sampling.cfg",
            )
        )
        settings.sampling.clip_skip = int(
            st.number_input(
                "Clip skip",
                min_value=1,
                max_value=4,
                step=1,
                value=int(settings.sampling.clip_skip),
                key="ui.sampling.clip_skip",
            )
        )
        sampler = settings.sampling.sampler
        scheduler = settings.sampling.scheduler
        cfg = settings.sampling.cfg
        clip_skip = settings.sampling.clip_skip

        if hires_enabled:
            st.subheader("First pass")
            settings.hires.first_steps = int(
                st.slider(
                    "Steps",
                    1,
                    60,
                    value=int(settings.hires.first_steps),
                    key="ui.hires.first_steps",
                )
            )
            settings.hires.first_denoise = float(
                st.slider(
                    "Denoise",
                    0.0,
                    1.0,
                    step=0.05,
                    value=float(settings.hires.first_denoise),
                    key="ui.hires.first_denoise",
                )
            )
            st.subheader("Second pass")
            settings.hires.second_steps = int(
                st.slider(
                    "Steps",
                    1,
                    60,
                    value=int(settings.hires.second_steps),
                    key="ui.hires.second_steps",
                )
            )
            settings.hires.second_denoise = float(
                st.slider(
                    "Denoise",
                    0.0,
                    1.0,
                    step=0.05,
                    value=float(settings.hires.second_denoise),
                    key="ui.hires.second_denoise",
                )
            )
        else:
            settings.simple.steps = int(
                st.slider(
                    "Steps",
                    min_value=1,
                    max_value=50,
                    value=int(settings.simple.steps),
                    key="ui.simple.steps",
                )
            )
            settings.simple.denoise = float(
                st.slider(
                    "Denoise",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(settings.simple.denoise),
                    key="ui.simple.denoise",
                )
            )

        settings.seed.random_seed = st.checkbox(
            "Random seed",
            value=bool(settings.seed.random_seed),
            key="ui.seed.random_seed",
        )
        random_seed = settings.seed.random_seed
        settings.seed.seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                max_value=2**32 - 1,
                step=1,
                value=int(settings.seed.seed),
                key="ui.seed.seed",
                disabled=random_seed,
            )
        )

        if hires_enabled:
            settings.backend.backend = "torch"

        # --- Backend ---
        st.header("Backend")
        # TRT doesn't support hires (resolution-specific engines, two passes)
        settings.backend.backend = st.radio(
            "Backend",
            ["torch", "trt"],
            index=_choice_index(["torch", "trt"], settings.backend.backend),
            key="ui.backend.backend",
            disabled=hires_enabled,
            help="TRT is disabled in hires mode (engines are resolution-specific).",
        )
        if hires_enabled:
            settings.backend.backend = "torch"
        backend = settings.backend.backend

        settings.backend.precision = st.radio(
            "Precision",
            ["fp16", "bf16", "fp32"],
            index=_choice_index(["fp16", "bf16", "fp32"], settings.backend.precision),
            key="ui.backend.precision",
        )
        precision = settings.backend.precision
        settings.backend.compile_unet = st.checkbox(
            "Compile UNet",
            value=bool(settings.backend.compile_unet),
            key="ui.backend.compile_unet",
            disabled=backend != "torch",
            help="torch.compile(reduce-overhead). Adds ~30 s startup, saves ~1 s/gen after.",
        )
        compile_unet = settings.backend.compile_unet

        # --- ADetailer params (shown when face model is selected) ---
        if face_model:
            st.header("ADetailer")
            settings.adetailer.face_denoise = float(
                st.slider(
                    "Face denoise",
                    0.0,
                    1.0,
                    step=0.05,
                    value=float(settings.adetailer.face_denoise),
                    key="ui.adetailer.face_denoise",
                )
            )
            settings.adetailer.face_steps = int(
                st.slider(
                    "Face steps",
                    1,
                    40,
                    value=int(settings.adetailer.face_steps),
                    key="ui.adetailer.face_steps",
                )
            )
            settings.adetailer.face_size = int(
                st.number_input(
                    "Face size",
                    min_value=512,
                    max_value=1024,
                    step=64,
                    value=max(512, int(settings.adetailer.face_size)),
                    key="ui.adetailer.face_size",
                )
            )
            settings.adetailer.face_padding = int(
                st.number_input(
                    "Face padding",
                    min_value=0,
                    max_value=128,
                    step=8,
                    value=int(settings.adetailer.face_padding),
                    key="ui.adetailer.face_padding",
                )
            )
            settings.adetailer.face_feather = int(
                st.number_input(
                    "Face feather",
                    min_value=0,
                    max_value=64,
                    step=4,
                    value=int(settings.adetailer.face_feather),
                    key="ui.adetailer.face_feather",
                )
            )

        # --- Server status ---
        st.divider()
        trt_resolution = resolution if backend == "trt" else ""
        cfg_key = _server_cfg_key(
            settings.models.checkpoint,
            upscale_model or "",
            face_model or "",
            lora or "",
            int(settings.models.lora_rank),
            precision,
            backend,
            bool(compile_unet),
            trt_resolution,
        )
        server_cfg_changed = st.session_state.get("_server_cfg") != cfg_key
        if server_cfg_changed and _server_alive():
            _kill_server()
        if not _server_alive():
            st.caption("⚪ Server not started")
        else:
            st.caption("🟢 Server ready")

        if _server_alive():
            if st.button("Stop server", width="stretch"):
                _kill_server()
                st.rerun()
        st.button("Reset to defaults", width="stretch", on_click=_reset_defaults)

    # --- main area ---
    st.title("SDXL Inference")
    settings.prompt.prompt = st.text_area(
        "Prompt",
        height=80,
        value=settings.prompt.prompt,
        key="ui.prompt.prompt",
    )
    prompt = settings.prompt.prompt
    settings.prompt.negative = st.text_area(
        "Negative prompt",
        height=60,
        value=settings.prompt.negative,
        key="ui.prompt.negative",
    )
    generate = st.button("Generate", type="primary", width="stretch")

    _save_settings(settings)

    if not generate:
        st.stop()

    if not prompt.strip():
        st.warning("Enter a prompt first.")
        st.stop()

    if hires_enabled and upscale_model is None:
        st.error(
            "Hires-fix is enabled but no upscale model is selected. Choose one in the sidebar."
        )
        st.stop()

    seed = int(settings.seed.seed)
    if random_seed:
        seed = random.randint(0, 2**32 - 1)
        st.session_state["pending_seed"] = seed

    sock_path = st.session_state["_sock_path"]
    out_path = st.session_state["_out_path"]
    out_first_path = st.session_state["_out_first_path"]
    if not _server_alive():
        with st.status("Starting server…", expanded=True) as srv_status:
            st.write(f"Loading: {Path(settings.models.checkpoint).stem}")
            if compile_unet:
                st.write("torch.compile warm-up — this takes ~30 s once")
            if backend == "trt":
                st.write(f"TRT engine: {resolution} ({precision})")
            proc = _launch_server(
                sock_path=sock_path,
                checkpoint=settings.models.checkpoint,
                upscale_model=upscale_model,
                face_model=face_model,
                lora=lora,
                lora_rank=int(settings.models.lora_rank),
                precision=precision,
                backend=backend,
                compile_unet=bool(compile_unet),
                trt_resolution=resolution,
                engine_dir=startup.engine_dir,
                onnx_dir=startup.onnx_dir,
            )
            st.session_state["_server_proc"] = proc
            if not _wait_for_socket(sock_path, timeout=300, proc=proc):
                _kill_server()
                srv_status.update(label="Server failed to start", state="error")
                if proc.poll() is not None:
                    st.error(
                        f"Server crashed on startup (exit {proc.poll()}). "
                        "Check the terminal for the error message."
                    )
                else:
                    st.error("Server did not become ready within 5 minutes.")
                st.stop()
            st.session_state["_server_cfg"] = cfg_key
            srv_status.update(label="Server ready", state="complete")

    # Establish health monitor if not already active (once per server instance).
    if "_health_conn" not in st.session_state:
        try:
            st.session_state["_health_conn"] = _start_health_monitor(sock_path)
        except Exception as exc:
            st.warning(f"Could not establish server health monitor: {exc}")

    # Build request
    req: dict = {
        "type": "hires" if hires_enabled else "simple",
        "prompt": prompt,
        "negative": settings.prompt.negative,
        "output": str(out_path),
        "resolution": resolution,
        "sampler": sampler,
        "scheduler": scheduler,
        "cfg": float(cfg),
        "clip_skip": int(clip_skip),
        "seed": str(seed),
        "enable_prompt_weighting": True,
        "progress": True,
        "face_denoise": float(settings.adetailer.face_denoise),
        "face_steps": int(settings.adetailer.face_steps),
        "face_size": int(settings.adetailer.face_size),
        "face_padding": int(settings.adetailer.face_padding),
        "face_feather": int(settings.adetailer.face_feather),
    }
    if hires_enabled:
        req["hires_resolution"] = hires_resolution
        req["save_first"] = str(out_first_path)
        req["first_steps"] = int(settings.hires.first_steps)
        req["first_denoise"] = float(settings.hires.first_denoise)
        req["second_steps"] = int(settings.hires.second_steps)
        req["second_denoise"] = float(settings.hires.second_denoise)
    else:
        req["steps"] = int(settings.simple.steps)
        req["denoise"] = float(settings.simple.denoise)

    # Send request
    with st.status("Generating…", expanded=False) as gen_status:
        try:
            resp = _request(sock_path, req)
        except Exception as exc:
            gen_status.update(label="Request failed", state="error")
            st.error(f"Could not reach server: {exc}")
            st.stop()

        if resp is None or resp.get("status") == "error":
            gen_status.update(label="Generation failed", state="error")
            msg = resp.get("message", "Unknown error") if resp else "No response from server"
            st.error(msg)
            st.stop()

        elapsed = resp.get("elapsed", 0.0)
        gen_status.update(label=f"Done — {elapsed:.2f} s", state="complete")

    # Display result
    if hires_enabled:
        col_main, col_first = st.columns([3, 1])
        with col_main:
            st.image(Image.open(out_path), caption=prompt, width="stretch")
            with open(out_path, "rb") as f:
                st.download_button(
                    "Download",
                    f.read(),
                    file_name="output.png",
                    mime="image/png",
                    width="stretch",
                )
        with col_first:
            st.caption("First pass")
            if out_first_path.exists():
                st.image(Image.open(out_first_path), width="stretch")
    else:
        st.image(Image.open(out_path), caption=prompt, width="stretch")
        with open(out_path, "rb") as f:
            st.download_button(
                "Download",
                f.read(),
                file_name="output.png",
                mime="image/png",
                width="stretch",
            )

    # Metrics
    with st.expander("Run info", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Generation time", f"{elapsed:.2f} s")
            st.metric("Seed", str(seed))
            if hires_enabled:
                st.metric(
                    "Passes",
                    f"{settings.hires.first_steps} + {settings.hires.second_steps} steps",
                )
            else:
                st.metric("Steps", str(settings.simple.steps))
        with c2:
            st.metric(
                "Mode",
                f"hires ({resolution} → {hires_resolution})"
                if hires_enabled
                else f"simple ({resolution})",
            )
            st.metric("Backend", f"{backend} / {precision}")
            if resp.get("peak_vram_gb"):
                st.metric("Peak VRAM", f"{resp['peak_vram_gb']:.2f} GB")
            if resp.get("reserved_vram_gb"):
                st.metric("Reserved VRAM", f"{resp['reserved_vram_gb']:.2f} GB")


def run() -> None:
    """Console-script entry point: wraps `streamlit run`."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--"] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
