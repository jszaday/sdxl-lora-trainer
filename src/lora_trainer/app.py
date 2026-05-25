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
import time
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

from lora_trainer.pipeline import SAMPLERS, SCHEDULERS
from lora_trainer.trt.config import SDXL_RESOLUTIONS

_NO_MODEL = "— None —"
_LEN_FMT = ">I"
_SETTINGS_PATH = Path.home() / ".config" / "lora-trainer" / "app_settings.json"

_DEFAULTS: dict = {
    # Resolution
    "resolution": "1024x1024",
    "hires_resolution": "1024x1024",
    # Sampling
    "sampler": "euler_ancestral",
    "scheduler": "normal",
    "cfg": 7.0,
    "clip_skip": 1,
    # Simple mode
    "steps": 25,
    "denoise": 1.0,
    # Hires mode
    "hires_enabled": False,
    "first_steps": 25,
    "first_denoise": 1.0,
    "second_steps": 25,
    "second_denoise": 0.7,
    # Seed
    "random_seed": True,
    "seed": 42,
    # Backend
    "backend": "torch",
    "precision": "fp16",
    "compile_unet": False,
    # ADetailer
    "face_denoise": 0.4,
    "face_steps": 20,
    "face_size": 512,
    "face_padding": 32,
    "face_feather": 16,
    # Prompts
    "prompt": "",
    "negative": "",
}
_FIELD_KEYS = tuple(_DEFAULTS)

_managed_procs: list[subprocess.Popen] = []


@atexit.register
def _cleanup_procs() -> None:
    for proc in _managed_procs:
        if proc.poll() is None:
            proc.terminate()


# --- settings ---


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text())
    except Exception:
        return {}


def _field_value(key: str) -> object:
    values = st.session_state.get("_field_values", {})
    return st.session_state.get(key, values.get(key, _DEFAULTS[key]))


def _set_field_value(key: str, value: object) -> None:
    st.session_state[key] = value
    st.session_state.setdefault("_field_values", {})[key] = value


def _sync_field_values() -> None:
    values = st.session_state.setdefault("_field_values", {})
    for key in _FIELD_KEYS:
        if key in st.session_state:
            values[key] = st.session_state[key]


def _hydrate_field_keys() -> None:
    values = st.session_state.setdefault("_field_values", {})
    for key in _FIELD_KEYS:
        st.session_state.setdefault(key, values.get(key, _DEFAULTS[key]))


def _reset_defaults() -> None:
    for key, value in _DEFAULTS.items():
        _set_field_value(key, value)
    _SETTINGS_PATH.unlink(missing_ok=True)


def _save_settings(state: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    values = dict(state.get("_field_values", {}))
    values.update({k: state[k] for k in _FIELD_KEYS if k in state})
    keys = (
        *_DEFAULTS,
        "checkpoint",
        "upscale_model",
        "face_model",
        "lora",
        "lora_rank",
    )
    _SETTINGS_PATH.write_text(
        json.dumps({k: state.get(k, values.get(k)) for k in keys if k in state or k in values})
    )


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


def _request(sock_path: str, req: dict) -> dict | None:
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)
    with conn:
        _send_msg(conn, req)
        return _recv_msg(conn)


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


def _wait_for_socket(sock_path: str, timeout: int = 300) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.settimeout(1.0)
            conn.connect(sock_path)
            conn.close()
            return True
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def _sync_hires_res() -> None:
    """When base resolution changes, keep hires_resolution in sync if they were equal."""
    prev = st.session_state.get("_prev_resolution")
    new = st.session_state.get("resolution", _DEFAULTS["resolution"])
    if st.session_state.get("hires_resolution") == prev:
        _set_field_value("hires_resolution", new)
    _set_field_value("resolution", new)
    st.session_state["_prev_resolution"] = new


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

    if "pending_seed" in st.session_state:
        _set_field_value("seed", st.session_state.pop("pending_seed"))

    # Load settings once per browser session
    if "settings_loaded" not in st.session_state:
        stored = _load_settings()
        st.session_state["_field_values"] = {k: stored.get(k, v) for k, v in _DEFAULTS.items()}
        for k in _DEFAULTS:
            st.session_state[k] = st.session_state["_field_values"][k]
        st.session_state["checkpoint"] = (
            stored.get("checkpoint")
            if stored.get("checkpoint") in checkpoint_paths
            else checkpoint_paths[0]
        )
        st.session_state["upscale_model"] = (
            stored.get("upscale_model")
            if stored.get("upscale_model") in upscale_paths
            else _NO_MODEL
        )
        st.session_state["face_model"] = (
            stored.get("face_model") if stored.get("face_model") in face_paths else _NO_MODEL
        )
        st.session_state["lora"] = (
            stored.get("lora") if stored.get("lora") in lora_paths else _NO_MODEL
        )
        st.session_state["lora_rank"] = stored.get("lora_rank", 16)
        st.session_state["_prev_resolution"] = _field_value("resolution")
        st.session_state["_prev_hires_enabled"] = bool(_field_value("hires_enabled"))
        st.session_state["settings_loaded"] = True
    else:
        _hydrate_field_keys()

    def _ensure_valid(key: str, options: list) -> None:
        if _field_value(key) not in options:
            _set_field_value(key, options[0])

    _ensure_valid("resolution", list(SDXL_RESOLUTIONS))
    _ensure_valid("hires_resolution", list(SDXL_RESOLUTIONS))
    _ensure_valid("sampler", SAMPLERS)
    _ensure_valid("scheduler", SCHEDULERS)
    _ensure_valid("precision", ["fp16", "bf16", "fp32"])
    _ensure_valid("backend", ["torch", "trt"])

    # --- sidebar ---
    with st.sidebar:
        st.header("Model")
        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoint_paths,
            format_func=lambda p: Path(p).stem,
            key="checkpoint",
        )
        upscale_model_raw = st.selectbox(
            "Upscale model",
            upscale_paths,
            format_func=lambda p: _NO_MODEL if p == _NO_MODEL else Path(p).stem,
            key="upscale_model",
            disabled=not has_upscalers,
            help=(
                "Spandrel upscale model for hires-fix. Add --upscale_model_dir to enable."
                if not has_upscalers
                else None
            ),
        )
        upscale_model = None if upscale_model_raw == _NO_MODEL else upscale_model_raw

        face_model_raw = st.selectbox(
            "Face model (ADetailer)",
            face_paths,
            format_func=lambda p: _NO_MODEL if p == _NO_MODEL else Path(p).stem,
            key="face_model",
            disabled=not face_models,
            help=(
                "YOLO face model for ADetailer. Add --face_model_dir to enable."
                if not face_models
                else None
            ),
        )
        face_model = None if face_model_raw == _NO_MODEL else face_model_raw

        lora_raw = st.selectbox(
            "LoRA",
            lora_paths,
            format_func=lambda p: _NO_MODEL if p == _NO_MODEL else Path(p).stem,
            key="lora",
        )
        lora = None if lora_raw == _NO_MODEL else lora_raw
        lora_rank = st.number_input(
            "LoRA rank", min_value=1, key="lora_rank", disabled=lora is None
        )

        # --- Hires-fix ---
        if has_upscalers:
            st.header("Hires-fix")
            if upscale_model is None and _field_value("hires_enabled"):
                _set_field_value("hires_enabled", False)
            hires_enabled = st.checkbox(
                "Enable hires-fix",
                key="hires_enabled",
                disabled=upscale_model is None,
                help="Runs SDXL → upscale → downscale → SDXL for sharper high-res results.",
            )
        else:
            hires_enabled = False
            _set_field_value("hires_enabled", False)

        # --- Resolution ---
        st.header("Resolution")
        resolution = st.selectbox(
            "Base resolution",
            list(SDXL_RESOLUTIONS),
            key="resolution",
            on_change=_sync_hires_res,
        )
        if hires_enabled and not st.session_state.get("_prev_hires_enabled", False):
            _set_field_value("hires_resolution", resolution)
        if hires_enabled:
            hires_resolution = st.selectbox(
                "Hires resolution",
                list(SDXL_RESOLUTIONS),
                key="hires_resolution",
                help="Target resolution for the second SDXL pass. Tracks base until changed.",
            )
        else:
            hires_resolution = resolution

        # --- Sampling ---
        st.header("Sampling")
        sampler = st.selectbox("Sampler", SAMPLERS, key="sampler")
        scheduler = st.selectbox("Scheduler", SCHEDULERS, key="scheduler")
        cfg = st.slider("CFG", min_value=1.0, max_value=15.0, step=0.5, key="cfg")
        clip_skip = st.number_input("Clip skip", min_value=1, max_value=4, step=1, key="clip_skip")

        if hires_enabled:
            st.subheader("First pass")
            st.slider("Steps", 1, 60, key="first_steps")
            st.slider(
                "Denoise",
                0.0,
                1.0,
                step=0.05,
                key="first_denoise",
            )
            st.subheader("Second pass")
            st.slider("Steps", 1, 60, key="second_steps")
            st.slider(
                "Denoise",
                0.0,
                1.0,
                step=0.05,
                key="second_denoise",
            )
        else:
            st.slider("Steps", min_value=1, max_value=50, key="steps")
            st.slider("Denoise", min_value=0.0, max_value=1.0, step=0.05, key="denoise")

        random_seed = st.checkbox("Random seed", key="random_seed")
        st.number_input(
            "Seed",
            min_value=0,
            max_value=2**32 - 1,
            step=1,
            key="seed",
            disabled=random_seed,
        )

        if hires_enabled:
            _set_field_value("backend", "torch")

        # --- Backend ---
        st.header("Backend")
        # TRT doesn't support hires (resolution-specific engines, two passes)
        backend = st.radio(
            "Backend",
            ["torch", "trt"],
            key="backend",
            disabled=hires_enabled,
            help="TRT is disabled in hires mode (engines are resolution-specific).",
        )
        if hires_enabled:
            backend = "torch"

        precision = st.radio("Precision", ["fp16", "bf16", "fp32"], key="precision")
        compile_unet = st.checkbox(
            "Compile UNet",
            key="compile_unet",
            disabled=backend != "torch",
            help="torch.compile(reduce-overhead). Adds ~30 s startup, saves ~1 s/gen after.",
        )

        # --- ADetailer params (shown when face model is selected) ---
        if face_model:
            st.header("ADetailer")
            st.slider(
                "Face denoise",
                0.0,
                1.0,
                step=0.05,
                key="face_denoise",
            )
            st.slider("Face steps", 1, 40, key="face_steps")
            st.number_input(
                "Face size",
                min_value=256,
                max_value=1024,
                step=64,
                key="face_size",
            )
            st.number_input(
                "Face padding",
                min_value=0,
                max_value=128,
                step=8,
                key="face_padding",
            )
            st.number_input(
                "Face feather",
                min_value=0,
                max_value=64,
                step=4,
                key="face_feather",
            )

        # --- Server status ---
        st.divider()
        trt_resolution = resolution if backend == "trt" else ""
        cfg_key = _server_cfg_key(
            checkpoint,
            upscale_model or "",
            face_model or "",
            lora or "",
            int(lora_rank),
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
    prompt = st.text_area("Prompt", height=80, key="prompt")
    st.text_area("Negative prompt", height=60, key="negative")
    generate = st.button("Generate", type="primary", width="stretch")

    st.session_state["_prev_hires_enabled"] = bool(hires_enabled)
    _sync_field_values()
    _save_settings(st.session_state)

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

    seed = int(_field_value("seed"))
    if random_seed:
        seed = random.randint(0, 2**32 - 1)
        st.session_state["pending_seed"] = seed

    sock_path = st.session_state["_sock_path"]
    out_path = st.session_state["_out_path"]
    out_first_path = st.session_state["_out_first_path"]
    if not _server_alive():
        with st.status("Starting server…", expanded=True) as srv_status:
            st.write(f"Loading: {Path(checkpoint).stem}")
            if compile_unet:
                st.write("torch.compile warm-up — this takes ~30 s once")
            if backend == "trt":
                st.write(f"TRT engine: {resolution} ({precision})")
            proc = _launch_server(
                sock_path=sock_path,
                checkpoint=checkpoint,
                upscale_model=upscale_model,
                face_model=face_model,
                lora=lora,
                lora_rank=int(lora_rank),
                precision=precision,
                backend=backend,
                compile_unet=bool(compile_unet),
                trt_resolution=resolution,
                engine_dir=startup.engine_dir,
                onnx_dir=startup.onnx_dir,
            )
            st.session_state["_server_proc"] = proc
            if not _wait_for_socket(sock_path, timeout=300):
                _kill_server()
                srv_status.update(label="Server failed to start", state="error")
                st.error("Server did not become ready within 5 minutes.")
                st.stop()
            st.session_state["_server_cfg"] = cfg_key
            srv_status.update(label="Server ready", state="complete")

    # Build request
    req: dict = {
        "type": "hires" if hires_enabled else "simple",
        "prompt": prompt,
        "negative": str(_field_value("negative")),
        "output": str(out_path),
        "resolution": resolution,
        "sampler": sampler,
        "scheduler": scheduler,
        "cfg": float(cfg),
        "clip_skip": int(clip_skip),
        "seed": str(seed),
        "enable_prompt_weighting": True,
        "progress": True,
        "face_denoise": float(_field_value("face_denoise")),
        "face_steps": int(_field_value("face_steps")),
        "face_size": int(_field_value("face_size")),
        "face_padding": int(_field_value("face_padding")),
        "face_feather": int(_field_value("face_feather")),
    }
    if hires_enabled:
        req["hires_resolution"] = hires_resolution
        req["save_first"] = str(out_first_path)
        req["first_steps"] = int(_field_value("first_steps"))
        req["first_denoise"] = float(_field_value("first_denoise"))
        req["second_steps"] = int(_field_value("second_steps"))
        req["second_denoise"] = float(_field_value("second_denoise"))
    else:
        req["steps"] = int(_field_value("steps"))
        req["denoise"] = float(_field_value("denoise"))

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
                    f"{_field_value('first_steps')} + {_field_value('second_steps')} steps",
                )
            else:
                st.metric("Steps", str(_field_value("steps")))
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
