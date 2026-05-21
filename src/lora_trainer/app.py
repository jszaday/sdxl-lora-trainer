"""Streamlit inference app — frontloaded SDXL pipeline with hot-start generations."""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import streamlit as st
import torch

from lora_trainer.model import clear_single_file_cache
from lora_trainer.pipeline import (
    PRECISIONS as _PRECISIONS,
)
from lora_trainer.pipeline import (
    SAMPLERS as _SAMPLERS,
)
from lora_trainer.pipeline import (
    SCHEDULERS as _SCHEDULERS,
)
from lora_trainer.pipeline import (
    dtype_from_precision,
    load_inference_models,
    load_torch_unet_backend,
    prepare_trt_engine,
)
from lora_trainer.sampling import decode_latents, encode_prompts_for_sampling
from lora_trainer.trt.backends import TensorRTUnavailableError, TensorRTUnetBackend
from lora_trainer.trt.config import SDXL_RESOLUTIONS, parse_resolution
from lora_trainer.trt.inference import sample_frozen_sdxl
from lora_trainer.utils import set_seed

_NO_LORA = "— None —"


def _scan_safetensors(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.safetensors"))


def _parse_startup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory to scan for SDXL .safetensors checkpoints.",
    )
    parser.add_argument("--lora_dir", type=Path, default=None)
    parser.add_argument("--engine_dir", type=Path, default=Path("engines"))
    parser.add_argument("--onnx_dir", type=Path, default=Path("engines/onnx"))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


# --- cached pipeline components -------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_pipeline(checkpoint: str, precision: str, lora_checkpoint: str | None, lora_rank: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_inference_models(
        checkpoint,
        device=device,
        dtype=dtype_from_precision(precision),
        lora_checkpoint=lora_checkpoint,
        lora_rank=lora_rank,
    )


@st.cache_resource(show_spinner=False)
def _load_trt_backend(engine_path: str) -> TensorRTUnetBackend:
    return TensorRTUnetBackend(Path(engine_path))


@st.cache_resource(show_spinner=False)
def _load_torch_backend(
    checkpoint: str, precision: str, lora_checkpoint: str | None, lora_rank: int
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_torch_unet_backend(
        checkpoint,
        device=device,
        dtype=dtype_from_precision(precision),
        lora_checkpoint=lora_checkpoint,
        lora_rank=lora_rank,
        flash_attention=True,
    )


# --- pipeline cache management --------------------------------------------------


def _pipeline_key(
    checkpoint: str,
    precision: str,
    lora_checkpoint: str | None,
    lora_rank: int,
    backend: str,
    resolution: str,
) -> tuple:
    return (checkpoint, precision, lora_checkpoint, lora_rank, backend, resolution)


def _evict_pipeline_cache() -> None:
    _load_pipeline.clear()
    _load_torch_backend.clear()
    _load_trt_backend.clear()
    clear_single_file_cache()
    gc.collect()
    torch.cuda.empty_cache()


# --- app ------------------------------------------------------------------------


_SETTINGS_PATH = Path.home() / ".config" / "lora-trainer" / "settings.json"

_DEFAULTS = {
    "resolution": "1216x832",
    "sampler": "euler",
    "scheduler": "karras",
    "cfg": 5.5,
    "steps": 30,
    "denoise": 1.0,
    "clip_skip": 1,
    "random_seed": True,
    "seed": 42,
    "backend": "torch",
    "precision": "fp16",
    "prompt": "",
    "negative": "",
}


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text())
    except Exception:
        return {}


def _save_settings(state: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    keys = (*_DEFAULTS, "checkpoint", "lora", "lora_rank")
    _SETTINGS_PATH.write_text(json.dumps({k: state[k] for k in keys if k in state}))


def main() -> None:
    st.set_page_config(page_title="SDXL Inference", layout="wide")
    startup = _parse_startup_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "pending_seed" in st.session_state:
        st.session_state["seed"] = st.session_state.pop("pending_seed")

    checkpoints = _scan_safetensors(startup.checkpoint_dir)
    if not checkpoints:
        st.error(f"No .safetensors checkpoints found in {startup.checkpoint_dir}")
        st.stop()

    loras = [_NO_LORA] + (
        [str(p) for p in _scan_safetensors(startup.lora_dir)] if startup.lora_dir else []
    )
    checkpoint_paths = [str(p) for p in checkpoints]

    if "settings_loaded" not in st.session_state:
        stored = _load_settings()
        for key, default in _DEFAULTS.items():
            st.session_state[key] = stored.get(key, default)
        st.session_state["checkpoint"] = (
            stored["checkpoint"]
            if stored.get("checkpoint") in checkpoint_paths
            else checkpoint_paths[0]
        )
        st.session_state["lora"] = stored["lora"] if stored.get("lora") in loras else _NO_LORA
        st.session_state["lora_rank"] = stored.get("lora_rank", 16)
        st.session_state["settings_loaded"] = True

    def _ensure_valid(key: str, options: list) -> None:
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[0]

    _ensure_valid("resolution", list(SDXL_RESOLUTIONS))
    _ensure_valid("sampler", _SAMPLERS)
    _ensure_valid("scheduler", _SCHEDULERS)
    _ensure_valid("backend", ["trt", "torch"])
    _ensure_valid("precision", _PRECISIONS)

    # --- sidebar ----------------------------------------------------------------
    with st.sidebar:
        st.header("Model")
        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoint_paths,
            format_func=lambda p: Path(p).stem,
            key="checkpoint",
        )
        lora_raw = st.selectbox(
            "LoRA",
            loras,
            format_func=lambda p: _NO_LORA if p == _NO_LORA else Path(p).stem,
            key="lora",
        )
        lora_checkpoint = None if lora_raw == _NO_LORA else lora_raw
        lora_rank = st.number_input(
            "LoRA rank", min_value=1, key="lora_rank", disabled=lora_checkpoint is None
        )

        st.header("Sampling")
        resolution = st.selectbox("Resolution", list(SDXL_RESOLUTIONS), key="resolution")
        sampler = st.selectbox("Sampler", _SAMPLERS, key="sampler")
        scheduler = st.selectbox("Scheduler", _SCHEDULERS, key="scheduler")
        cfg = st.slider("CFG", min_value=1.0, max_value=15.0, step=0.5, key="cfg")
        steps = st.slider("Steps", min_value=1, max_value=50, key="steps")
        denoise = st.slider("Denoise", min_value=0.0, max_value=1.0, step=0.05, key="denoise")
        clip_skip = st.number_input("Clip skip", min_value=1, max_value=4, step=1, key="clip_skip")
        random_seed = st.checkbox("Random seed", key="random_seed")
        st.number_input(
            "Seed",
            min_value=0,
            max_value=2**32 - 1,
            step=1,
            key="seed",
            disabled=random_seed,
        )

        st.header("Backend")
        backend = st.radio("Backend", ["trt", "torch"], key="backend")
        precision = st.radio("Precision", _PRECISIONS, key="precision")

    _save_settings(st.session_state)

    # --- main area --------------------------------------------------------------
    st.title("SDXL Inference")
    prompt = st.text_area("Prompt", height=80, key="prompt")
    negative = st.text_area("Negative prompt", height=60, key="negative")
    generate = st.button("Generate", type="primary", width="stretch")

    if not generate:
        st.stop()

    if not prompt.strip():
        st.warning("Enter a prompt first.")
        st.stop()

    seed = int(st.session_state["seed"])
    set_seed(seed)
    res = parse_resolution(resolution)

    current_key = _pipeline_key(
        checkpoint, precision, lora_checkpoint, int(lora_rank), backend, resolution
    )
    if st.session_state.get("loaded_pipeline_key") != current_key:
        _evict_pipeline_cache()

    with st.status("Loading pipeline...", expanded=False) as status:
        st.write("VAE + text encoders…")
        vae, te1, te2, tok1, tok2 = _load_pipeline(
            checkpoint, precision, lora_checkpoint, int(lora_rank)
        )

        st.write("UNet backend…")
        if backend == "trt":
            st.write("TensorRT engine…")
            try:
                engine_path = prepare_trt_engine(
                    checkpoint,
                    res,
                    engine_dir=startup.engine_dir,
                    onnx_dir=startup.onnx_dir,
                    precision=precision,
                    device=device,
                    lora_checkpoint=Path(lora_checkpoint) if lora_checkpoint else None,
                    lora_rank=int(lora_rank),
                )
                unet_backend = _load_trt_backend(str(engine_path))
            except TensorRTUnavailableError as exc:
                st.error(str(exc))
                st.stop()
        else:
            unet_backend = _load_torch_backend(
                checkpoint, precision, lora_checkpoint, int(lora_rank)
            )

        status.update(label="Pipeline ready", state="complete")
    st.session_state["loaded_pipeline_key"] = current_key

    dtype = dtype_from_precision(precision)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0_encode = time.perf_counter()
    with torch.inference_mode():
        prompt_embeds, pooled = encode_prompts_for_sampling(
            [prompt],
            te1,
            te2,
            tok1,
            tok2,
            device,
            clip_skip=int(clip_skip),
            enable_prompt_weighting=True,
        )
        neg_embeds, pooled_neg = encode_prompts_for_sampling(
            [negative],
            te1,
            te2,
            tok1,
            tok2,
            device,
            clip_skip=int(clip_skip),
            enable_prompt_weighting=True,
        )
    t_encode = time.perf_counter() - t0_encode

    progress_bar = st.progress(0.0, text="Starting…")
    _step_timestamps: list[float] = []
    t0_sample = time.perf_counter()

    def on_step(current: int, total: int) -> None:
        _step_timestamps.append(time.perf_counter())
        progress_bar.progress(current / total, text=f"Step {current} / {total}")

    latents = sample_frozen_sdxl(
        unet_backend,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        pooled_prompt_embeds=pooled,
        pooled_negative_prompt_embeds=pooled_neg,
        resolution=res,
        sampler=sampler,
        scheduler_name=scheduler,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        device=device,
        dtype=dtype,
        seed=int(seed),
        denoise=float(denoise),
        progress=False,
        on_step=on_step,
    )
    t_sample = time.perf_counter() - t0_sample
    del prompt_embeds, neg_embeds, pooled, pooled_neg

    progress_bar.progress(1.0, text="Decoding…")
    t0_decode = time.perf_counter()
    images = decode_latents(vae, latents.to(torch.float32))
    t_decode = time.perf_counter() - t0_decode
    del latents
    progress_bar.empty()

    image_np = images[0].cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).round().clip(0, 255).astype("uint8")
    del images
    gc.collect()
    torch.cuda.empty_cache()

    if random_seed:
        st.session_state["pending_seed"] = random.randint(0, 2**32 - 1)

    st.image(image_np, caption=prompt, width="stretch")

    # --- performance metrics ----------------------------------------------------
    if _step_timestamps:
        prev = t0_sample
        step_ms = []
        for ts in _step_timestamps:
            step_ms.append((ts - prev) * 1000)
            prev = ts
        avg_step_ms = sum(step_ms) / len(step_ms)
        steps_per_sec = 1000.0 / avg_step_ms if avg_step_ms > 0 else 0.0
    else:
        avg_step_ms = steps_per_sec = 0.0

    if device == "cuda":
        vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        vram_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    else:
        vram_peak_gb = vram_reserved_gb = 0.0

    t_total = t_encode + t_sample + t_decode

    with st.expander("Performance", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total time", f"{t_total:.2f}s")
            st.metric("Encoding", f"{t_encode * 1000:.0f}ms")
            st.metric("Sampling", f"{t_sample:.2f}s")
            st.metric("Decoding", f"{t_decode * 1000:.0f}ms")
        with c2:
            st.metric("Steps/sec", f"{steps_per_sec:.2f}")
            st.metric("Avg step", f"{avg_step_ms:.0f}ms")
            if step_ms:
                st.metric("Fastest step", f"{min(step_ms):.0f}ms")
                st.metric("Slowest step", f"{max(step_ms):.0f}ms")
        with c3:
            gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
            st.metric("GPU", gpu_name)
            st.metric("Backend", backend)
            st.metric("Precision", precision)
            if device == "cuda":
                st.metric("Peak VRAM", f"{vram_peak_gb:.2f} GB")
                st.metric("Reserved VRAM", f"{vram_reserved_gb:.2f} GB")


def run() -> None:
    """Console-script entry point: wraps `streamlit run`."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--"] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
