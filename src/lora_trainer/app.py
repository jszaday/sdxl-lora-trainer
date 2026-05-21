"""Streamlit inference app — frontloaded SDXL pipeline with hot-start generations."""

import argparse
import json
import random
import sys
from pathlib import Path

import streamlit as st
import torch

from lora_trainer.model import (
    load_lora_weights,
    load_sdxl_unet,
    load_text_encoders,
    load_vae,
    merge_lora_layers,
)
from lora_trainer.sampling import decode_latents, encode_prompts_for_sampling
from lora_trainer.trt.backends import (
    TensorRTUnavailableError,
    TensorRTUnetBackend,
    TorchUnetBackend,
)
from lora_trainer.trt.cache import build_engine_cache_key, resolve_engine_artifacts
from lora_trainer.trt.config import SDXL_RESOLUTIONS, parse_resolution
from lora_trainer.trt.inference import sample_frozen_sdxl
from lora_trainer.utils import set_seed

_SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "lms",
    "pndm",
    "ddim",
]
_SCHEDULERS = ["karras", "simple", "normal", "exponential", "sgm_uniform"]
_PRECISIONS = ["fp16", "bf16", "fp32"]
_NO_LORA = "— None —"


def _dtype(precision: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)


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
    parser.add_argument(
        "--lora_dir",
        type=Path,
        default=None,
        help="Directory to scan for LoRA .safetensors checkpoints.",
    )
    parser.add_argument(
        "--engine_dir",
        type=Path,
        default=Path("engines"),
        help="TensorRT engine directory (fixed for the session).",
    )
    parser.add_argument("--onnx_dir", type=Path, default=Path("engines/onnx"))
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


# --- cached pipeline components -------------------------------------------------
# Each function is keyed on its arguments; changing checkpoint or LoRA loads a
# fresh instance while the old one stays warm in the Streamlit resource cache.


@st.cache_resource(show_spinner=False)
def _load_pipeline(checkpoint: str, precision: str, lora_checkpoint: str | None, lora_rank: int):
    """Load VAE + text encoders in one shot to avoid loading the checkpoint twice."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _dtype(precision)
    rank = lora_rank if lora_checkpoint else None
    vae = load_vae(checkpoint, device=device, dtype=dtype)
    vae.to(torch.float32)
    te1, te2, tok1, tok2, _ = load_text_encoders(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=rank,
        adapter="lora",
    )
    if lora_checkpoint:
        load_lora_weights(lora_checkpoint, text_encoder_1=te1, text_encoder_2=te2)
        merge_lora_layers(te1)
        merge_lora_layers(te2)
    vae.requires_grad_(False).eval()
    te1.requires_grad_(False).eval()
    te2.requires_grad_(False).eval()
    return vae, te1, te2, tok1, tok2


@st.cache_resource(show_spinner=False)
def _load_trt_backend(engine_path: str) -> TensorRTUnetBackend:
    return TensorRTUnetBackend(Path(engine_path))


@st.cache_resource(show_spinner=False)
def _load_torch_backend(
    checkpoint: str, precision: str, lora_checkpoint: str | None, lora_rank: int
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _dtype(precision)
    rank = lora_rank if lora_checkpoint else None
    unet, _ = load_sdxl_unet(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=rank,
        adapter="lora",
    )
    if lora_checkpoint:
        load_lora_weights(lora_checkpoint, unet=unet)
        merge_lora_layers(unet)
    unet.requires_grad_(False).eval()
    return TorchUnetBackend(unet)


# --- app ------------------------------------------------------------------------


_SETTINGS_PATH = Path.home() / ".config" / "lora-trainer" / "settings.json"

_DEFAULTS = {
    "resolution": "1216x832",
    "sampler": "euler",
    "scheduler": "karras",
    "cfg": 5.5,
    "steps": 30,
    "random_seed": True,
    "seed": 42,
    "backend": "trt",
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
        # Restore model selectors with fallback to first available.
        st.session_state["checkpoint"] = (
            stored["checkpoint"]
            if stored.get("checkpoint") in checkpoint_paths
            else checkpoint_paths[0]
        )
        st.session_state["lora"] = stored["lora"] if stored.get("lora") in loras else _NO_LORA
        st.session_state["lora_rank"] = stored.get("lora_rank", 16)
        st.session_state["settings_loaded"] = True

    # Validate option-list keys in case available files changed since last save.
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

    with st.status("Loading pipeline...", expanded=False) as status:
        st.write("VAE + text encoders…")
        vae, te1, te2, tok1, tok2 = _load_pipeline(
            checkpoint, precision, lora_checkpoint, int(lora_rank)
        )

        st.write("UNet backend…")
        if backend == "trt":
            key = build_engine_cache_key(
                checkpoint,
                res,
                precision=precision,
                lora_checkpoint=Path(lora_checkpoint) if lora_checkpoint else None,
            )
            engine_path = resolve_engine_artifacts(
                startup.engine_dir, startup.onnx_dir, key
            ).engine_path
            try:
                unet_backend = _load_trt_backend(str(engine_path))
            except TensorRTUnavailableError as exc:
                st.error(str(exc))
                st.stop()
        else:
            unet_backend = _load_torch_backend(
                checkpoint, precision, lora_checkpoint, int(lora_rank)
            )

        status.update(label="Pipeline ready", state="complete")

    dtype = _dtype(precision)

    with torch.inference_mode():
        prompt_embeds, pooled = encode_prompts_for_sampling(
            [prompt],
            te1,
            te2,
            tok1,
            tok2,
            device,
            clip_skip=1,
            enable_prompt_weighting=True,
        )
        neg_embeds, pooled_neg = encode_prompts_for_sampling(
            [negative],
            te1,
            te2,
            tok1,
            tok2,
            device,
            clip_skip=1,
            enable_prompt_weighting=True,
        )

    progress_bar = st.progress(0.0, text="Starting…")

    def on_step(current: int, total: int) -> None:
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
        progress=False,
        on_step=on_step,
    )
    del prompt_embeds, neg_embeds, pooled, pooled_neg

    progress_bar.progress(1.0, text="Decoding…")
    images = decode_latents(vae, latents.to(torch.float32))
    del latents
    progress_bar.empty()

    image_np = images[0].cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).round().clip(0, 255).astype("uint8")
    del images
    torch.cuda.empty_cache()

    if random_seed:
        st.session_state["pending_seed"] = random.randint(0, 2**32 - 1)

    st.image(image_np, caption=prompt, width="stretch")


def run() -> None:
    """Console-script entry point: wraps `streamlit run`."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--"] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
