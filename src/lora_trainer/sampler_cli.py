"""Frozen SDXL inference entry point with TensorRT-ready shapes."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from .pipeline import (
    SAMPLERS as _SAMPLERS,
)
from .pipeline import (
    SCHEDULERS as _SCHEDULERS,
)
from .pipeline import (
    build_trt_backend,
    dtype_from_precision,
    load_inference_models,
    load_torch_unet_backend,
)
from .sampling import decode_latents, encode_prompts_for_sampling
from .trt.backends import TensorRTUnavailableError
from .trt.config import SDXL_RESOLUTIONS, infer_resolution_from_latents, parse_resolution
from .trt.inference import sample_frozen_sdxl
from .trt.latents import load_latents, save_latents
from .utils import set_seed


def _save_images(images: torch.Tensor, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        suffix = output.suffix or ".png"
        path = output if images.shape[0] == 1 else output.with_name(f"{output.stem}_{idx}{suffix}")
        image_np = image.to(torch.float32).cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).round().clip(0, 255).astype("uint8")
        Image.fromarray(image_np).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen SDXL inference with fixed TensorRT-ready resolutions",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Base SDXL checkpoint or HuggingFace model ID."
    )
    parser.add_argument("--prompt", required=True, help="Positive prompt.")
    parser.add_argument("--negative", default="", help="Negative prompt.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample.png"),
        help="Output image path. Batched outputs append _N before suffix.",
    )
    parser.add_argument(
        "--resolution",
        choices=list(SDXL_RESOLUTIONS),
        default="1024x1024",
        help="Fixed SDXL-native output resolution.",
    )
    parser.add_argument(
        "--latents", type=Path, default=None, help="Starting latents (.safetensors or .pt)."
    )
    parser.add_argument(
        "--save_latents", type=Path, default=None, help="Path to save final latents."
    )
    parser.add_argument(
        "--no_decode", action="store_true", help="Skip VAE decode; only save latents."
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "trt"],
        default="torch",
        help="UNet backend. Defaults to torch (flash attention); use trt for TensorRT.",
    )
    parser.add_argument("--engine_dir", type=Path, default=Path("engines"))
    parser.add_argument("--onnx_dir", type=Path, default=Path("engines/onnx"))
    parser.add_argument(
        "--no_build_engine",
        action="store_true",
        help="Fail instead of auto-building a missing TensorRT engine.",
    )
    parser.add_argument(
        "--force_build_engine",
        action="store_true",
        help="Re-export ONNX and rebuild the TRT engine before inference.",
    )
    parser.add_argument(
        "--workspace_gb", type=float, default=12.0, help="TensorRT builder workspace limit in GiB."
    )
    parser.add_argument(
        "--compile_unet",
        action="store_true",
        help="Apply torch.compile to the UNet for fixed-shape inference.",
    )
    parser.add_argument(
        "--no_flash_attention",
        dest="flash_attention",
        action="store_false",
        default=True,
        help="Disable flash attention for the torch backend.",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress.")
    parser.add_argument(
        "--scheduler", choices=_SCHEDULERS, default="karras", help="Noise schedule."
    )
    parser.add_argument("--sampler", choices=_SAMPLERS, default="euler", help="Sampler algorithm.")
    parser.add_argument("--cfg", type=float, default=5.5, help="Classifier-free guidance scale.")
    parser.add_argument("--sampler_steps", type=int, default=30, help="Number of denoising steps.")
    parser.add_argument(
        "--denoise",
        type=float,
        default=1.0,
        help="Fraction of denoise schedule to run (img2img from --latents).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device. Defaults to cuda when available.")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--lora_checkpoint", type=Path, default=None, help="LoRA checkpoint to merge."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="LoRA rank for --lora_checkpoint."
    )
    parser.add_argument("--clip_skip", type=int, default=1, help="CLIP skip for text_encoder_1.")
    parser.add_argument(
        "--disable_prompt_weighting",
        dest="enable_prompt_weighting",
        action="store_false",
        default=True,
        help="Treat parentheses literally instead of applying prompt weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision)
    set_seed(args.seed)

    initial_latents = None
    resolution = parse_resolution(args.resolution)
    if args.latents is not None:
        initial_latents = load_latents(args.latents, device=device, dtype=dtype)
        inferred = infer_resolution_from_latents(initial_latents)
        if args.resolution == "1024x1024" and inferred.name != args.resolution:
            resolution = inferred
        elif inferred != resolution:
            raise ValueError(
                f"Latents imply {inferred.name}, but --resolution is {resolution.name}"
            )

    print(f"Using device: {device}")
    print(
        f"Resolution: {resolution.name} "
        f"({resolution.latent_height}x{resolution.latent_width} latent)"
    )

    lora_ckpt = args.lora_checkpoint
    print(f"Loading SDXL components from: {args.checkpoint}")
    vae, te1, te2, tok1, tok2 = load_inference_models(
        args.checkpoint,
        device=device,
        dtype=dtype,
        lora_checkpoint=lora_ckpt,
        lora_rank=args.lora_rank,
    )

    if args.backend == "trt":
        unet_backend = build_trt_backend(
            args.checkpoint,
            resolution,
            engine_dir=args.engine_dir,
            onnx_dir=args.onnx_dir,
            precision=args.precision,
            device=device,
            lora_checkpoint=lora_ckpt,
            lora_rank=args.lora_rank,
            workspace_gb=args.workspace_gb,
            force=args.force_build_engine,
            no_build=args.no_build_engine,
        )
    else:
        unet_backend = load_torch_unet_backend(
            args.checkpoint,
            device=device,
            dtype=dtype,
            lora_checkpoint=lora_ckpt,
            lora_rank=args.lora_rank,
            compile_unet=args.compile_unet,
            flash_attention=args.flash_attention,
        )

    prompt_embeds, pooled_prompt_embeds = encode_prompts_for_sampling(
        [args.prompt],
        te1,
        te2,
        tok1,
        tok2,
        device,
        clip_skip=args.clip_skip,
        enable_prompt_weighting=args.enable_prompt_weighting,
    )
    negative_prompt_embeds, pooled_negative_prompt_embeds = encode_prompts_for_sampling(
        [args.negative],
        te1,
        te2,
        tok1,
        tok2,
        device,
        clip_skip=args.clip_skip,
        enable_prompt_weighting=args.enable_prompt_weighting,
    )

    latents = sample_frozen_sdxl(
        unet_backend,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        pooled_negative_prompt_embeds=pooled_negative_prompt_embeds,
        resolution=resolution,
        sampler=args.sampler,
        scheduler_name=args.scheduler,
        num_inference_steps=args.sampler_steps,
        guidance_scale=args.cfg,
        device=device,
        dtype=dtype,
        latents=initial_latents,
        seed=args.seed,
        denoise=args.denoise,
        progress=not args.no_progress,
    )

    if args.save_latents is not None:
        save_latents(args.save_latents, latents)
        print(f"Saved latents: {args.save_latents}")

    if not args.no_decode:
        images = decode_latents(vae, latents.to(torch.float32))
        _save_images(images, args.output)
        print(f"Saved image: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except TensorRTUnavailableError as exc:
        print(f"TensorRT error: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
