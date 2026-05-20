"""Frozen SDXL inference entry point with TensorRT-ready shapes."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from .model import (
    load_lora_weights,
    load_sdxl_unet,
    load_text_encoders,
    load_vae,
    merge_lora_layers,
)
from .sampling import decode_latents, encode_prompts_for_sampling
from .trt.backends import TensorRTUnavailableError, TensorRTUnetBackend, TorchUnetBackend
from .trt.build import build_unet_engine
from .trt.cache import build_engine_cache_key, resolve_engine_artifacts
from .trt.config import SDXL_RESOLUTIONS, infer_resolution_from_latents, parse_resolution
from .trt.inference import sample_frozen_sdxl
from .trt.latents import load_latents, save_latents
from .utils import set_seed


def _dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _save_images(images: torch.Tensor, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        suffix = output.suffix or ".png"
        path = output
        if images.shape[0] > 1:
            path = output.with_name(f"{output.stem}_{idx}{suffix}")
        image_np = image.to(torch.float32).cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).round().clip(0, 255).astype("uint8")
        Image.fromarray(image_np).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen SDXL inference with fixed TensorRT-ready resolutions",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Base SDXL checkpoint path or HuggingFace model ID.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Positive prompt to render.",
    )
    parser.add_argument(
        "--negative",
        default="",
        help="Negative prompt to steer away from unwanted content.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample.png"),
        help="Output image path. Batched outputs append _N before the suffix.",
    )
    parser.add_argument(
        "--resolution",
        choices=list(SDXL_RESOLUTIONS),
        default="1024x1024",
        help="Fixed SDXL-native output resolution.",
    )
    parser.add_argument(
        "--latents",
        type=Path,
        default=None,
        help="Optional starting latents as .safetensors or torch checkpoint.",
    )
    parser.add_argument(
        "--save_latents",
        type=Path,
        default=None,
        help="Optional path to save final latents before VAE decode.",
    )
    parser.add_argument(
        "--no_decode",
        action="store_true",
        help="Skip VAE decoding and only save latents when --save_latents is set.",
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "trt"],
        default="trt",
        help="UNet backend. Defaults to TensorRT; use torch only for baseline comparisons.",
    )
    parser.add_argument(
        "--engine_dir",
        type=Path,
        default=Path("engines"),
        help="Directory containing TensorRT plan files for --backend trt.",
    )
    parser.add_argument(
        "--onnx_dir",
        type=Path,
        default=Path("engines/onnx"),
        help="Directory for intermediate ONNX exports when auto-building TensorRT engines.",
    )
    parser.add_argument(
        "--no_build_engine",
        action="store_true",
        help="Fail instead of auto-building a missing TensorRT engine.",
    )
    parser.add_argument(
        "--force_build_engine",
        action="store_true",
        help="Re-export ONNX and rebuild the TensorRT engine before inference.",
    )
    parser.add_argument(
        "--workspace_gb",
        type=float,
        default=12.0,
        help="TensorRT builder workspace limit in GiB for auto-builds.",
    )
    parser.add_argument(
        "--compile_unet",
        action="store_true",
        help="Apply torch.compile to the UNet backend for fixed-shape inference.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["simple", "normal", "karras", "exponential", "sgm_uniform"],
        default="karras",
        help="Noise schedule used by the denoise loop.",
    )
    parser.add_argument(
        "--sampler",
        choices=[
            "euler",
            "euler_ancestral",
            "heun",
            "dpmpp_2m",
            "dpmpp_2m_sde",
            "dpmpp_sde",
            "lms",
            "pndm",
            "ddim",
        ],
        default="euler",
        help="Sampler algorithm for scheduler.step updates.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=5.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=30,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--denoise",
        type=float,
        default=1.0,
        help="Fraction of denoise schedule to run when starting from supplied latents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generated initial latents.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Inference precision for model weights and latents.",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=Path,
        default=None,
        help="Optional LoRA checkpoint to merge into the frozen inference model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank to instantiate before loading --lora_checkpoint.",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=1,
        help="CLIP skip for text_encoder_1 hidden states.",
    )
    parser.add_argument(
        "--disable_prompt_weighting",
        dest="enable_prompt_weighting",
        action="store_false",
        default=True,
        help="Treat parentheses in prompts literally instead of applying prompt weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_precision(args.precision)
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

    if args.backend == "trt" and not args.no_build_engine:
        artifacts = build_unet_engine(
            args.checkpoint,
            resolution,
            engine_dir=args.engine_dir,
            onnx_dir=args.onnx_dir,
            precision=args.precision,
            device=device,
            lora_checkpoint=args.lora_checkpoint,
            lora_rank=args.lora_rank,
            workspace_gb=args.workspace_gb,
            force=args.force_build_engine,
        )
    else:
        artifacts = None

    lora_rank = args.lora_rank if args.lora_checkpoint is not None else None
    print(f"Loading SDXL components from: {args.checkpoint}")
    unet = None
    if args.backend == "torch" or args.lora_checkpoint is not None:
        unet, _ = load_sdxl_unet(
            args.checkpoint,
            device=device,
            dtype=dtype,
            lora_rank=lora_rank,
            adapter="lora",
        )
    vae = load_vae(args.checkpoint, device=device, dtype=dtype)
    text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, _ = load_text_encoders(
        args.checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=lora_rank,
        adapter="lora",
    )

    if args.lora_checkpoint is not None:
        print(f"Loading and merging LoRA: {args.lora_checkpoint}")
        if unet is None:
            raise ValueError("Internal error: LoRA loading requires a Torch UNet")
        load_lora_weights(
            args.lora_checkpoint,
            unet=unet,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
        )
        merge_lora_layers(unet)
        merge_lora_layers(text_encoder_1)
        merge_lora_layers(text_encoder_2)

    if unet is not None:
        unet.requires_grad_(False)
        unet.eval()
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.eval()
    text_encoder_1.eval()
    text_encoder_2.eval()

    prompt_embeds, pooled_prompt_embeds = encode_prompts_for_sampling(
        [args.prompt],
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        device,
        clip_skip=args.clip_skip,
        enable_prompt_weighting=args.enable_prompt_weighting,
    )
    negative_prompt_embeds, pooled_negative_prompt_embeds = encode_prompts_for_sampling(
        [args.negative],
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        device,
        clip_skip=args.clip_skip,
        enable_prompt_weighting=args.enable_prompt_weighting,
    )

    if args.backend == "trt":
        if artifacts is not None:
            engine_path = artifacts.engine_path
        else:
            key = build_engine_cache_key(
                args.checkpoint,
                resolution,
                precision=args.precision,
                lora_checkpoint=args.lora_checkpoint,
            )
            engine_path = resolve_engine_artifacts(args.engine_dir, args.onnx_dir, key).engine_path
        unet_backend = TensorRTUnetBackend(engine_path)
    else:
        if unet is None:
            raise ValueError("Torch backend requires a loaded UNet")
        unet_backend = TorchUnetBackend(unet, compile_unet=args.compile_unet)

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
    )

    if args.save_latents is not None:
        save_latents(args.save_latents, latents)
        print(f"Saved latents: {args.save_latents}")

    if not args.no_decode:
        vae = vae.to(dtype=torch.float32)
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
