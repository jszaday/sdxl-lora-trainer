"""Hires-fix pipeline: SDXL → 4x upscale → downscale → SDXL."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from .pipeline import load_inference_models, load_torch_unet_backend
from .sampling import decode_latents, encode_latents, encode_prompts_for_sampling
from .trt.config import ResolutionSpec
from .trt.inference import sample_frozen_sdxl
from .trt.upscaler import CompiledUpscalerBackend, tiled_upscale


def _resize(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if images.shape[2] == height and images.shape[3] == width:
        return images
    return F.interpolate(images, size=(height, width), mode="bicubic", antialias=True).clamp(
        0.0, 1.0
    )


def _feather_mask(h: int, w: int, feather: int, device: str) -> torch.Tensor:
    mask = torch.ones(1, 1, h, w, device=device)
    for i in range(min(feather, h // 2, w // 2)):
        v = i / feather
        mask[:, :, i, :].clamp_(max=v)
        mask[:, :, h - 1 - i, :].clamp_(max=v)
        mask[:, :, :, i].clamp_(max=v)
        mask[:, :, :, w - 1 - i].clamp_(max=v)
    return mask


def refine_faces(
    images: torch.Tensor,
    vae,
    unet_backend,
    pos_embeds: torch.Tensor,
    pos_pooled: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_pooled: torch.Tensor,
    *,
    detector,
    sampler: str,
    scheduler_name: str,
    steps: int,
    denoise: float,
    cfg: float,
    face_size: int,
    padding: int,
    feather: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    progress: bool,
) -> torch.Tensor:
    """ADetailer-style face refinement: detect → crop → img2img → paste back.

    detector is a pre-loaded YOLO instance (not a path) so the server can keep it
    in memory across requests without re-loading the model each call.
    """
    face_res = ResolutionSpec(name=f"{face_size}x{face_size}", width=face_size, height=face_size)
    result = images.clone()
    _, _, img_h, img_w = images.shape

    for b in range(images.shape[0]):
        img_np = (images[b].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
        boxes = detector(img_np, verbose=False)[0].boxes

        if boxes is None or len(boxes) == 0:
            print("  No faces detected.")
            continue

        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            x1 = max(0, (int(x1 - padding) // 8) * 8)
            y1 = max(0, (int(y1 - padding) // 8) * 8)
            x2 = min(img_w, ((int(x2 + padding) + 7) // 8) * 8)
            y2 = min(img_h, ((int(y2 + padding) + 7) // 8) * 8)
            crop_h, crop_w = y2 - y1, x2 - x1

            crop = result[b : b + 1, :, y1:y2, x1:x2]
            crop_resized = _resize(crop, face_size, face_size)

            clean_latents = encode_latents(vae, crop_resized)
            refined_latents = sample_frozen_sdxl(
                unet_backend,
                prompt_embeds=pos_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pos_pooled,
                pooled_negative_prompt_embeds=neg_pooled,
                resolution=face_res,
                sampler=sampler,
                scheduler_name=scheduler_name,
                num_inference_steps=steps,
                guidance_scale=cfg,
                device=device,
                dtype=dtype,
                latents=clean_latents,
                seed=seed,
                denoise=denoise,
                progress=progress,
            )
            refined = decode_latents(vae, refined_latents.to(torch.float32))
            refined_crop = _resize(refined, crop_h, crop_w).to(device)

            mask = _feather_mask(crop_h, crop_w, feather, device)
            result[b : b + 1, :, y1:y2, x1:x2] = mask * refined_crop + (1 - mask) * result[
                b : b + 1, :, y1:y2, x1:x2
            ].to(device)

    return result


def run_hires_fix_preloaded(
    vae,
    te1,
    te2,
    tok1,
    tok2,
    unet_backend,
    upscaler,
    prompt: str,
    *,
    negative: str = "",
    first_resolution: ResolutionSpec,
    hires_resolution: ResolutionSpec,
    first_steps: int = 30,
    first_denoise: float = 1.0,
    second_steps: int = 20,
    second_denoise: float = 0.65,
    cfg: float = 5.5,
    sampler: str = "euler",
    scheduler_name: str = "karras",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
    second_seed: int | None = None,
    clip_skip: int = 1,
    upscale_tile_size: int = 512,
    upscale_overlap: int = 32,
    enable_prompt_weighting: bool = True,
    progress: bool = True,
    face_detector=None,
    face_denoise: float = 0.4,
    face_steps: int = 20,
    face_size: int = 512,
    face_padding: int = 32,
    face_feather: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a full hires-fix pass with pre-loaded models.

    Returns (first_pass_images, final_images) as [B, 3, H, W] float32 in [0, 1].
    first_pass_images is at first_resolution; final_images is at hires_resolution.
    """
    print("Encoding prompts …")
    pos_embeds, pos_pooled = encode_prompts_for_sampling(
        [prompt],
        te1,
        te2,
        tok1,
        tok2,
        device,
        clip_skip=clip_skip,
        enable_prompt_weighting=enable_prompt_weighting,
    )
    neg_embeds, neg_pooled = encode_prompts_for_sampling(
        [negative],
        te1,
        te2,
        tok1,
        tok2,
        device,
        clip_skip=clip_skip,
        enable_prompt_weighting=enable_prompt_weighting,
    )

    _second_seed = second_seed if second_seed is not None else seed

    # ── 1. First SDXL pass ──────────────────────────────────────────────────
    print(f"\n[1/3] SDXL  {first_resolution.name}  {first_steps} steps  denoise={first_denoise}")
    latents_1 = sample_frozen_sdxl(
        unet_backend,
        prompt_embeds=pos_embeds,
        negative_prompt_embeds=neg_embeds,
        pooled_prompt_embeds=pos_pooled,
        pooled_negative_prompt_embeds=neg_pooled,
        resolution=first_resolution,
        sampler=sampler,
        scheduler_name=scheduler_name,
        num_inference_steps=first_steps,
        guidance_scale=cfg,
        device=device,
        dtype=dtype,
        seed=seed,
        denoise=first_denoise,
        progress=progress,
    )
    images_1 = decode_latents(vae, latents_1.to(torch.float32))

    # ── 2. Upscale → downscale ──────────────────────────────────────────────
    print(f"\n[2/3] Upscale {upscaler.scale}×  →  downscale to {hires_resolution.name}")
    upscaled = tiled_upscale(
        images_1.to(device=device, dtype=torch.float32),
        upscaler,
        tile_size=upscale_tile_size,
        overlap=upscale_overlap,
        scale=upscaler.scale,
    )
    downscaled = _resize(upscaled, hires_resolution.height, hires_resolution.width)

    # ── 3. Encode → add noise → second SDXL pass ───────────────────────────
    print(f"\n[3/3] SDXL  {hires_resolution.name}  {second_steps} steps  denoise={second_denoise}")
    clean_latents = encode_latents(vae, downscaled)
    latents_2 = sample_frozen_sdxl(
        unet_backend,
        prompt_embeds=pos_embeds,
        negative_prompt_embeds=neg_embeds,
        pooled_prompt_embeds=pos_pooled,
        pooled_negative_prompt_embeds=neg_pooled,
        resolution=hires_resolution,
        sampler=sampler,
        scheduler_name=scheduler_name,
        num_inference_steps=second_steps,
        guidance_scale=cfg,
        device=device,
        dtype=dtype,
        latents=clean_latents,
        seed=_second_seed,
        denoise=second_denoise,
        progress=progress,
    )
    images_final = decode_latents(vae, latents_2.to(torch.float32))

    # ── 4. ADetailer face refinement ───────────────────────────────────────
    if face_detector is not None:
        print(f"\n[4/4] ADetailer  face_size={face_size}  denoise={face_denoise}")
        images_final = refine_faces(
            images_final,
            vae,
            unet_backend,
            pos_embeds,
            pos_pooled,
            neg_embeds,
            neg_pooled,
            detector=face_detector,
            sampler=sampler,
            scheduler_name=scheduler_name,
            steps=face_steps,
            denoise=face_denoise,
            cfg=cfg,
            face_size=face_size,
            padding=face_padding,
            feather=face_feather,
            device=device,
            dtype=dtype,
            seed=_second_seed,
            progress=progress,
        )

    return images_1, images_final


def run_hires_fix(
    checkpoint: str,
    upscale_model_path: Path,
    prompt: str,
    *,
    negative: str = "",
    first_resolution: ResolutionSpec,
    hires_resolution: ResolutionSpec,
    first_steps: int = 30,
    first_denoise: float = 1.0,
    second_steps: int = 20,
    second_denoise: float = 0.65,
    cfg: float = 5.5,
    sampler: str = "euler",
    scheduler_name: str = "karras",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
    second_seed: int | None = None,
    clip_skip: int = 1,
    upscale_tile_size: int = 512,
    upscale_overlap: int = 32,
    lora_checkpoint: Path | None = None,
    lora_rank: int = 16,
    enable_prompt_weighting: bool = True,
    progress: bool = True,
    face_model_path: Path | None = None,
    face_denoise: float = 0.4,
    face_steps: int = 20,
    face_size: int = 512,
    face_padding: int = 32,
    face_feather: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load models and run a full hires-fix pass.

    Returns (first_pass_images, final_images) as [B, 3, H, W] float32 in [0, 1].
    first_pass_images is at first_resolution; final_images is at hires_resolution.
    """
    print(f"Loading SDXL from: {checkpoint}")
    vae, te1, te2, tok1, tok2 = load_inference_models(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_checkpoint=lora_checkpoint,
        lora_rank=lora_rank,
    )
    unet_backend = load_torch_unet_backend(
        checkpoint,
        device=device,
        dtype=dtype,
        lora_checkpoint=lora_checkpoint,
        lora_rank=lora_rank,
    )

    print(f"Loading upscale model: {Path(upscale_model_path).name}")
    upscaler = CompiledUpscalerBackend(
        upscale_model_path, device=device, tile_size=upscale_tile_size
    )

    face_detector = None
    if face_model_path is not None:
        from ultralytics import YOLO

        face_detector = YOLO(str(face_model_path))

    return run_hires_fix_preloaded(
        vae,
        te1,
        te2,
        tok1,
        tok2,
        unet_backend,
        upscaler,
        prompt,
        negative=negative,
        first_resolution=first_resolution,
        hires_resolution=hires_resolution,
        first_steps=first_steps,
        first_denoise=first_denoise,
        second_steps=second_steps,
        second_denoise=second_denoise,
        cfg=cfg,
        sampler=sampler,
        scheduler_name=scheduler_name,
        device=device,
        dtype=dtype,
        seed=seed,
        second_seed=second_seed,
        clip_skip=clip_skip,
        upscale_tile_size=upscale_tile_size,
        upscale_overlap=upscale_overlap,
        enable_prompt_weighting=enable_prompt_weighting,
        progress=progress,
        face_detector=face_detector,
        face_denoise=face_denoise,
        face_steps=face_steps,
        face_size=face_size,
        face_padding=face_padding,
        face_feather=face_feather,
    )


class InferencePipeline:
    """Full inference pipeline: SDXL → [upscale → SDXL] → [ADetailer].

    Plain Python class — not an nn.Module. All sub-models are already frozen
    and in eval mode before being passed in; there is no training, gradient
    tracking, or AMP state to manage here.
    """

    def __init__(
        self,
        vae,
        te1,
        te2,
        tok1,
        tok2,
        unet_backend,
        *,
        upscaler=None,
        face_detector=None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        upscale_tile_size: int = 512,
        upscale_overlap: int = 32,
    ) -> None:
        self.vae = vae
        self.te1 = te1
        self.te2 = te2
        self.tok1 = tok1
        self.tok2 = tok2
        self.unet_backend = unet_backend
        self.upscaler = upscaler
        self.face_detector = face_detector
        self.device = device
        self.dtype = dtype
        self.upscale_tile_size = upscale_tile_size
        self.upscale_overlap = upscale_overlap

    def __call__(
        self,
        prompt: str,
        negative: str = "",
        *,
        mode: str = "simple",
        resolution: ResolutionSpec,
        sampler: str = "euler",
        scheduler_name: str = "karras",
        steps: int = 30,
        denoise: float = 1.0,
        cfg: float = 7.0,
        clip_skip: int = 1,
        seed: int = 42,
        enable_prompt_weighting: bool = True,
        progress: bool = True,
        on_step=None,
        # hires-only
        hires_resolution: ResolutionSpec | None = None,
        second_steps: int = 20,
        second_denoise: float = 0.65,
        # face refinement (both modes)
        face_denoise: float = 0.4,
        face_steps: int = 20,
        face_size: int = 512,
        face_padding: int = 32,
        face_feather: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the pipeline. Returns (first_images, final_images) as [1,3,H,W] float32 in [0,1].

        For mode='simple', both elements are the same tensor.
        """
        if mode == "hires":
            if self.upscaler is None:
                raise RuntimeError("mode='hires' requires an upscaler (none loaded at startup)")
            return run_hires_fix_preloaded(
                self.vae,
                self.te1,
                self.te2,
                self.tok1,
                self.tok2,
                self.unet_backend,
                self.upscaler,
                prompt,
                negative=negative,
                first_resolution=resolution,
                hires_resolution=hires_resolution or resolution,
                first_steps=steps,
                first_denoise=denoise,
                second_steps=second_steps,
                second_denoise=second_denoise,
                cfg=cfg,
                sampler=sampler,
                scheduler_name=scheduler_name,
                device=self.device,
                dtype=self.dtype,
                seed=seed,
                clip_skip=clip_skip,
                upscale_tile_size=self.upscale_tile_size,
                upscale_overlap=self.upscale_overlap,
                enable_prompt_weighting=enable_prompt_weighting,
                progress=progress,
                face_detector=self.face_detector,
                face_denoise=face_denoise,
                face_steps=face_steps,
                face_size=face_size,
                face_padding=face_padding,
                face_feather=face_feather,
            )

        # --- simple mode ---
        pos_embeds, pos_pooled = encode_prompts_for_sampling(
            [prompt],
            self.te1,
            self.te2,
            self.tok1,
            self.tok2,
            self.device,
            clip_skip=clip_skip,
            enable_prompt_weighting=enable_prompt_weighting,
        )
        neg_embeds, neg_pooled = encode_prompts_for_sampling(
            [negative],
            self.te1,
            self.te2,
            self.tok1,
            self.tok2,
            self.device,
            clip_skip=clip_skip,
            enable_prompt_weighting=enable_prompt_weighting,
        )
        latents = sample_frozen_sdxl(
            self.unet_backend,
            prompt_embeds=pos_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pos_pooled,
            pooled_negative_prompt_embeds=neg_pooled,
            resolution=resolution,
            sampler=sampler,
            scheduler_name=scheduler_name,
            num_inference_steps=steps,
            guidance_scale=cfg,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
            denoise=denoise,
            progress=progress,
            on_step=on_step,
        )
        images = decode_latents(self.vae, latents.to(torch.float32))

        if self.face_detector is not None:
            images = refine_faces(
                images,
                self.vae,
                self.unet_backend,
                pos_embeds,
                pos_pooled,
                neg_embeds,
                neg_pooled,
                detector=self.face_detector,
                sampler=sampler,
                scheduler_name=scheduler_name,
                steps=face_steps,
                denoise=face_denoise,
                cfg=cfg,
                face_size=face_size,
                padding=face_padding,
                feather=face_feather,
                device=self.device,
                dtype=self.dtype,
                seed=seed,
                progress=progress,
            )

        return images, images
