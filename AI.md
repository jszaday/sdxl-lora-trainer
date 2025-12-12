# AI.md — SDXL LoRA Trainer

## Mission

Build a **single-purpose, high-UX SDXL LoRA trainer**:

- CLI-first: `python -m lora_trainer.train --checkpoint ... --train_data ... --steps ...`
- Obvious flags for anyone “ComfyUI-brained”: `--sampler`, `--scheduler`, `--cfg`, `--sampler_steps`, `--batch_size`, `--steps`, `--samples`, etc.
- Fast feedback: tqdm progress, TensorBoard logs, periodic sample images.
- Clean internals: no legacy branches, no dead modules, no dangling experiments.

This repo should always be in a **coherent, shippable state**.

---

## Principles

1. **Single responsibility**  
   This repo trains SDXL LoRA (and runs validation samples). Nothing else.

2. **No fossil layers**  
   - Do not keep unused code “just in case”.
   - When design changes, **delete** old paths.
   - No “v2_”, “old_”, “_backup” modules. History lives in git.

3. **Test-driven**  
   - Add or update tests **before** or alongside feature work.
   - No new top-level module without corresponding tests.
   - CI should run tests and a tiny end-to-end smoke run.

4. **UX obsessed**  
   - Flag names are short and familiar: `--sampler`, `--scheduler`, `--cfg`, `--sampler_steps`, `--batch_size`, `--steps`, `--samples_per_prompt`, `--sample_every`, etc.
   - Good `--help` text; defaults are sane.
   - Always show progress with `tqdm`.
   - Clear stdout logs; TensorBoard for curves + images.

5. **Modular, not over-engineered**  
   - Small, focused modules; keep imports acyclic and obvious.
   - If a module grows messy, refactor early and delete the old shape.

6. **Grounded in off-the-shelf stacks**  
   - Use `diffusers` + `transformers` (or similar) for SDXL + schedulers + samplers.
   - Wrap, don’t reimplement, unless there’s a concrete performance or UX reason.

---

## Repo Layout

Target structure:

```text
lora-trainer/
  AGENT.md
  pyproject.toml / setup.cfg
  README.md

  src/lora_trainer/
    __init__.py
    cli.py          # entry point wiring args → config → train/run
    config.py       # dataclasses / parsing / validation
    data.py         # datasets, dataloaders
    model.py        # SDXL + LoRA injection, loading from checkpoint
    train_loop.py   # core training loop, schedulers, gradient steps
    sampling.py     # validation sampling (scheduler + sampler + CFG)
    schedulers.py   # scheduler selection wrapper over diffusers/etc
    logging.py      # tqdm, TensorBoard, run dirs
    utils.py        # small shared helpers only

  tests/
    test_config.py
    test_data.py
    test_model.py
    test_train_loop.py
    test_sampling.py
    test_cli_smoke.py
````

No stray top-level scripts. All entry goes through `cli.py` and `python -m lora_trainer.cli` (or a console_script wrapper).

---

## CLI UX (first-class interface)

Primary command:

```bash
python -m lora_trainer.cli \
  --checkpoint base_sdxl.safetensors \
  --train_data /path/to/images \
  --steps 5000 \
  --batch_size 4 \
  --workspace ./runs/my_experiment \
  --scheduler karras \
  --sampler euler_ancestral \
  --cfg 5.0 \
  --sampler_steps 30 \
  --sample_prompts prompts.txt \
  --sample_every 500 \
  --samples_per_prompt 2
```

Required flags:

* `--checkpoint`: base SDXL checkpoint (or HF hub id).
* `--train_data`: directory of images (+ optional `.txt` captions).
* `--steps`: number of optimizer steps.
* `--batch_size`: training batch size.
* `--workspace`: directory for checkpoints/logs/samples.

Sampling / UX flags:

* `--scheduler {simple,normal,karras,...}`
* `--sampler {euler,euler_ancestral,ddim,heun,...}`
* `--cfg FLOAT` (classifier-free guidance scale).
* `--sampler_steps INT` (diffusion steps for validation sampling).
* `--sample_prompts PATH` (file with one prompt per line).
* `--sample_every INT` (global steps between sample runs).
* `--samples_per_prompt INT`.

Other:

* `--grad_accum INT` (gradient accumulation steps).
* `--num_workers INT`.
* `--image_size INT`.
* `--seed INT`.

Help strings must explain **what a parameter does in practice**, not just restate the name.

---

## Phased Development Plan

### Phase 1 — Core Skeleton (No SDXL Yet)

Goal: A **minimal runnable trainer** with tests, ready to swap in SDXL later.

**Features**

* `config.py`:

  * `TrainingConfig` dataclass (all CLI options parsed and validated).
  * Epoch/step math (`steps_per_epoch`, `num_epochs`, `effective_batch = batch_size * grad_accum`).

* `data.py`:

  * `ImageFolderWithCaptions` dataset (images + optional `.txt` captions).
  * `build_dataloader(config)` returning a PyTorch `DataLoader`.

* `model.py`:

  * `DummyUNet` stub for now.
  * `select_lora_params(model)` placeholder (currently returns all params).

* `train_loop.py`:

  * Core training loop using `TrainingConfig`, DataLoader, model, optimizer.
  * Compute epochs from `num_images / effective_batch` and `steps`.
  * `tqdm` for **global step progress**.
  * Checkpoint every N steps.
  * Log scalar loss to TensorBoard.

* `logging.py`:

  * `create_run_dirs(workspace)` → `{workspace}/tb`, `{workspace}/checkpoints`, `{workspace}/samples`.
  * `init_tensorboard(logdir)` → `SummaryWriter`.
  * Consistent run name / step logging helpers.

* `cli.py`:

  * Parse args → `TrainingConfig`.
  * Wire everything and call `train()`.

**Tests**

* `test_config.py`:

  * Valid configs compute expected `num_epochs` & `steps_per_epoch`.
  * Invalid values (e.g. negative steps) raise errors.

* `test_data.py`:

  * Dataset sizes match files.
  * Sample returns `pixel_values` tensor and `caption` string.

* `test_train_loop.py`:

  * Tiny in-memory dataset + `DummyUNet` → training runs for a few steps without crash.
  * Global step count matches requested steps.

* `test_cli_smoke.py`:

  * CLI invoked via `subprocess` with a temp directory, `--steps 3`, tiny dataset → completes.

At the end of Phase 1: **the binary works**, but it’s a fake UNet and no real SDXL logic yet.

---

### Phase 2 — SDXL + LoRA Integration

Goal: Replace the dummy model with **real SDXL UNet + LoRA** using `diffusers` / `transformers`.

**Features**

* `model.py`:

  * `load_sdxl_unet(checkpoint_or_model_id, device, dtype)` using diffusers.
  * LoRA injection:

    * Utility to attach LoRA modules to target layers (e.g., specific attention and feedforward modules).
    * `select_lora_params()` returns only LoRA parameters.
  * Optional `--lora_rank`, `--lora_alpha` if needed, but keep the flag set small and obvious.

* `train_loop.py`:

  * Real diffusion loss:

    * Precompute/no-op: for now, maybe keep VAE on-the-fly or use a simple latent fake; actual precomputed latent phase can wait.
    * Sample timesteps `t`, add noise to latents, predict noise via UNet, compute MSE vs noise.
  * Respect `grad_accum`.

**Tests**

* `test_model.py`:

  * SDXL UNet loads on CPU (mock or tiny config).
  * LoRA modules attached to expected layers.
  * `select_lora_params()` returns non-empty subset.

* `test_train_loop.py`:

  * With a very small model variant and tiny dataset, a few steps complete and update LoRA params.

At the end of Phase 2: we have **real LoRA training**, basic but correct.

---

### Phase 3 — Samplers, Schedulers, and Validation Samples

Goal: Implement the **validation sampling** path using the user-facing `--scheduler`, `--sampler`, `--cfg`, `--sampler_steps`, `--sample_prompts`, `--sample_every`, `--samples_per_prompt`.

**Features**

* `schedulers.py`:

  * `build_noise_scheduler(name, config)` returning a diffusers scheduler instance mapped from the user string.
  * Minimal mapping table: e.g. `"simple"`, `"normal"`, `"karras"` → corresponding schedulers.

* `sampling.py`:

  * `run_validation_samples(model, sampler_config, global_step, writer, device, image_size)`:

    * Encodes prompts via text encoders.
    * Prepares unconditional/conditional embeddings.
    * Runs diffusion steps with classifier-free guidance using selected scheduler/sampler.
    * Logs sample grid to TensorBoard and saves PNGs to `{workspace}/samples/step_{global_step}.png`.
  * Honour:

    * `--scheduler`
    * `--sampler`
    * `--cfg`
    * `--sampler_steps`
    * `--samples_per_prompt`.

* `train_loop.py`:

  * Call `run_validation_samples(...)` whenever `global_step % sample_every == 0` and prompts exist.

**Tests**

* `test_schedulers.py`:

  * Each named scheduler creates a viable diffusers scheduler.
  * Unknown names raise clear `ValueError`.

* `test_sampling.py`:

  * With a tiny fake model / scheduler, sampling runs and creates an image tensor and a PNG file.

At the end of Phase 3: the tool **trains LoRA and periodically samples images** to visualize progress.

---

### Phase 4 — Performance & Ergonomics

Goal: Make the trainer feel **fast and polished**, not like a sluggish research script.

**Features**

* Data:

  * Pin memory, sensible `num_workers`.
  * Option to precompute latents/embeddings offline (future extension); keep hooks ready but don’t bloat UX.

* Loop:

  * Use `tqdm` over global steps:

    * Show `loss`, `lr`, `epoch`, `step/steps`, ETA.
  * Minimal host/device sync points.

* Logging:

  * Optionally log:

    * `loss`, `lr`, `grad_norm` curves.
    * `cfg`, sampler, scheduler into TensorBoard hparams / JSON file.

* UX:

  * Clear startup printout summarizing all effective settings (including computed `steps_per_epoch`, `num_epochs`, `effective_batch`).
  * Friendly error messages when configs are inconsistent (e.g. `steps` too small vs dataset/documented expectations).

### TODO: Low-VRAM Mode

- Add `--low_vram` flag that:
  - Enables gradient checkpointing on UNet/text encoders.
  - Offloads unused modules (VAE, text encoders, non-LoRA UNet parts) to CPU via `enable_sequential_cpu_offload`/`enable_model_cpu_offload` or `accelerate` cpu_offload.
  - Optionally switches optimizer to bitsandbytes AdamW8bit and keeps optimizer states on CPU.
  - Forces consistent mixed precision (fp16/bf16) and keeps LoRA params in-model dtype.
  - Minimizes device hops in the training inner loop; ensures components are moved back before sampling.

### TODO: Structured Sampling Prompts

- Allow a JSON/JSONL prompts file with per-sample fields: `positive`, `negative`, `seed`, and optional `cfg`, `sampler_steps`, etc.
- Maintain simple text-file compatibility (one prompt per line) as a fallback.
- Wire parsing into sampling so each sample can carry its own negative prompt and seed for reproducibility.

### TODO: Optimizer Flexibility

- Add a `--optimizer` choice with plug-ins: `adamw`, `adamw_8bit` (bitsandbytes), `lion`, `prodigy`, etc.
- Accept optimizer-specific knobs (weight decay, beta params, trust region/ema toggles) with sane defaults per optimizer.
- Ensure dtype/device compatibility for 8bit/low-precision optimizers and make sure state init respects LoRA-only training.
- Parse a compact function-call-like spec from the flag, e.g. `--optimizer "prodigy(lr=1e-4, weight_decay=0, slice_p=11)"`, merging provided kwargs over defaults and validating fields.

**Tests**

* Ensure progress bar still works in non-interactive environments (or degrade gracefully).
* Tiny run with TensorBoard logging enabled completes and creates events.

---

## Coding Standards

* Use `black` + `isort` + `ruff` (or equivalent) from day one.
* Type hints on public functions and core dataclasses.
* Keep functions short; if something needs explaining in comments, consider splitting it.

Non-obvious logic gets short, pointed comments. No comment novels.

---

## Summary

This agent’s job is to produce a **small, sharp, SDXL LoRA trainer**:

* A single CLI with sane, ComfyUI-like flags.
* Clear progress and visualization.
* Clean internals, no legacy junk.
* Built test-first, with each phase leaving the repo in a coherent, usable state.

If a change makes the design better and invalidates old code, **remove the old code**. Always prefer a smaller, cleaner tree over “supporting everything.”
