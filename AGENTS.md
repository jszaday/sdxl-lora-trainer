# AGENTS.md — SDXL LoRA Trainer

## Mission

Build and maintain a **single-purpose, high-UX SDXL LoRA trainer**:

- CLI-first: `python -m lora_trainer.cli --checkpoint ... --train_data ... --steps ...`
- Obvious flags for anyone “ComfyUI-brained”: `--sampler`, `--scheduler`, `--cfg`, `--sampler_steps`, `--batch_size`, `--steps`, `--samples`, etc.
- Fast feedback: tqdm progress, TensorBoard logs, periodic sample images.
- Clean internals: no legacy branches, no dead modules, no dangling experiments.

This repo must always be in a **coherent, shippable state**.

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

```text
lora-trainer/
  AGENTS.md
  pyproject.toml
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

  src/lora_converter/
    __init__.py
    cli.py          # entry point for checkpoint conversion
    converter.py    # logic for rewriting keys to ComfyUI/LyCORIS formats

  tests/
    test_config.py
    test_data.py
    test_model.py
    test_train_loop.py
    test_sampling.py
    test_cli_smoke.py
```

No stray top-level scripts. All entry goes through `cli.py` and `python -m lora_trainer.cli` (or a console_script wrapper).

---

## CLI UX (first-class interface)

Primary command example:

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

## Coding Standards

* Use `black` + `isort` + `ruff` (or equivalent) for all code.
* Type hints are required on public functions and core dataclasses.
* Keep functions short; if something needs explaining in comments, consider splitting it.
* Non-obvious logic gets short, pointed comments. No comment novels.
* Always prioritize clarity and maintainability over "clever" implementations.
