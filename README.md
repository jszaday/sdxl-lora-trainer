# SDXL LoRA Trainer

Single-purpose, high-UX SDXL LoRA training toolkit with a CLI-first design.

## Overview

This is a focused SDXL LoRA trainer designed for ease of use with ComfyUI-style parameters. It provides:

- Clean CLI interface with familiar flags
- Fast feedback via tqdm progress bars and TensorBoard
- Periodic validation sample generation
- Supports both classic LoRA and LyCORIS adapters
- Well-tested, modular codebase

## Installation

**Recommended: Use a virtual environment**

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install with dev dependencies for testing
pip install -e ".[dev]"
```

## Quick Start

```bash
python -m lora_trainer.cli \
  --checkpoint stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data /path/to/training/images \
  --steps 5000 \
  --batch_size 4 \
  --workspace ./runs/my_experiment
```

## Usage

### Required Arguments

- `--checkpoint`: Base SDXL checkpoint path or HuggingFace model ID
- `--train_data`: Directory containing training images (with optional `.txt` caption files)
- `--steps`: Total number of training steps
- `--batch_size`: Batch size per GPU
- `--workspace`: Output directory for checkpoints, logs, and samples

### Training Data Format

Place your training images in a directory:

```
/path/to/training/images/
  image1.jpg
  image1.txt  (optional caption)
  image2.png
  image2.txt  (optional caption)
  ...
```

If a `.txt` file with the same basename exists, it will be used as the caption. Otherwise, the caption defaults to an empty string.

### Sampling & Validation

Generate validation samples during training:

```bash
python -m lora_trainer.cli \
  --checkpoint base_sdxl.safetensors \
  --train_data /path/to/images \
  --steps 5000 \
  --batch_size 4 \
  --workspace ./runs/experiment \
  --optimizer "lion(lr=1e-4,betas=(0.9,0.99),weight_decay=0.01)" \
  --lr_scheduler "constant_with_warmup(warmup_steps=100)" \
  --sample_prompts prompts.json \
  --sample_every 500 \
  --scheduler karras \
  --sampler euler_ancestral \
  --cfg 7.0 \
  --sampler_steps 30 \
  --samples_per_prompt 2
```

Structured prompts are provided as JSON/JSONL with per-sample fields:

```json
[
  { "prompt": "a photo of a mountain landscape", "negative": "low-res", "seed": 1234, "name": "mountain" },
  { "prompt": "a portrait of a person", "negative": "blurry" },
  { "prompt": "a cute cat", "seed": 42, "name": "cat_portrait" }
]
```

JSONL is also supported (one JSON object per line). Each entry may include:
- `prompt`/`positive`: required positive text
- `negative`: optional negative prompt (defaults to empty)
- `seed`: optional integer seed
- `name`: optional name for the output file (defaults to index-based naming)

Each sample is saved as an individual image file:
- With name: `step_000500_mountain.png`
- Without name: `step_000500_0.png`, `step_000500_1.png`, etc.

### All CLI Options

**Optimizer:**
- `--learning_rate`: Learning rate (default: 1e-4)
- `--grad_accum`: Gradient accumulation steps (default: 1)
- `--optimizer`: Optimizer spec (default: `adamw`). Supports `adamw`, `lion`, `prodigy`. You can pass kwargs: e.g., `prodigy(lr=1e-4, weight_decay=0)`.
  - Install extras if needed: `prodigyopt` for `prodigy`, `torch_optimizer` for `lion`.
- `--lr_scheduler`: LR scheduler spec, e.g. `constant_with_warmup(warmup_steps=100)`. Supported names: `constant`, `constant_with_warmup`, `linear`, `cosine`, `cosine_with_restarts`, `polynomial`.

**Data:**
- `--image_size`: Image size for training (default: 1024)
- `--num_workers`: Data loading workers (default: 4)

**Sampling:**
- `--scheduler`: Noise scheduler - `simple`, `normal`, `karras`, `exponential`, `sgm_uniform` (default: normal)
- `--sampler`: Sampler algorithm - `euler`, `euler_ancestral`, `heun`, `dpmpp_2m`, `dpmpp_2m_sde`, `dpmpp_sde`, `lms`, `pndm`, `ddim` (default: euler)
- `--cfg`: Classifier-free guidance scale (default: 7.0)
- `--sampler_steps`: Diffusion steps for sampling (default: 30)
- `--sample_prompts`: Path to prompts file
- `--sample_every`: Generate samples every N steps (default: 500)
- `--samples_per_prompt`: Number of samples per prompt (default: 1)
- `--sample_clip_skip`: Clip skip for text_encoder_1 hidden states (1 = penultimate; default: 1)
- `--enable_training_prompt_weighting`: Enable weighting syntax in training captions (default: False)

**LoRA:**
- `--adapter`: Adapter spec. Examples: `lora(rank=16,alpha=16)` or `locon(rank=16,alpha=16,dropout=0.1)` (default: `lora`)

**Misc:**
- `--device`: Device to use for training - `cuda`, `cpu`, `mps`, etc. (auto-detected if not specified)
- `--seed`: Random seed (default: 42)
- `--mixed_precision`: Mixed precision mode - `no`, `fp16`, `bf16` (default: fp16)
- `--resume_from`: Checkpoint file or directory to resume training from. If a directory is given, the newest `.pt` file is picked.
- `--low_vram`: Enable memory-saving bundle (gradient checkpointing + 8-bit optimizer)
- `--gradient_checkpointing`: Enable gradient checkpointing specifically (saves VRAM)

### LyCORIS Mode

Switch to LyCORIS adapters with familiar knobs:

```bash
python -m lora_trainer.cli \
  --checkpoint stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data /path/to/training/images \
  --steps 5000 \
  --batch_size 4 \
  --workspace ./runs/my_lyco_experiment \
  --adapter "lycoris(algo=lokr,dim=16,alpha=2.0)"
```

Or use a spec string (LoCon example):

```bash
python -m lora_trainer.cli \
  --checkpoint stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data /path/to/training/images \
  --steps 5000 \
  --batch_size 4 \
  --workspace ./runs/my_locon_experiment \
  --adapter "locon(rank=16,alpha=2.0,dropout=0.1)"
```

### Standalone Sampler CLI

Run sampling without training using the structured prompts file:

```bash
python -m lora_trainer.sampler_cli \
  --checkpoint base_sdxl.safetensors \
  --sample_prompts prompts.json \
  --workspace ./runs/sampler_outputs \
  --scheduler karras \
  --cfg 7.0 \
  --sampler_steps 30
```

- Supports the same prompt JSON/JSONL format as training (including optional `name` field).
- Optional: `--lora_checkpoint` to load LoRA weights from a training checkpoint before sampling.
- Pass `--adapter "lora"` or `--adapter "lycoris"` to specify adapter type.
- Outputs individual sample images to `{workspace}/samples/` (e.g., `step_000000_0.png` or `step_000000_mountain.png`).
- Each sample is logged separately to TensorBoard at `{workspace}/tb/` with unique tags (`samples/0`, `samples/mountain`, etc.).

**Note:** When loading a checkpoint, adapter parameters (rank, alpha, etc.) are automatically detected and loaded from the checkpoint. You typically only need to specify `--adapter` type and optionally `--lora_checkpoint` path.

### Checkpoint Conversion

#### LoRA Conversion

Convert a training `.pt` checkpoint (or LoRA-only `.pt`) into ComfyUI-ready safetensors with Comfy's `lora_unet_*` naming:

```bash
python -m lora_converter.cli /path/to/step_000500.pt --output final_lora.safetensors
# or, if installed as a console script:
lora-convert /path/to/step_000500.pt
```

The converter extracts only LoRA tensors, rewrites the keys to ComfyUI's expected format, and avoids duplicate aliases.

#### LyCORIS Conversion

Convert any LyCORIS checkpoint to safetensors format for web use:

```bash
# Convert any checkpoint with --lycoris flag
python -m lora_converter.cli checkpoint_step_1000.pt --lycoris

# Or with custom output path
python -m lora_converter.cli checkpoint_step_1000.pt --output my_lycoris.safetensors --lycoris
```

The LyCORIS converter:
- Works with **any** checkpoint (not just final), including intermediate training steps
- Auto-detects the algorithm type (lokr, loha, diag-oft, locon)
- Auto-infers network dimensions from tensor shapes
- Saves weights in native LyCORIS format (no conversion needed)
- Combines UNet and text encoder weights into a single file
- **Automatic**: Final checkpoints are automatically converted to `final_lycoris.safetensors` during training

## Monitoring Training

### TensorBoard

The trainer writes logs to `{workspace}/tb/`. Launch TensorBoard separately to view them:

```bash
# Install TensorBoard (in your venv)
pip install tensorboard

# Launch TensorBoard with custom port
tensorboard --logdir ./runs/my_experiment/tb --port 6006
```

Then open http://localhost:6006 in your browser to view:
- Training loss curves
- Learning rate schedules
- Generated validation samples (images)

### Output Structure

Your workspace will contain:

```
./runs/my_experiment/
  checkpoints/        # Model checkpoints
    checkpoint_step_000500.pt
    checkpoint_step_001000.pt
    checkpoint_final.pt
    final_lora.safetensors      # (if adapter=lora)
    final_lycoris.safetensors   # (if adapter=lycoris)
  tb/                 # TensorBoard logs
  samples/            # Individual validation images
    step_000500_0.png
    step_000500_1.png
    step_000500_mountain.png    # (if name specified in prompt)
    step_001000_0.png
    step_001000_1.png

## Resuming Training

You can continue a run from any saved checkpoint:

```bash
python -m lora_trainer.cli \
  --checkpoint stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data /path/to/training/images \
  --steps 5000 \
  --batch_size 4 \
  --workspace ./runs/my_experiment \
  --resume_from ./runs/my_experiment/checkpoints/step_002500.pt
```

- Point `--resume_from` at a specific `.pt` file, or at the `checkpoints/` directory to automatically use the most recent checkpoint.
- Model and optimizer states are restored and training continues from the saved `global_step`. If the checkpoint step is already >= target `--steps`, the run exits.
- Validation sampling will also run at step 0 when prompts are provided, so you get a before/after comparison.
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=lora_trainer --cov-report=html
```

### Code Formatting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/
```

## Features

- **SDXL Training**: Focused support for Stable Diffusion XL LoRA training.
- **LyCORIS Support**: Native integration for LoHa, LoKr, and other LyCORIS adapters.
- **Clean CLI**: Familiar flags for ComfyUI users (`--sampler`, `--scheduler`, `--cfg`, etc.).
- **Aspect Bucketing**: Automatic aspect-ratio bucketing for high-quality training on mixed-resolution datasets.
- **Fast Feedback**: Real-time progress bars and validation sample generation during training.
- **Performance Optimized**: Supports gradient checkpointing and 8-bit optimizers for low-VRAM environments.
- **Flexible Data**: Support for both raw images and pre-cached latents/embeddings for maximum throughput.
- **TensorBoard Integration**: Detailed logging of loss, learning rate, and validation samples.

## Contributing

This repository follows strict principles:

1. **No dead code**: When design changes, old code is deleted immediately.
2. **Test-driven**: All features are accompanied by comprehensive tests.
3. **Single responsibility**: This tool focuses exclusively on SDXL LoRA/LyCORIS training.
4. **UX-first**: Prioritizes clear flags, sane defaults, and helpful feedback.

To contribute:
1. Ensure all tests pass: `pytest`
2. Follow the established style: `black`, `isort`, and `ruff`
3. Add tests for any new functionality.

## License

[Add your license here]
