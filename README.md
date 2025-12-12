# SDXL LoRA Trainer

Single-purpose, high-UX SDXL LoRA training toolkit with a CLI-first design.

## Overview

This is a focused SDXL LoRA trainer designed for ease of use with ComfyUI-style parameters. It provides:

- Clean CLI interface with familiar flags
- Fast feedback via tqdm progress bars and TensorBoard
- Periodic validation sample generation
- Well-tested, modular codebase

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd lora-trainer

# Install in development mode
pip install -e .

# Or install with dev dependencies
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
  --sample_prompts prompts.txt \
  --sample_every 500 \
  --scheduler karras \
  --sampler euler_ancestral \
  --cfg 7.0 \
  --sampler_steps 30 \
  --samples_per_prompt 2
```

Create `prompts.txt` with one prompt per line:

```
a photo of a mountain landscape
a portrait of a person
a cute cat
```

### All CLI Options

**Optimizer:**
- `--learning_rate`: Learning rate (default: 1e-4)
- `--grad_accum`: Gradient accumulation steps (default: 1)

**Data:**
- `--image_size`: Image size for training (default: 1024)
- `--num_workers`: Data loading workers (default: 4)

**Sampling:**
- `--scheduler`: Noise scheduler - `simple`, `normal`, `karras` (default: normal)
- `--sampler`: Sampler algorithm - `euler`, `euler_ancestral`, `ddim`, `heun` (default: euler)
- `--cfg`: Classifier-free guidance scale (default: 7.0)
- `--sampler_steps`: Diffusion steps for sampling (default: 30)
- `--sample_prompts`: Path to prompts file
- `--sample_every`: Generate samples every N steps (default: 500)
- `--samples_per_prompt`: Number of samples per prompt (default: 1)

**LoRA:**
- `--lora_rank`: Rank of LoRA matrices (default: 16)
- `--lora_alpha`: LoRA alpha scaling (default: 16.0)

**Misc:**
- `--seed`: Random seed (default: 42)
- `--mixed_precision`: Mixed precision mode - `no`, `fp16`, `bf16` (default: fp16)

## Monitoring Training

### TensorBoard

View training metrics in real-time:

```bash
tensorboard --logdir ./runs/my_experiment/tb
```

### Output Structure

Your workspace will contain:

```
./runs/my_experiment/
  checkpoints/        # Model checkpoints
    step_000500.pt
    step_001000.pt
    final.pt
  tb/                 # TensorBoard logs
  samples/            # Validation images
    step_000500.png
    step_001000.png
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

## Current Status

**Phase 1: Complete** ✓
- Core training infrastructure
- CLI with argument parsing
- Dataset and dataloader
- Training loop with checkpointing
- TensorBoard logging
- Comprehensive test suite

**Phase 2: In Progress**
- Real SDXL UNet integration
- LoRA module injection
- Diffusion loss implementation

**Phase 3: Planned**
- Validation sampling
- Scheduler & sampler integration
- Sample image generation

**Phase 4: Planned**
- Performance optimizations
- Enhanced logging
- Additional UX improvements

## Contributing

This repository follows strict principles:

1. No dead code - when design changes, old code is deleted
2. Test-driven development - tests accompany all features
3. Single responsibility - this tool only trains SDXL LoRA
4. UX-first design - clear flags, good defaults, obvious behavior

## License

[Add your license here]
