#!/usr/bin/env bash

set -x

python -m lora_trainer.cli "${@:3}" \
  --checkpoint "$HOME/workspace/ComfyUI/models/checkpoints/$1.safetensors" \
  --train_data "$HOME/workspace/$2/training_data" \
  --steps 10 \
  --batch_size 1 \
  --workspace ./runs/test_run \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 8.0 \
  --image_size 1024 \
  --num_workers 0 \
  --mixed_precision fp16
