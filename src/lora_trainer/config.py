"""Configuration dataclasses and validation for SDXL LoRA training."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Complete training configuration for SDXL LoRA training.

    All CLI arguments are parsed into this dataclass, with validation
    and computed properties for training loop control.
    """

    # Required parameters
    checkpoint: str
    train_data: Path
    steps: int
    batch_size: int
    workspace: Path

    # Optimizer parameters
    learning_rate: float = 1e-4
    grad_accum: int = 1
    optimizer: str = "adamw"

    # Data parameters
    image_size: int = 1024  # SDXL native resolution
    num_workers: int = 4

    # Sampling/validation parameters
    scheduler: str = "normal"
    sampler: str = "euler"
    cfg: float = 7.0
    sampler_steps: int = 30
    sample_prompts: Path | None = None
    sample_every: int = 500
    samples_per_prompt: int = 1

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 16.0

    # Misc
    seed: int = 42
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    resume_from: Path | None = None  # Optional checkpoint path/dir to resume from

    # Internal fields computed after init
    num_images: int = field(init=False, default=0)
    steps_per_epoch: int = field(init=False, default=0)
    num_epochs: int = field(init=False, default=0)
    effective_batch_size: int = field(init=False, default=0)

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        # Convert string paths to Path objects
        self.train_data = Path(self.train_data)
        self.workspace = Path(self.workspace)
        if self.sample_prompts is not None:
            self.sample_prompts = Path(self.sample_prompts)
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)

        # Validate required parameters
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.grad_accum <= 0:
            raise ValueError(f"grad_accum must be positive, got {self.grad_accum}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        # Validate paths
        if not self.train_data.exists():
            raise ValueError(f"train_data path does not exist: {self.train_data}")
        if not self.train_data.is_dir():
            raise ValueError(f"train_data must be a directory: {self.train_data}")

        if self.sample_prompts is not None and not self.sample_prompts.exists():
            raise ValueError(f"sample_prompts file does not exist: {self.sample_prompts}")
        if self.resume_from is not None and not self.resume_from.exists():
            raise ValueError(f"resume_from path does not exist: {self.resume_from}")

        # Validate sampling parameters
        if self.cfg < 0:
            raise ValueError(f"cfg must be non-negative, got {self.cfg}")
        if self.sampler_steps <= 0:
            raise ValueError(f"sampler_steps must be positive, got {self.sampler_steps}")
        if self.sample_every <= 0:
            raise ValueError(f"sample_every must be positive, got {self.sample_every}")
        if self.samples_per_prompt <= 0:
            raise ValueError(f"samples_per_prompt must be positive, got {self.samples_per_prompt}")

        # Validate LoRA parameters
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")

        # Validate image size
        if self.image_size <= 0 or self.image_size % 8 != 0:
            raise ValueError(
                f"image_size must be positive and divisible by 8, got {self.image_size}"
            )

        # Compute effective batch size
        self.effective_batch_size = self.batch_size * self.grad_accum

        # Count images in train_data directory
        self.num_images = self._count_images()

        # Compute training schedule
        if self.num_images > 0:
            self.steps_per_epoch = max(1, self.num_images // self.effective_batch_size)
            self.num_epochs = max(
                1, (self.steps + self.steps_per_epoch - 1) // self.steps_per_epoch
            )
        else:
            # Allow zero images for testing purposes
            self.steps_per_epoch = 1
            self.num_epochs = 1

    def _count_images(self) -> int:
        """Count image files in the train_data directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        count = 0
        for path in self.train_data.iterdir():
            if path.suffix.lower() in image_extensions:
                count += 1
        return count

    def print_summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        lines = [
            "=" * 60,
            "Training Configuration",
            "=" * 60,
            f"Checkpoint:          {self.checkpoint}",
            f"Train Data:          {self.train_data}",
            f"Workspace:           {self.workspace}",
            "",
            f"Images:              {self.num_images}",
            f"Batch Size:          {self.batch_size}",
            f"Grad Accumulation:   {self.grad_accum}",
            f"Effective Batch:     {self.effective_batch_size}",
            f"Steps per Epoch:     {self.steps_per_epoch}",
            f"Total Steps:         {self.steps}",
            f"Total Epochs:        {self.num_epochs}",
            f"Learning Rate:       {self.learning_rate}",
            f"Optimizer:           {self.optimizer}",
            "",
            f"LoRA Rank:           {self.lora_rank}",
            f"LoRA Alpha:          {self.lora_alpha}",
            "",
            f"Image Size:          {self.image_size}x{self.image_size}",
            f"Mixed Precision:     {self.mixed_precision}",
            f"Seed:                {self.seed}",
            f"Resume From:         {self.resume_from or 'None'}",
            "",
            f"Scheduler:           {self.scheduler}",
            f"Sampler:             {self.sampler}",
            f"CFG Scale:           {self.cfg}",
            f"Sampler Steps:       {self.sampler_steps}",
            f"Sample Every:        {self.sample_every} steps",
            f"Samples per Prompt:  {self.samples_per_prompt}",
            "=" * 60,
        ]
        return "\n".join(lines)
