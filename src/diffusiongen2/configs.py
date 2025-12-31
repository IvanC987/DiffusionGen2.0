from dataclasses import dataclass
from omegaconf import MISSING
from typing import Literal


@dataclass
class UNetConfig:
    in_channels: int = 4
    channels: tuple = MISSING
    n_bn: int = MISSING
    n_groups: int = MISSING
    T: int = 1000
    t_embd: int = MISSING
    t_embd_scale: int = MISSING
    n_embd: int = MISSING
    n_heads: int = MISSING
    n_layers: int = MISSING
    dropout: float = MISSING
    enable_flash_attention: bool = MISSING


@dataclass
class DiffusionConfig:
    schedule: str = MISSING
    T: int = 1000  # Repeated, but will assert to ensure equal to UNetConfig's
    beta1: float = 0.00085
    beta2: float = 0.0120
    s: float = 0.008


@dataclass
class DataConfig:
    shard_dir: str = MISSING
    val_split: float = MISSING
    vae_model_id: str = "stabilityai/sd-vae-ft-ema"


@dataclass
class TrainingConfig:
    run_name: str = MISSING

    effective_batch_size: int = 128  # [64, 128, 256]
    batch_size: int = 64

    compile_model: bool = MISSING

    epochs: int = MISSING
    ema_beta: float = 0.95  # 90% of the weights would be at the 45 most recent steps for beta=0.95. (0.96, 56), (0.97, 76)
    cfg_dropout: float = 0.1
    seed: int = 89

    ckpt_interval: int = MISSING


@dataclass
class OptimConfig:
    max_lr: float = MISSING
    min_lr: float = MISSING
    grad_clip: float = 1.0
    weight_decay: float = 0.01  # GPT recommends smaller, Gemini says larger. Adjust accordingly.


@dataclass
class EvalConfig:
    prompts_path: str = MISSING     # .pt file where it contains prompts, prompt embd, and null embd for evaluation
    cfg_scales: tuple = MISSING     # Default scales to test

    log_every_n_steps: int = 16     # Frequency for logging training loss to console
    val_every_n_epochs: int = 1     # Frequency for running validation
    sample_every_n_epochs: int = 5  # Frequency for image gen (can be expensive and doesn't change much over later epochs)



@dataclass
class ProjectConfig:
    unet: UNetConfig = MISSING
    diffusion: DiffusionConfig = MISSING
    data: DataConfig = MISSING
    training: TrainingConfig = MISSING
    optim: OptimConfig = MISSING
    eval: EvalConfig = MISSING
