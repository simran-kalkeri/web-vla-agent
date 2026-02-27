"""
Configuration system for VLA Web Agent (Multimodal Sequential VLM).

Loads defaults from dataclasses, optionally overridden by YAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    # QLoRA
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    quantization_bits: int = 4
    action_types: List[str] = field(
        default_factory=lambda: ["CLICK", "TYPE", "SELECT", "SCROLL"]
    )
    # Image resolution limits for Qwen2-VL processor
    image_min_pixels: int = 256 * 28 * 28    # 200704
    image_max_pixels: int = 1280 * 28 * 28   # 1003520


@dataclass
class TrainingConfig:
    stage1_epochs: int = 5
    stage2_epochs: int = 10
    stage3_epochs: int = 0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "checkpoints"
    logging_steps: int = 10
    max_seq_length: int = 4096


@dataclass
class DataConfig:
    dataset_name: str = "osunlp/Multimodal-Mind2Web"
    max_dom_nodes: int = 500
    max_action_history: int = 10
    max_text_per_node: int = 200


@dataclass
class EnvironmentConfig:
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_ms: int = 30000
    max_steps: int = 30


@dataclass
class UncertaintyConfig:
    min_log_prob_threshold: float = -2.0
    beam_width: int = 3
    max_regenerations: int = 2


@dataclass
class EvaluationConfig:
    splits: List[str] = field(
        default_factory=lambda: ["test_task", "test_website", "test_domain"]
    )


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "web-vla-agent"


@dataclass
class VLAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


_CONFIG_TYPES = (
    ModelConfig, TrainingConfig, DataConfig, EnvironmentConfig,
    UncertaintyConfig, EvaluationConfig, LoggingConfig,
)


def _update_dataclass(dc: object, overrides: dict) -> None:
    """Recursively update a dataclass instance from a dict."""
    for key, value in overrides.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(current, _CONFIG_TYPES):
            _update_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(yaml_path: Optional[str] = None) -> VLAConfig:
    """Create VLAConfig with defaults, optionally overlay from YAML."""
    cfg = VLAConfig()

    if yaml_path is None:
        yaml_path = os.environ.get("VLA_CONFIG")
    if yaml_path is None:
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / "configs" / "default.yaml"
        if candidate.exists():
            yaml_path = str(candidate)

    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path, "r") as fh:
            overrides = yaml.safe_load(fh) or {}
        _update_dataclass(cfg, overrides)

    return cfg
