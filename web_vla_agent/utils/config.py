"""
Structured configuration system for the VLA Web Agent (Refactored).

Loads defaults from dataclasses, optionally overridden by a YAML file.
All hyper-parameters are centralised here so every module imports from one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


# ── Sub-configs ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    vision_encoder: str = "openai/clip-vit-base-patch32"
    text_dim: int = 384
    vision_dim: int = 512
    hidden_dim: int = 256
    num_action_types: int = 4


@dataclass
class GraphConfig:
    """1-layer GCN configuration."""
    gcn_layers: int = 1
    structural_dim: int = 36    # tag one-hot (31) + 5 structural features
    dropout: float = 0.1


@dataclass
class GrounderConfig:
    """Single cross-attention block grounder configuration."""
    cross_attention_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1
    contrastive_temperature: float = 0.07


@dataclass
class UncertaintyConfig:
    """Entropy-only uncertainty configuration."""
    entropy_threshold: float = 1.5   # Auto-calibrated from validation
    temperature: float = 0.07


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 50
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    # 3-term loss: L = λ₁·L_CE_elem + λ₂·L_InfoNCE + λ₃·L_CE_action
    lambda_element: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_action: float = 1.0
    # Focal loss for action collapse fix
    focal_gamma: float = 2.0
    # 3-phase schedule
    phase1_epochs: int = 10
    phase2_epochs: int = 30
    phase1_element_acc_threshold: float = 0.30
    phase2_element_acc_threshold: float = 0.50
    phase2_action_acc_threshold: float = 0.70
    save_every_n_epochs: int = 5
    checkpoint_dir: str = "checkpoints"
    seed: int = 42


@dataclass
class EvaluationConfig:
    long_horizon_threshold: int = 5
    splits: List[str] = field(
        default_factory=lambda: ["test_task", "test_website", "test_domain"]
    )


@dataclass
class PlannerConfig:
    """Planner config — used in evaluation only, NOT during training."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_subgoals: int = 10
    max_retries: int = 3
    temperature: float = 0.3
    max_new_tokens: int = 1024


@dataclass
class MemoryConfig:
    short_term_capacity: int = 5
    max_retries_per_subgoal: int = 3
    loop_detection_window: int = 4
    stale_state_threshold: int = 3


@dataclass
class EnvironmentConfig:
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_ms: int = 30000
    screenshot_width: int = 1280
    screenshot_height: int = 720


@dataclass
class DataConfig:
    dataset_name: str = "osunlp/Multimodal-Mind2Web"
    max_dom_elements: int = 128
    max_action_history: int = 5
    screenshot_size: List[int] = field(default_factory=lambda: [224, 224])
    html_max_tokens: int = 1024


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "web-vla-agent"


# ── Root config ──────────────────────────────────────────────

@dataclass
class VLAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    grounder: GrounderConfig = field(default_factory=GrounderConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── Helpers ──────────────────────────────────────────────────

_CONFIG_TYPES = (
    ModelConfig, GraphConfig, GrounderConfig, UncertaintyConfig,
    PlannerConfig, TrainingConfig, EvaluationConfig,
    MemoryConfig, EnvironmentConfig, DataConfig, LoggingConfig,
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
    """
    Create a VLAConfig with defaults, optionally overlay values from YAML.
    """
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
