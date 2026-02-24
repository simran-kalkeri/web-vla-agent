"""
Evaluation Metrics — Publication-Grade (Refactored).

Required metrics:
  - element_accuracy
  - recall@K
  - mean_true_rank
  - action_accuracy
  - macro F1 (action types)
  - long_horizon_success_rate
  - entropy_distribution (mean, std, percentiles)
  - replan_fraction
  - per-domain generalization
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StepResult:
    """Result of a single predicted step."""
    predicted_action: str = ""
    true_action: str = ""
    predicted_element: str = ""
    true_element: str = ""
    action_correct: bool = False
    element_correct: bool = False
    step_correct: bool = False
    # Grounder diagnostics
    num_candidates: int = 0
    true_element_in_top_k: bool = False
    true_element_rank: int = -1
    # Entropy diagnostics
    predictive_entropy: float = 0.0


@dataclass
class TaskResult:
    """Result of a full task execution."""
    task_id: str = ""
    task_description: str = ""
    website: str = ""
    domain: str = ""
    success: bool = False
    num_steps: int = 0
    step_results: List[StepResult] = field(default_factory=list)
    # Entropy tracking
    mean_entropy: float = 0.0
    replan_triggered: bool = False


class MetricsTracker:
    """
    Accumulate per-task results and compute publication-grade metrics.

    Reports:
      - element_accuracy, recall@K, mean_true_rank
      - action_accuracy, macro F1 (per action type)
      - long_horizon_success_rate
      - entropy distribution stats
      - replan_fraction
      - per-domain success rates
    """

    def __init__(self, long_horizon_threshold: int = 5):
        self.long_horizon_threshold = long_horizon_threshold
        self._tasks: List[TaskResult] = []

    def add_task(self, result: TaskResult) -> None:
        self._tasks.append(result)

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics over accumulated results."""
        if not self._tasks:
            return {}

        metrics: Dict[str, float] = {}
        all_steps = [s for t in self._tasks for s in t.step_results]

        # ── Core metrics ─────────────────────────────────────
        if all_steps:
            metrics["element_accuracy"] = self._mean([s.element_correct for s in all_steps])
            metrics["action_accuracy"] = self._mean([s.action_correct for s in all_steps])
            metrics["step_accuracy"] = self._mean([s.step_correct for s in all_steps])

        # ── Recall@K and Mean True Rank ─────────────────────
        steps_with_cands = [s for s in all_steps if s.num_candidates > 0]
        if steps_with_cands:
            metrics["recall_at_k"] = self._mean([s.true_element_in_top_k for s in steps_with_cands])
            valid_ranks = [s.true_element_rank for s in steps_with_cands if s.true_element_rank >= 0]
            metrics["mean_true_rank"] = sum(valid_ranks) / len(valid_ranks) if valid_ranks else -1

        # ── Success rates ────────────────────────────────────
        metrics["success_rate"] = self._mean([t.success for t in self._tasks])

        long_tasks = [t for t in self._tasks if t.num_steps >= self.long_horizon_threshold]
        metrics["long_horizon_success_rate"] = (
            self._mean([t.success for t in long_tasks]) if long_tasks else 0.0
        )
        metrics["long_horizon_count"] = float(len(long_tasks))

        # ── Entropy distribution ─────────────────────────────
        all_entropies = [s.predictive_entropy for s in all_steps if s.predictive_entropy > 0]
        if all_entropies:
            import torch
            ent_t = torch.tensor(all_entropies)
            metrics["entropy_mean"] = ent_t.mean().item()
            metrics["entropy_std"] = ent_t.std().item()
            metrics["entropy_p50"] = torch.quantile(ent_t, 0.5).item()
            metrics["entropy_p90"] = torch.quantile(ent_t, 0.9).item()
            metrics["entropy_p95"] = torch.quantile(ent_t, 0.95).item()

        # ── Replan fraction ──────────────────────────────────
        metrics["replan_fraction"] = (
            self._mean([t.replan_triggered for t in self._tasks])
        )

        # ── Per-domain generalization ────────────────────────
        domain_tasks = defaultdict(list)
        for t in self._tasks:
            domain_tasks[t.domain or "unknown"].append(t)
        for domain, tasks in domain_tasks.items():
            metrics[f"success/{domain}"] = self._mean([t.success for t in tasks])

        # ── Action Type F1 ──────────────────────────────────
        metrics.update(self._action_type_f1(all_steps))

        return metrics

    # ── Per-Action F1 ────────────────────────────────────────

    @staticmethod
    def _action_type_f1(steps: List[StepResult]) -> Dict[str, float]:
        if not steps:
            return {}
        types = set()
        for s in steps:
            types.add(s.true_action)
            types.add(s.predicted_action)
        types.discard("")

        f1_per_type: Dict[str, float] = {}
        for at in types:
            tp = sum(1 for s in steps if s.predicted_action == at and s.true_action == at)
            fp = sum(1 for s in steps if s.predicted_action == at and s.true_action != at)
            fn = sum(1 for s in steps if s.predicted_action != at and s.true_action == at)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_per_type[f"f1/{at}"] = f1

        if f1_per_type:
            f1_per_type["f1/macro"] = sum(f1_per_type.values()) / len(f1_per_type)
        return f1_per_type

    @staticmethod
    def _mean(values: list) -> float:
        if not values:
            return 0.0
        return sum(float(v) for v in values) / len(values)

    # ── Reporting ────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary table."""
        metrics = self.compute()
        lines = [
            "=" * 65,
            "  VLA Web Agent — Evaluation Results",
            "=" * 65,
        ]
        groups = {
            "Grounding": ["element_accuracy", "recall_at_k", "mean_true_rank"],
            "Action": ["action_accuracy", "step_accuracy"],
            "Success": ["success_rate", "long_horizon_success_rate", "long_horizon_count"],
            "Entropy": ["entropy_mean", "entropy_std", "entropy_p50", "entropy_p90",
                        "entropy_p95", "replan_fraction"],
        }
        for group_name, keys in groups.items():
            lines.append(f"\n  ── {group_name} ──")
            for k in keys:
                if k in metrics:
                    v = metrics[k]
                    lines.append(f"    {k:40s} {v:.4f}" if isinstance(v, float) else f"    {k:40s} {v}")

        # Domain breakdown
        domain_keys = [k for k in sorted(metrics.keys()) if k.startswith("success/")]
        if domain_keys:
            lines.append(f"\n  ── Domain Generalization ──")
            for k in domain_keys:
                lines.append(f"    {k:40s} {metrics[k]:.4f}")

        # F1
        f1_keys = [k for k in sorted(metrics.keys()) if k.startswith("f1/")]
        if f1_keys:
            lines.append(f"\n  ── Action F1 ──")
            for k in f1_keys:
                lines.append(f"    {k:40s} {metrics[k]:.4f}")

        lines.append("=" * 65)
        return "\n".join(lines)
