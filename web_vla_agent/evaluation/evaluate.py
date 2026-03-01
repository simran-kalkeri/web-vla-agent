"""
Evaluation Module for VLA Web Agent.

Implements all required metrics:
  - Element grounding accuracy
  - Action accuracy (full JSON match)
  - Task completion rate
  - Mean steps to completion
  - Failure mode breakdown
  - Per-action F1
  - Inference latency measurement
"""
from __future__ import annotations

import json
import logging
import os
import time

# Reduce CUDA memory fragmentation (recommended by PyTorch for large models)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single step evaluation."""
    predicted_action: Dict[str, Any] = field(default_factory=dict)
    ground_truth_action: Dict[str, Any] = field(default_factory=dict)
    action_correct: bool = False
    element_correct: bool = False
    value_correct: bool = False
    full_match: bool = False
    latency_ms: float = 0.0
    log_prob: float = 0.0
    parse_success: bool = True


@dataclass
class TaskResult:
    """Result of a full task evaluation."""
    task_id: str = ""
    task: str = ""
    steps: List[StepResult] = field(default_factory=list)
    completed: bool = False
    num_steps: int = 0
    total_latency_ms: float = 0.0
    failure_reason: str = ""


class VLAEvaluator:
    """
    Evaluate VLA Web Agent on Mind2Web test sets.

    Computes all required metrics and produces detailed reports.
    """

    def __init__(self, action_types: Optional[List[str]] = None):
        self.action_types = action_types or ["CLICK", "TYPE", "SELECT", "SCROLL"]
        self.results: List[TaskResult] = []
        self.step_results: List[StepResult] = []

    def evaluate_step(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any],
        latency_ms: float = 0.0,
        log_prob: float = 0.0,
    ) -> StepResult:
        """
        Evaluate a single predicted action against ground truth.
        """
        pred_action = predicted.get("action", "").upper()
        gt_action = ground_truth.get("action", "").upper()

        action_correct = pred_action == gt_action

        pred_eid = predicted.get("element_id")
        gt_eid = ground_truth.get("element_id")
        element_correct = pred_eid == gt_eid

        pred_value = str(predicted.get("value", "")).strip().lower()
        gt_value = str(ground_truth.get("value", "")).strip().lower()
        value_correct = pred_value == gt_value if gt_value else True

        full_match = action_correct and element_correct and value_correct

        result = StepResult(
            predicted_action=predicted,
            ground_truth_action=ground_truth,
            action_correct=action_correct,
            element_correct=element_correct,
            value_correct=value_correct,
            full_match=full_match,
            latency_ms=latency_ms,
            log_prob=log_prob,
        )
        self.step_results.append(result)
        return result

    def evaluate_batch(
        self,
        model,
        samples: list,
        prompt_builder,
        action_decoder,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a batch of samples.

        Parameters
        ----------
        model : VLAModel
        samples : list of Mind2WebSample
        prompt_builder : PromptBuilder
        action_decoder : ActionDecoder
        max_samples : int, optional
        """
        eval_samples = samples[:max_samples] if max_samples else samples

        import torch

        for i, sample in enumerate(eval_samples):
            try:
                # Build prompt
                messages = prompt_builder.build_chat_messages(
                    task=sample.task,
                    serialized_dom=sample.serialized_dom,
                    action_history=sample.action_history,
                    screenshot_placeholder=sample.screenshot is not None,
                    extra_context=f"Website: {sample.website}" if sample.website else "",
                )

                # Generate with timing
                start = time.time()
                result = model.generate(
                    messages=messages,
                    image=sample.screenshot,
                    return_log_probs=True,
                )
                latency_ms = (time.time() - start) * 1000

                # Parse prediction
                pred_action = action_decoder.parse(result["text"])
                if pred_action is None:
                    pred_action = {"action": "UNKNOWN"}

                # Evaluate
                self.evaluate_step(
                    predicted=pred_action,
                    ground_truth=sample.action,
                    latency_ms=latency_ms,
                    log_prob=result.get("avg_log_prob", 0.0),
                )

            except Exception as e:
                logger.warning(f"Eval failed for sample {i}: {e}")
                self.step_results.append(StepResult(
                    ground_truth_action=sample.action,
                    parse_success=False,
                ))
            finally:
                # Free any cached activations / KV-cache from this sample
                torch.cuda.empty_cache()

            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(eval_samples)}")

        return self.compute_metrics()

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Returns dict with:
          - element_accuracy
          - action_accuracy
          - full_match_accuracy
          - per_action_f1
          - mean_latency_ms
          - failure_breakdown
          - parse_success_rate
        """
        if not self.step_results:
            return {}

        total = len(self.step_results)
        valid = [r for r in self.step_results if r.parse_success]

        metrics: Dict[str, Any] = {
            "total_samples": total,
            "valid_samples": len(valid),
            "parse_success_rate": len(valid) / total,
        }

        if valid:
            metrics["element_accuracy"] = sum(1 for r in valid if r.element_correct) / len(valid)
            metrics["action_accuracy"] = sum(1 for r in valid if r.action_correct) / len(valid)
            metrics["value_accuracy"] = sum(1 for r in valid if r.value_correct) / len(valid)
            metrics["full_match_accuracy"] = sum(1 for r in valid if r.full_match) / len(valid)

            latencies = [r.latency_ms for r in valid if r.latency_ms > 0]
            if latencies:
                metrics["mean_latency_ms"] = sum(latencies) / len(latencies)
                metrics["p95_latency_ms"] = sorted(latencies)[int(len(latencies) * 0.95)]

            log_probs = [r.log_prob for r in valid if r.log_prob != 0]
            if log_probs:
                metrics["mean_log_prob"] = sum(log_probs) / len(log_probs)

        # Per-action metrics
        metrics["per_action"] = self._per_action_metrics(valid)

        # Failure breakdown
        metrics["failure_breakdown"] = self._failure_breakdown(valid)

        return metrics

    def _per_action_metrics(self, results: List[StepResult]) -> Dict[str, Dict[str, float]]:
        """Compute per-action-type precision, recall, and F1."""
        per_action: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for r in results:
            gt_action = r.ground_truth_action.get("action", "").upper()
            pred_action = r.predicted_action.get("action", "").upper()

            if pred_action == gt_action:
                per_action[gt_action]["tp"] += 1
            else:
                per_action[gt_action]["fn"] += 1
                per_action[pred_action]["fp"] += 1

        metrics = {}
        for action_type in self.action_types + ["UNKNOWN", "DONE"]:
            counts = per_action.get(action_type, {"tp": 0, "fp": 0, "fn": 0})
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if tp + fp + fn > 0:
                metrics[action_type] = {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "support": tp + fn,
                }

        return metrics

    def _failure_breakdown(self, results: List[StepResult]) -> Dict[str, int]:
        """Breakdown of failure types."""
        failures = defaultdict(int)
        for r in results:
            if r.full_match:
                continue
            if not r.action_correct:
                failures["wrong_action_type"] += 1
            elif not r.element_correct:
                failures["wrong_element"] += 1
            elif not r.value_correct:
                failures["wrong_value"] += 1
        failures["total_failures"] = sum(failures.values())
        return dict(failures)

    def reset(self):
        """Reset all stored results."""
        self.results.clear()
        self.step_results.clear()

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """Print a human-readable evaluation report."""
        print("\n" + "=" * 60)
        print("  VLA Web Agent — Evaluation Report")
        print("=" * 60)

        print(f"\n  Total samples:      {metrics.get('total_samples', 0)}")
        print(f"  Parse success rate: {metrics.get('parse_success_rate', 0):.1%}")
        print(f"\n  Element accuracy:   {metrics.get('element_accuracy', 0):.1%}")
        print(f"  Action accuracy:    {metrics.get('action_accuracy', 0):.1%}")
        print(f"  Value accuracy:     {metrics.get('value_accuracy', 0):.1%}")
        print(f"  Full match:         {metrics.get('full_match_accuracy', 0):.1%}")

        if "mean_latency_ms" in metrics:
            print(f"\n  Mean latency:       {metrics['mean_latency_ms']:.0f}ms")
            print(f"  P95 latency:        {metrics.get('p95_latency_ms', 0):.0f}ms")

        if "per_action" in metrics:
            print("\n  Per-Action F1:")
            for action, m in metrics["per_action"].items():
                print(f"    {action:10s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  N={m['support']}")

        if "failure_breakdown" in metrics:
            print("\n  Failure Breakdown:")
            for ftype, count in metrics["failure_breakdown"].items():
                print(f"    {ftype:25s}: {count}")

        print("\n" + "=" * 60)


def main():
    """Main evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VLA Web Agent Evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test_task")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load model in 4-bit QLoRA (default: True). "
            "MUST match training setup — LoRA adapters trained on 4-bit base weights "
            "produce NaN/inf logits when run on bfloat16 base weights. "
            "4-bit automatically pins to a single GPU to avoid a PEFT multi-GPU bug. "
            "Pass --no-load-in-4bit for bfloat16 multi-GPU (only if LoRA was NOT QLoRA-trained)."
        ),
    )
    args = parser.parse_args()

    from utils.config import load_config
    from utils.logging import get_logger
    from models.vla_model import VLAModel
    from models.prompt_builder import PromptBuilder
    from models.action_decoder import ActionDecoder
    from data.mind2web_loader import Mind2WebLoader

    config = load_config(args.config)
    log = get_logger("vla.eval")

    # Load model.
    # Multi-GPU (device_map="auto"): use bfloat16 (load_in_4bit=False).
    #   4-bit + PEFT + multi-GPU triggers a CUDA illegal memory access in
    #   bitsandbytes when PEFT injects LoRA adapters across GPU shards.
    #   With 2×47 GB GPUs there is plenty of VRAM for bfloat16.
    # Single-GPU: pass --load-in-4bit if VRAM is tight.
    model = VLAModel(config=config, device=args.device, load_in_4bit=args.load_in_4bit)
    model.load()
    if args.checkpoint:
        model.load_lora(args.checkpoint)

    # Load test data
    loader = Mind2WebLoader(dataset_name=config.data.dataset_name)
    test_samples = loader.build_training_examples(
        split=args.split,
        max_samples=args.max_samples,
        include_screenshot=True,
    )
    log.info(f"Loaded {len(test_samples)} test samples from {args.split}")

    # Evaluate
    evaluator = VLAEvaluator()
    prompt_builder = PromptBuilder()
    decoder = ActionDecoder()

    metrics = evaluator.evaluate_batch(
        model=model,
        samples=test_samples,
        prompt_builder=prompt_builder,
        action_decoder=decoder,
        max_samples=args.max_samples,
    )

    evaluator.print_report(metrics)

    # Save results
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
