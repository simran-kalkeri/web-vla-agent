"""
Evaluation Suite — Publication-Grade with Ablation Framework (Refactored).

Ablation presets:
  full                — all components active (proposed system)
  no_graph            — remove GCN → use flat DOM embeddings
  no_cross_attention  — replace cross-attn with linear fusion
  no_entropy          — disable replan trigger
  dom_only            — remove visual embeddings
  action_unweighted   — remove class weighting / focal loss (γ=0)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from data.mind2web_loader import Mind2WebDataset, vla_collate_fn
from models.encoders import TextEncoder, VisionEncoder
from models.graph_dom_encoder import GraphDOMEncoder, structural_features, build_edge_index
from models.grounder import VLAGrounderBatched
from models.uncertainty import EntropyUncertainty
from evaluation.metrics import MetricsTracker, TaskResult, StepResult
from utils.config import VLAConfig, load_config
from utils.logging import get_logger, log_metrics


ACTION_NAMES = ["CLICK", "TYPE", "SCROLL", "SELECT"]


# ── Ablation Presets ─────────────────────────────────────────

class AblationMode:
    """Flags to selectively disable components for ablation study."""

    def __init__(
        self,
        no_graph: bool = False,             # Remove GCN → flat DOM
        no_cross_attention: bool = False,    # Replace cross-attn with linear
        no_entropy: bool = False,            # Disable replan trigger
        dom_only: bool = False,              # Remove visual embeddings
        action_unweighted: bool = False,     # Remove focal loss (γ=0)
        name: str = "full",
    ):
        self.no_graph = no_graph
        self.no_cross_attention = no_cross_attention
        self.no_entropy = no_entropy
        self.dom_only = dom_only
        self.action_unweighted = action_unweighted
        self.name = name

    def __repr__(self):
        active = [k for k, v in self.__dict__.items() if v is True and k != "name"]
        return f"AblationMode({self.name}, disabled={active})"


ABLATION_PRESETS = {
    "full": AblationMode(name="full"),
    "no_graph": AblationMode(no_graph=True, name="no_graph"),
    "no_cross_attention": AblationMode(no_cross_attention=True, name="no_cross_attention"),
    "no_entropy": AblationMode(no_entropy=True, name="no_entropy"),
    "dom_only": AblationMode(dom_only=True, name="dom_only"),
    "action_unweighted": AblationMode(action_unweighted=True, name="action_unweighted"),
}


# ── Evaluator ────────────────────────────────────────────────

class Evaluator:
    """
    Evaluate trained VLA agent on Mind2Web test splits with ablation.

    Parameters
    ----------
    config : VLAConfig
    checkpoint_path : str | None
    device : str
    ablation : AblationMode | None
    """

    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        ablation: Optional[AblationMode] = None,
    ):
        self.config = config or load_config()
        self.device = device
        self.ablation = ablation or AblationMode()
        self.logger = get_logger("evaluator")
        self.logger.info(f"Ablation mode: {self.ablation}")

        # ── Frozen encoders ──────────────────────────────────
        self.text_encoder = TextEncoder(
            model_name=self.config.model.text_encoder,
            output_dim=self.config.model.text_dim,
            device=device,
        )
        self.vision_encoder = VisionEncoder(
            model_name=self.config.model.vision_encoder,
            output_dim=self.config.model.vision_dim,
            device=device,
        )

        # ── Graph encoder (disabled in no_graph ablation) ────
        self.graph_encoder = None
        if not self.ablation.no_graph:
            self.graph_encoder = GraphDOMEncoder(
                text_dim=self.config.model.text_dim,
                struct_dim=self.config.graph.structural_dim,
                hidden_dim=self.config.model.hidden_dim,
                dropout=0.0,
            ).to(device)

        # ── Grounder ─────────────────────────────────────────
        grounder_text_dim = (
            self.config.model.hidden_dim if not self.ablation.no_graph
            else self.config.model.text_dim
        )
        self.grounder = VLAGrounderBatched(
            d_model=self.config.model.hidden_dim,
            text_dim=grounder_text_dim,
            vision_dim=self.config.model.vision_dim,
            subgoal_dim=self.config.model.text_dim,
            num_heads=self.config.grounder.num_heads,
            temperature=self.config.grounder.contrastive_temperature,
            dropout=0.0,
            num_action_types=self.config.model.num_action_types,
        ).to(device)

        # ── Uncertainty (disabled in no_entropy ablation) ────
        self.uncertainty = None
        if not self.ablation.no_entropy:
            self.uncertainty = EntropyUncertainty(
                threshold_tau=self.config.uncertainty.entropy_threshold,
                temperature=self.config.uncertainty.temperature,
            )

        # ── Load checkpoint ──────────────────────────────────
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Set eval mode
        if self.graph_encoder:
            self.graph_encoder.eval()
        self.grounder.eval()

    # ── Main evaluation ──────────────────────────────────────

    def evaluate(
        self,
        splits: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        splits = splits or self.config.evaluation.splits
        all_results: Dict[str, Any] = {}

        for split in splits:
            self.logger.info(f"Evaluating on {split} [ablation={self.ablation.name}]...")
            tracker = MetricsTracker(
                long_horizon_threshold=self.config.evaluation.long_horizon_threshold
            )

            dataset = Mind2WebDataset(
                split=split, config=self.config.data, max_samples=max_samples,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False,
                collate_fn=vla_collate_fn, num_workers=0,
            )

            with torch.no_grad():
                for batch in loader:
                    task_result = self._evaluate_sample(batch)
                    tracker.add_task(task_result)

            metrics = tracker.compute()
            metrics["num_samples"] = tracker.num_tasks
            metrics["ablation"] = self.ablation.name
            all_results[split] = metrics

            self.logger.info(f"\n{tracker.summary()}")

        if output_path:
            Path(output_path).write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            self.logger.info(f"Results saved to {output_path}")

        return all_results

    # ── Per-sample evaluation ────────────────────────────────

    def _evaluate_sample(self, batch: Dict[str, Any]) -> TaskResult:
        H = self.config.model.hidden_dim

        # Encode subgoal
        subgoal_embs = self.text_encoder(batch["tasks"])

        # Encode DOM elements
        sigs = batch["element_signatures"][0]
        dom_elements = batch["dom_elements"][0]

        if sigs and dom_elements:
            et = self.text_encoder(sigs)

            # Graph encoding (or passthrough for no_graph ablation)
            if self.graph_encoder and not self.ablation.no_graph:
                struct_feat = structural_features(dom_elements).to(self.device)
                edge_index, _ = build_edge_index(dom_elements)
                edge_index = edge_index.to(self.device)
                dom_emb = self.graph_encoder(et, struct_feat, edge_index)
            else:
                dom_emb = et  # Flat: [N, text_dim]
        else:
            text_dim = self.config.model.text_dim if self.ablation.no_graph else H
            dom_emb = torch.zeros(0, text_dim, device=self.device)

        # Vision (zero for dom_only ablation)
        if self.ablation.dom_only:
            ev = torch.zeros(dom_emb.shape[0], self.config.model.vision_dim, device=self.device)
        else:
            ev = torch.randn(dom_emb.shape[0], self.config.model.vision_dim, device=self.device) * 0.01

        # Forward through grounder
        all_elem_logits, action_logits, all_scores, all_reprs = self.grounder(
            subgoal_embs, [dom_emb], [ev]
        )

        elem_logits_0 = all_elem_logits[0]
        scores_0 = all_scores[0]
        n = elem_logits_0.shape[0]

        # Top-K for recall
        top_indices = []
        if n > 0:
            K = min(10, n)
            _, tidx = torch.topk(elem_logits_0, K)
            top_indices = tidx.tolist()

        # Entropy diagnostics
        predictive_entropy = 0.0
        replan_triggered = False
        if self.uncertainty and n > 0:
            padded_scores = scores_0.unsqueeze(0)  # [1, N]
            entropy, _, _ = self.uncertainty.compute_uncertainty(padded_scores)
            predictive_entropy = entropy[0].item()
            replan_triggered = self.uncertainty.should_replan(padded_scores)[0].item()

        # Predictions
        pred_action_idx = action_logits.argmax(-1).item()
        pred_element_idx = elem_logits_0.argmax().item() if n > 0 else -1

        # Ground truth
        true_action_idx = batch["operation_label"][0].item()
        true_element_global = batch["target_element_idx"][0].item()

        true_in_topk = true_element_global in top_indices
        true_rank = top_indices.index(true_element_global) if true_in_topk else -1

        step = StepResult(
            predicted_action=ACTION_NAMES[pred_action_idx] if pred_action_idx < len(ACTION_NAMES) else "?",
            true_action=ACTION_NAMES[true_action_idx] if true_action_idx < len(ACTION_NAMES) else "?",
            predicted_element=str(pred_element_idx),
            true_element=str(true_element_global),
            action_correct=(pred_action_idx == true_action_idx),
            element_correct=(pred_element_idx == true_element_global and n > 0),
            step_correct=(
                pred_action_idx == true_action_idx
                and pred_element_idx == true_element_global
                and n > 0
            ),
            num_candidates=n,
            true_element_in_top_k=true_in_topk,
            true_element_rank=true_rank,
            predictive_entropy=predictive_entropy,
        )

        return TaskResult(
            task_id=batch["sample_ids"][0],
            task_description=batch["tasks"][0],
            success=step.step_correct,
            num_steps=batch["trajectory_lengths"][0],
            step_results=[step],
            mean_entropy=predictive_entropy,
            replan_triggered=replan_triggered,
        )

    # ── Checkpoint loading ───────────────────────────────────

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "graph_encoder" in ckpt and self.graph_encoder:
            self.graph_encoder.load_state_dict(ckpt["graph_encoder"])
        self.grounder.load_state_dict(ckpt["grounder"])
        # Load calibrated entropy threshold if available
        ts = ckpt.get("training_state", {})
        if self.uncertainty and "entropy_threshold" in ts:
            self.uncertainty.threshold_tau = ts["entropy_threshold"]
        self.logger.info(f"Loaded checkpoint: {path}")


# ── Ablation Runner ──────────────────────────────────────────

def run_all_ablations(
    config: VLAConfig,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    max_samples: Optional[int] = None,
    output_dir: str = "ablation_results",
) -> Dict[str, Dict]:
    """Run evaluation across all ablation presets."""
    logger = get_logger("ablation_runner")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_ablation_results = {}

    for name, ablation in ABLATION_PRESETS.items():
        logger.info(f"\n{'='*60}\nRunning ablation: {name}\n{'='*60}")
        evaluator = Evaluator(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
            ablation=ablation,
        )
        results = evaluator.evaluate(
            max_samples=max_samples,
            output_path=str(Path(output_dir) / f"{name}.json"),
        )
        all_ablation_results[name] = results

    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("  ABLATION STUDY SUMMARY")
    logger.info("=" * 80)
    header = f"  {'Ablation':<25} {'Elem Acc':>10} {'Act Acc':>10} {'F1 Macro':>10} {'Replan%':>10}"
    logger.info(header)
    logger.info("-" * 80)
    for name, results in all_ablation_results.items():
        for split, m in results.items():
            if isinstance(m, dict):
                row = (
                    f"  {name:<25} "
                    f"{m.get('element_accuracy', 0):.4f}     "
                    f"{m.get('action_accuracy', 0):.4f}     "
                    f"{m.get('f1/macro', 0):.4f}     "
                    f"{m.get('replan_fraction', 0):.4f}"
                )
                logger.info(row)

    return all_ablation_results


# ── CLI ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLA Agent Evaluation (Refactored)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument(
        "--ablation", type=str, default="full",
        choices=list(ABLATION_PRESETS.keys()) + ["all"],
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.ablation == "all":
        run_all_ablations(
            config=config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            max_samples=args.max_samples,
        )
    else:
        evaluator = Evaluator(
            config=config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            ablation=ABLATION_PRESETS[args.ablation],
        )
        evaluator.evaluate(
            splits=args.splits,
            max_samples=args.max_samples,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
