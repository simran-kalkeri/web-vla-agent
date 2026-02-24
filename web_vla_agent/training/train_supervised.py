"""
Supervised Training Loop — 3-Phase Schedule (CPU-Optimized).

Uses precomputed embeddings from CachedEmbeddingDataset for CPU efficiency.
No encoder inference during training — all frozen outputs are cached as .pt files.

Phase 1: Grounding Stabilization (GCN + cross-attn + element head)
Phase 2: Joint Grounding + Action (focal loss)
Phase 3: Entropy Replanning (auto-calibrate τ)

Metrics logged every epoch:
    element_accuracy, recall@1, recall@5, mean_true_rank,
    CE_elem, InfoNCE, gradient_norm, action_accuracy (Phase 2+),
    per-class F1 (Phase 2+), entropy stats
"""
from __future__ import annotations

import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from data.precompute_embeddings import CachedEmbeddingDataset, cached_collate_fn
from models.graph_dom_encoder import GraphDOMEncoder
from models.grounder import VLAGrounderBatched
from models.uncertainty import EntropyUncertainty
from utils.config import VLAConfig, load_config
from utils.logging import get_logger, log_metrics


# ── Reproducibility ──────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Focal Loss ───────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss: FL(p_t) = -(1-p_t)^γ · log(p_t). γ=0 → standard CE."""
    ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    if gamma > 0:
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = ((1 - p_t) ** gamma) * ce_loss
    else:
        loss = ce_loss
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


# ── Training state ───────────────────────────────────────────

class TrainingState:
    def __init__(self):
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_elem_acc: float = 0.0
        self.best_action_acc: float = 0.0
        self.current_phase: int = 1
        self.entropy_threshold: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        return vars(self).copy()


# ── Trainer ──────────────────────────────────────────────────

class SupervisedTrainer:
    """
    3-Phase supervised trainer using precomputed embeddings.
    No encoder inference during training — all frozen outputs cached.
    """

    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        device: str = "cpu",
        use_graph: bool = True,
        cache_dir: str = "cached_embeddings",
    ):
        self.config = config or load_config()
        self.device = device
        self.use_graph = use_graph
        self.cache_dir = cache_dir
        set_seed(self.config.training.seed)

        self.logger = get_logger(
            "trainer",
            level=self.config.logging.level,
            log_dir=self.config.logging.log_dir,
        )

        # ── Graph encoder (trainable, optional for baseline) ─
        self.graph_encoder = None
        if self.use_graph:
            self.graph_encoder = GraphDOMEncoder(
                text_dim=self.config.model.text_dim,
                struct_dim=self.config.graph.structural_dim,
                hidden_dim=self.config.model.hidden_dim,
                dropout=self.config.graph.dropout,
            ).to(device)

        # ── Grounder (trainable) ─────────────────────────────
        grounder_text_dim = (
            self.config.model.hidden_dim if self.use_graph
            else self.config.model.text_dim
        )
        self.grounder = VLAGrounderBatched(
            d_model=self.config.model.hidden_dim,
            text_dim=grounder_text_dim,
            vision_dim=self.config.model.vision_dim,
            subgoal_dim=self.config.model.text_dim,
            num_heads=self.config.grounder.num_heads,
            temperature=self.config.grounder.contrastive_temperature,
            dropout=self.config.grounder.dropout,
            num_action_types=self.config.model.num_action_types,
        ).to(device)

        # ── Uncertainty (no trainable params) ────────────────
        self.uncertainty = EntropyUncertainty(
            threshold_tau=self.config.uncertainty.entropy_threshold,
            temperature=self.config.uncertainty.temperature,
        )

        self.action_class_weights: Optional[torch.Tensor] = None

        # ── Optimizer ────────────────────────────────────────
        trainable_params = list(self.grounder.parameters())
        if self.graph_encoder:
            trainable_params += list(self.graph_encoder.parameters())

        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        self.state = TrainingState()

        total_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        self.logger.info(f"Trainable parameters: {total_params:,}")

    # ── Phase management ─────────────────────────────────────

    def _get_phase(self, epoch: int) -> int:
        cfg = self.config.training
        if epoch < cfg.phase1_epochs:
            return 1
        elif epoch < cfg.phase2_epochs:
            return 2
        return 3

    def _action_head_enabled(self, phase: int) -> bool:
        return phase >= 2

    # ── Class weights ────────────────────────────────────────

    def _compute_class_weights(self, dataset: CachedEmbeddingDataset) -> torch.Tensor:
        counts = Counter()
        for i in range(min(len(dataset), 1000)):
            sample = dataset[i]
            counts[sample["op_label"].item()] += 1

        num_classes = self.config.model.num_action_types
        total = sum(counts.values())
        weights = torch.ones(num_classes, device=self.device)
        for cls_idx in range(num_classes):
            count = counts.get(cls_idx, 1)
            weights[cls_idx] = total / (num_classes * max(count, 1))

        self.logger.info(f"Action class weights: {weights.tolist()}")
        self.logger.info(f"Action class counts: {dict(counts)}")
        return weights

    # ── Main loop ────────────────────────────────────────────

    def train(
        self,
        max_samples: Optional[int] = None,
        resume_checkpoint: Optional[str] = None,
        max_epochs: Optional[int] = None,
    ) -> None:
        cfg = self.config.training
        num_epochs = max_epochs or cfg.num_epochs
        mode_str = "GCN" if self.use_graph else "FLAT (no_graph baseline)"

        self.logger.info("=" * 70)
        self.logger.info(f"  VLA Agent — Phase 1 Grounding Training [{mode_str}]")
        self.logger.info("=" * 70)
        self.logger.info(f"  Using CACHED embeddings from: {self.cache_dir}")
        self.logger.info(f"  Loss: λ₁={cfg.lambda_element}·CE_elem + "
                         f"λ₂={cfg.lambda_contrastive}·InfoNCE")
        self.logger.info(f"  Phase 1: epochs 0–{cfg.phase1_epochs-1} (grounding only)")
        self.logger.info(f"  Phase 2: epochs {cfg.phase1_epochs}–{cfg.phase2_epochs-1} (joint)")

        # Dataset from cache
        dataset = CachedEmbeddingDataset(
            cache_dir=self.cache_dir, split="train", max_samples=max_samples,
        )
        self.logger.info(f"Cached dataset size: {len(dataset)} samples")

        self.action_class_weights = self._compute_class_weights(dataset)

        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

        # LR scheduler
        steps_per_epoch = max(len(dataset) // cfg.batch_size, 1)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = max(int(total_steps * cfg.warmup_ratio), 1)
        warmup_sched = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1))
        scheduler = SequentialLR(
            self.optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps]
        )

        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=cached_collate_fn, num_workers=0, pin_memory=False,
        )

        for epoch in range(self.state.epoch, num_epochs):
            self.state.epoch = epoch
            phase = self._get_phase(epoch)

            if phase != self.state.current_phase:
                self.state.current_phase = phase
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"  ENTERING PHASE {phase}")
                self.logger.info(f"{'='*50}")

            t0 = time.time()
            epoch_metrics = self._train_epoch(loader, scheduler, phase)
            epoch_time = time.time() - t0

            # Epoch summary
            self.logger.info(
                f"\nEpoch {epoch+1}/{num_epochs} [Phase {phase}] [{mode_str}] "
                f"({epoch_time:.1f}s)\n"
                f"  L_total     = {epoch_metrics.get('L_total', 0):.4f}\n"
                f"  L_CE_elem   = {epoch_metrics.get('L_element', 0):.4f}\n"
                f"  L_InfoNCE   = {epoch_metrics.get('L_infonce', 0):.4f}\n"
                f"  elem_acc    = {epoch_metrics.get('element_accuracy', 0):.4f}\n"
                f"  recall@1    = {epoch_metrics.get('recall_at_1', 0):.4f}\n"
                f"  recall@5    = {epoch_metrics.get('recall_at_5', 0):.4f}\n"
                f"  mean_rank   = {epoch_metrics.get('mean_true_rank', -1):.2f}\n"
                f"  grad_norm   = {epoch_metrics.get('grad_norm', 0):.4f}"
            )

            if self._action_head_enabled(phase):
                self.logger.info(
                    f"  action_acc  = {epoch_metrics.get('action_accuracy', 0):.4f}\n"
                    f"  L_action    = {epoch_metrics.get('L_action', 0):.4f}"
                )
                # Per-class F1
                for k, v in sorted(epoch_metrics.items()):
                    if k.startswith("f1/"):
                        self.logger.info(f"  {k:14s} = {v:.4f}")

            # Track best
            elem_acc = epoch_metrics.get("element_accuracy", 0)
            action_acc = epoch_metrics.get("action_accuracy", 0)
            if elem_acc > self.state.best_elem_acc:
                self.state.best_elem_acc = elem_acc
            if action_acc > self.state.best_action_acc:
                self.state.best_action_acc = action_acc

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        self.logger.info(f"\nTraining complete [{mode_str}]!")
        self.logger.info(f"  Best element_accuracy:  {self.state.best_elem_acc:.4f}")
        self.logger.info(f"  Best action_accuracy:   {self.state.best_action_acc:.4f}")
        self._save_checkpoint(num_epochs - 1)

    # ── Single epoch ─────────────────────────────────────────

    def _train_epoch(
        self, loader: DataLoader, scheduler: Any, phase: int
    ) -> Dict[str, float]:
        if self.graph_encoder:
            self.graph_encoder.train()
        self.grounder.grounder.train()

        total_metrics: Dict[str, float] = {}
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            self.optimizer.zero_grad()
            loss, metrics = self._train_step(batch, phase)

            if loss.item() == 0.0:
                continue

            loss.backward()

            # Gradient norm
            trainable_params = list(self.grounder.parameters())
            if self.graph_encoder:
                trainable_params += list(self.graph_encoder.parameters())
            grad_norm = nn.utils.clip_grad_norm_(
                trainable_params, self.config.training.max_grad_norm,
            ).item()
            metrics["grad_norm"] = grad_norm

            self.optimizer.step()
            scheduler.step()
            self.state.global_step += 1

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1

            if batch_idx % 50 == 0:
                self.logger.info(
                    f"  batch {batch_idx:>4d} | loss={loss.item():.4f} "
                    f"elem={metrics.get('element_accuracy',0):.3f} "
                    f"r@1={metrics.get('recall_at_1',0):.3f} "
                    f"r@5={metrics.get('recall_at_5',0):.3f} "
                    f"rank={metrics.get('mean_true_rank',-1):.1f} "
                    f"grad={grad_norm:.3f}"
                )

        for k in total_metrics:
            total_metrics[k] /= max(num_batches, 1)
        return total_metrics

    # ── Single step (CACHE-BASED) ────────────────────────────

    def _train_step(
        self, batch: Dict[str, Any], phase: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass using precomputed cached embeddings.
        No encoder calls — just GCN + grounder + loss.
        """
        cfg = self.config.training

        task_embs = batch["task_embs"].to(self.device)          # [B, text_dim]
        text_embs_list = batch["text_embs_list"]                # list of [N_i, text_dim]
        struct_feat_list = batch["struct_feat_list"]             # list of [N_i, struct_dim]
        edge_index_list = batch["edge_index_list"]              # list of [2, E_i]
        target_idx = batch["target_idx"].to(self.device)        # [B]
        op_labels = batch["op_label"].to(self.device)           # [B]

        B = task_embs.shape[0]

        # 1. Graph encode (or passthrough for flat baseline)
        dom_embs_list = []
        for i in range(B):
            text_emb_i = text_embs_list[i].to(self.device)
            struct_i = struct_feat_list[i].to(self.device)
            edge_i = edge_index_list[i].to(self.device)

            if self.use_graph and self.graph_encoder:
                dom_emb = self.graph_encoder(text_emb_i, struct_i, edge_i)
            else:
                dom_emb = text_emb_i  # Flat baseline
            dom_embs_list.append(dom_emb)

        # 2. Vision placeholder (random noise — no screenshots in Phase 1)
        vision_embs_list = []
        for dom_emb in dom_embs_list:
            n = dom_emb.shape[0]
            ev = torch.randn(n, self.config.model.vision_dim, device=self.device) * 0.01
            vision_embs_list.append(ev)

        # 3. Forward through grounder
        positive_indices = target_idx.tolist()

        all_elem_logits, action_logits, all_scores, all_reprs = self.grounder(
            task_embs, dom_embs_list, vision_embs_list
        )

        # 4. Loss

        # L_CE_element
        max_n = max((el.shape[0] for el in all_elem_logits), default=1)
        padded_elem_logits = torch.full((B, max_n), float("-inf"), device=self.device)
        elem_targets = torch.full((B,), -1, dtype=torch.long, device=self.device)

        for i in range(B):
            n = all_elem_logits[i].shape[0]
            if n > 0:
                padded_elem_logits[i, :n] = all_elem_logits[i]
            idx = positive_indices[i]
            if 0 <= idx < n:
                elem_targets[i] = idx

        valid_mask = elem_targets >= 0
        if valid_mask.any():
            loss_element = F.cross_entropy(
                padded_elem_logits[valid_mask], elem_targets[valid_mask],
            )
        else:
            loss_element = torch.tensor(0.0, device=self.device, requires_grad=True)

        # L_InfoNCE
        loss_infonce = self.grounder.compute_loss_batched(
            task_embs, dom_embs_list, vision_embs_list, positive_indices
        )

        # L_CE_action (Phase 2+ only)
        if self._action_head_enabled(phase):
            loss_action = focal_loss(
                action_logits, op_labels,
                gamma=cfg.focal_gamma, weight=self.action_class_weights,
            )
        else:
            loss_action = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Total
        total_loss = (
            cfg.lambda_element * loss_element
            + cfg.lambda_contrastive * loss_infonce
            + (cfg.lambda_action * loss_action if self._action_head_enabled(phase) else 0.0)
        )

        # 5. Metrics
        with torch.no_grad():
            elem_acc, recall_1, recall_5, mean_rank = self._grounding_metrics(
                all_elem_logits, positive_indices
            )

            pred_actions = action_logits.argmax(dim=-1)
            action_acc = (pred_actions == op_labels).float().mean().item()

            per_class_f1 = {}
            if self._action_head_enabled(phase):
                action_names = ["CLICK", "TYPE", "SCROLL", "SELECT"]
                for cls_idx, name in enumerate(action_names):
                    tp = ((pred_actions == cls_idx) & (op_labels == cls_idx)).sum().item()
                    fp = (pred_actions == cls_idx).sum().item() - tp
                    fn = (op_labels == cls_idx).sum().item() - tp
                    prec = tp / max(tp + fp, 1)
                    rec = tp / max(tp + fn, 1)
                    per_class_f1[f"f1/{name}"] = 2*prec*rec / max(prec+rec, 1e-8)

            entropy_stats = {}
            if all_scores:
                padded_scores = torch.full((B, max_n), float("-inf"), device=self.device)
                for i in range(B):
                    n = all_scores[i].shape[0]
                    if n > 0:
                        padded_scores[i, :n] = all_scores[i]
                entropy_stats = self.uncertainty.entropy_statistics(padded_scores)

        return total_loss, {
            "L_total": total_loss.item(),
            "L_element": loss_element.item(),
            "L_infonce": loss_infonce.item(),
            "L_action": loss_action.item(),
            "element_accuracy": elem_acc,
            "recall_at_1": recall_1,
            "recall_at_5": recall_5,
            "mean_true_rank": mean_rank,
            "action_accuracy": action_acc,
            "phase": float(phase),
            **per_class_f1,
            **entropy_stats,
        }

    # ── Grounding metrics ────────────────────────────────────

    @staticmethod
    def _grounding_metrics(
        all_elem_logits: List[torch.Tensor],
        positive_indices: List[int],
    ) -> Tuple[float, float, float, float]:
        """Compute element_accuracy, recall@1, recall@5, mean_true_rank."""
        correct, total = 0, 0
        in_top1, in_top5 = 0, 0
        ranks = []

        for i, logits in enumerate(all_elem_logits):
            n = logits.shape[0]
            idx = positive_indices[i]
            if n == 0 or idx < 0 or idx >= n:
                continue
            total += 1

            _, sorted_indices = logits.sort(descending=True)
            rank_pos = (sorted_indices == idx).nonzero(as_tuple=True)[0]
            if rank_pos.numel() > 0:
                rank = rank_pos[0].item()
                ranks.append(rank)
                if rank == 0:
                    correct += 1
                    in_top1 += 1
                if rank < 5:
                    in_top5 += 1

        if total == 0:
            return 0.0, 0.0, 0.0, -1.0

        return (
            correct / total,
            in_top1 / total,
            in_top5 / total,
            sum(ranks) / len(ranks) if ranks else -1.0,
        )

    # ── Checkpointing ────────────────────────────────────────

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt_dir = Path(self.config.training.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"checkpoint_epoch_{epoch+1}.pt"

        state_dict = {
            "grounder": self.grounder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_state": self.state.to_dict(),
            "action_class_weights": self.action_class_weights,
            "use_graph": self.use_graph,
        }
        if self.graph_encoder:
            state_dict["graph_encoder"] = self.graph_encoder.state_dict()

        torch.save(state_dict, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "graph_encoder" in ckpt and self.graph_encoder:
            self.graph_encoder.load_state_dict(ckpt["graph_encoder"])
        self.grounder.load_state_dict(ckpt["grounder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "action_class_weights" in ckpt and ckpt["action_class_weights"] is not None:
            self.action_class_weights = ckpt["action_class_weights"].to(self.device)
        ts = ckpt.get("training_state", {})
        self.state.epoch = ts.get("epoch", 0)
        self.state.global_step = ts.get("global_step", 0)
        self.state.best_elem_acc = ts.get("best_elem_acc", 0.0)
        self.state.best_action_acc = ts.get("best_action_acc", 0.0)
        self.state.current_phase = ts.get("current_phase", 1)
        self.logger.info(f"Resumed from {path} (epoch {self.state.epoch})")


# ── CLI ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLA 3-Phase Supervised Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="cached_embeddings")
    parser.add_argument(
        "--no-graph", action="store_true",
        help="Run flat DOM baseline (no GCN)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = SupervisedTrainer(
        config=config, device=args.device,
        use_graph=not args.no_graph,
        cache_dir=args.cache_dir,
    )
    trainer.train(
        max_samples=args.max_samples,
        resume_checkpoint=args.resume,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
