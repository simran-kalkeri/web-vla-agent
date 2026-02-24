"""
Embedding Precomputation — Candidate-Based Pipeline.

Precomputes frozen text encoder outputs for all candidates in the pool:
  1. Task text embedding             [text_dim]
  2. Candidate text embeddings       [N, text_dim]
  3. Structural features (from candidates) [N, struct_dim]
  4. Graph edge_index (sequential chain for candidates)
  5. Target element index            scalar
  6. Operation label                 scalar

Usage:
    python3 -m data.precompute_embeddings --split train
    python3 -m data.precompute_embeddings --split train --max-samples 100
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from data.mind2web_loader import Mind2WebDataset, vla_collate_fn
from models.encoders import TextEncoder
from models.graph_dom_encoder import structural_features, build_edge_index
from utils.config import VLAConfig, load_config
from utils.logging import get_logger


class EmbeddingPrecomputer:
    """Precompute and cache frozen text encoder outputs."""

    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        cache_dir: str = "cached_embeddings",
        device: str = "cpu",
    ):
        self.config = config or load_config()
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.logger = get_logger("precomputer")

        self.text_encoder = TextEncoder(
            model_name=self.config.model.text_encoder,
            output_dim=self.config.model.text_dim,
            device=device,
        )
        self.logger.info(f"Text encoder loaded: {self.config.model.text_encoder}")

    def precompute(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        """Run precomputation for a dataset split."""
        split_dir = self.cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = Mind2WebDataset(
            split=split, config=self.config.data, max_samples=max_samples,
        )
        self.logger.info(f"Precomputing {split}: {len(dataset)} samples → {split_dir}")

        # Single-sample iteration (no batching to avoid collate overhead)
        start_time = time.time()
        skipped = 0
        valid = 0

        for idx in range(len(dataset)):
            cache_path = split_dir / f"sample_{idx}.pt"

            if cache_path.exists():
                if idx % 500 == 0:
                    self.logger.info(f"  [{idx}/{len(dataset)}] already cached, skipping")
                continue

            try:
                sample = dataset[idx]
                cached = self._process_sample(sample, idx)
                if cached is not None:
                    torch.save(cached, cache_path)
                    valid += 1
                else:
                    skipped += 1
            except Exception as e:
                self.logger.warning(f"  [{idx}] Error: {e}")
                skipped += 1

            if idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                target_ok = "✓" if cached and cached["target_idx"].item() >= 0 else "✗"
                self.logger.info(
                    f"  [{idx}/{len(dataset)}] "
                    f"rate={rate:.1f}/s "
                    f"valid={valid} skip={skipped} "
                    f"target={target_ok}"
                )

        elapsed = time.time() - start_time
        self.logger.info(
            f"\nDone: {valid}/{len(dataset)} cached in {elapsed:.0f}s "
            f"({valid / max(elapsed,1):.1f}/s), skipped={skipped}"
        )

        # Save metadata
        meta = {
            "split": split,
            "num_samples": len(dataset),
            "num_cached": valid,
            "num_skipped": skipped,
            "text_dim": self.config.model.text_dim,
            "struct_dim": self.config.graph.structural_dim,
        }
        torch.save(meta, split_dir / "metadata.pt")

    def _process_sample(
        self, sample: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample into cached tensors."""
        sigs = sample["element_signatures"]
        dom_elements = sample["dom_elements"]
        task = sample["task"]
        target_idx = sample["target_element_idx"]
        op_label = sample["operation_label"]

        if not sigs or not dom_elements:
            return None

        with torch.no_grad():
            # Task embedding
            task_emb = self.text_encoder([task]).squeeze(0).cpu()

            # Candidate element embeddings (chunked for efficiency)
            chunk_size = 32
            text_embs = []
            for i in range(0, len(sigs), chunk_size):
                chunk = sigs[i:i + chunk_size]
                emb = self.text_encoder(chunk).cpu()
                text_embs.append(emb)
            text_embs = torch.cat(text_embs, dim=0)

            # Structural features
            struct_feat = structural_features(dom_elements)

            # Graph edges (sequential chain for candidates)
            edge_index, edge_type = build_edge_index(dom_elements)

        return {
            "task_emb": task_emb,
            "text_embs": text_embs,
            "struct_feat": struct_feat,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
            "op_label": torch.tensor(op_label, dtype=torch.long),
            "num_elements": torch.tensor(len(sigs), dtype=torch.long),
        }


# ── Cached Dataset ───────────────────────────────────────────

class CachedEmbeddingDataset(torch.utils.data.Dataset):
    """Load precomputed embeddings from cache. No encoder calls."""

    def __init__(
        self,
        cache_dir: str = "cached_embeddings",
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.split_dir = Path(cache_dir) / split
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Cache not found: {self.split_dir}. "
                f"Run: python3 -m data.precompute_embeddings --split {split}"
            )
        self.paths = sorted(self.split_dir.glob("sample_*.pt"))
        if max_samples is not None:
            self.paths = self.paths[:max_samples]

        meta_path = self.split_dir / "metadata.pt"
        self.metadata = torch.load(meta_path, weights_only=False) if meta_path.exists() else {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return torch.load(self.paths[idx], weights_only=False)


def cached_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate cached embedding samples."""
    return {
        "task_embs": torch.stack([s["task_emb"] for s in batch]),
        "text_embs_list": [s["text_embs"] for s in batch],
        "struct_feat_list": [s["struct_feat"] for s in batch],
        "edge_index_list": [s["edge_index"] for s in batch],
        "target_idx": torch.stack([s["target_idx"] for s in batch]),
        "op_label": torch.stack([s["op_label"] for s in batch]),
        "num_elements": torch.stack([s["num_elements"] for s in batch]),
    }


# ── CLI ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute embeddings")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache-dir", type=str, default="cached_embeddings")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    precomputer = EmbeddingPrecomputer(
        config=config, cache_dir=args.cache_dir, device=args.device,
    )
    precomputer.precompute(split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
