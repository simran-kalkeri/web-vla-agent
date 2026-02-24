"""
Embedding Precomputation — CPU Efficiency Requirement.

Precomputes and caches all frozen encoder outputs so training
never recomputes embeddings:

  1. Task text embeddings       [text_dim]     — from MiniLM
  2. DOM element text embeddings [N, text_dim] — from MiniLM
  3. DOM structural features    [N, struct_dim]
  4. DOM graph edge_index       [2, E]
  5. Target element index       scalar
  6. Operation label            scalar

Stored as individual .pt files in cache_dir.
Training loads from cache → ~100× faster.

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
    """
    Precompute and cache all frozen encoder outputs for the dataset.

    This eliminates the CPU bottleneck of running MiniLM per batch.
    """

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

        # Frozen text encoder
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
        batch_size: int = 1,
    ) -> None:
        """Run precomputation for a dataset split."""
        split_dir = self.cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = Mind2WebDataset(
            split=split, config=self.config.data, max_samples=max_samples,
        )
        self.logger.info(f"Precomputing {split}: {len(dataset)} samples → {split_dir}")

        loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=vla_collate_fn, num_workers=0,
        )

        start_time = time.time()
        skipped = 0

        for idx, batch in enumerate(loader):
            cache_path = split_dir / f"sample_{idx}.pt"

            # Skip if already cached
            if cache_path.exists():
                if idx % 500 == 0:
                    self.logger.info(f"  [{idx}/{len(dataset)}] already cached, skipping")
                continue

            try:
                cached = self._process_sample(batch, idx)
                if cached is not None:
                    torch.save(cached, cache_path)
                else:
                    skipped += 1
            except Exception as e:
                self.logger.warning(f"  [{idx}] Error: {e}")
                skipped += 1

            if idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                self.logger.info(
                    f"  [{idx}/{len(dataset)}] "
                    f"rate={rate:.1f} samples/s, "
                    f"elapsed={elapsed:.0f}s, "
                    f"skipped={skipped}"
                )

        elapsed = time.time() - start_time
        self.logger.info(
            f"\nPrecomputation complete: {len(dataset) - skipped}/{len(dataset)} cached "
            f"in {elapsed:.0f}s ({(len(dataset) - skipped) / elapsed:.1f} samples/s)"
        )

        # Save metadata
        meta = {
            "split": split,
            "num_samples": len(dataset),
            "num_cached": len(dataset) - skipped,
            "text_dim": self.config.model.text_dim,
            "struct_dim": self.config.graph.structural_dim,
        }
        torch.save(meta, split_dir / "metadata.pt")

    def _process_sample(
        self, batch: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single sample and return cached tensors."""
        sigs = batch["element_signatures"][0]
        dom_elements = batch["dom_elements"][0]
        task = batch["tasks"][0]
        target_idx = batch["target_element_idx"][0].item()
        op_label = batch["operation_label"][0].item()

        if not sigs or not dom_elements:
            return None

        with torch.no_grad():
            # Task embedding
            task_emb = self.text_encoder([task]).squeeze(0).cpu()  # [text_dim]

            # DOM element embeddings (batched for efficiency)
            # Process in chunks to avoid memory issues
            chunk_size = 32
            text_embs = []
            for i in range(0, len(sigs), chunk_size):
                chunk = sigs[i:i + chunk_size]
                emb = self.text_encoder(chunk).cpu()
                text_embs.append(emb)
            text_embs = torch.cat(text_embs, dim=0)  # [N, text_dim]

            # Structural features
            struct_feat = structural_features(dom_elements)  # [N, struct_dim]

            # Graph edges
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
            "sample_id": str(batch["sample_ids"][0]),
            "trajectory_length": batch["trajectory_lengths"][0],
        }


# ── Cached Dataset ───────────────────────────────────────────

class CachedEmbeddingDataset(torch.utils.data.Dataset):
    """
    Load precomputed embeddings from cache_dir.
    Each sample is a dict of tensors — no encoder calls needed.
    """

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

        # Find all cached samples
        self.paths = sorted(self.split_dir.glob("sample_*.pt"))
        if max_samples is not None:
            self.paths = self.paths[:max_samples]

        # Load metadata
        meta_path = self.split_dir / "metadata.pt"
        self.metadata = torch.load(meta_path, weights_only=False) if meta_path.exists() else {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return torch.load(self.paths[idx], weights_only=False)


def cached_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate cached embedding samples — no padding, variable-length lists."""
    return {
        "task_embs": torch.stack([s["task_emb"] for s in batch]),         # [B, text_dim]
        "text_embs_list": [s["text_embs"] for s in batch],               # list of [N_i, text_dim]
        "struct_feat_list": [s["struct_feat"] for s in batch],            # list of [N_i, struct_dim]
        "edge_index_list": [s["edge_index"] for s in batch],             # list of [2, E_i]
        "target_idx": torch.stack([s["target_idx"] for s in batch]),      # [B]
        "op_label": torch.stack([s["op_label"] for s in batch]),          # [B]
        "num_elements": torch.stack([s["num_elements"] for s in batch]),  # [B]
    }


# ── CLI ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute embeddings for CPU training")
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
