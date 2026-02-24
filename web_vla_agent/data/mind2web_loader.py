"""
Multimodal-Mind2Web dataset loader — Candidate-Based.

Mind2Web uses a CANDIDATE-BASED approach:
  - Each sample provides pos_candidates (1+) and neg_candidates (~50–600)
  - The element pool = pos + neg (shuffled)
  - Target = index of the positive candidate in the pool
  - This guarantees 100% target match rate

Each candidate has: tag, attributes (JSON), backend_node_id, text content.
We build element signatures from these for text encoding.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocessing import DOMElement, BoundingBox
from utils.config import DataConfig


# ── Structured sample ───────────────────────────────────────

@dataclass
class Mind2WebSample:
    """A single processed Mind2Web training / evaluation sample."""
    sample_id: str = ""
    annotation_id: str = ""
    task: str = ""
    website: str = ""
    domain: str = ""
    operation: str = ""
    value: str = ""
    previous_actions: List[str] = field(default_factory=list)
    trajectory_length: int = 1

    # Candidate-based element pool
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    target_idx: int = -1  # Index of positive candidate in pool


# ── Operation / candidate parsing ────────────────────────────

OPERATION_MAP = {"CLICK": 0, "TYPE": 1, "SELECT": 2, "SCROLL": 3}


def parse_operation(op_raw: str) -> Tuple[str, str]:
    """Parse the JSON operation field from Mind2Web."""
    if not op_raw:
        return ("CLICK", "")
    try:
        op_dict = json.loads(op_raw) if isinstance(op_raw, str) else op_raw
        operation = op_dict.get("op", op_dict.get("original_op", "CLICK"))
        value = op_dict.get("value", "")
        return (operation.upper(), value)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ("CLICK", "")


def parse_candidate(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a single candidate JSON into a structured dict."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None

    tag = raw.get("tag", "div")
    backend_node_id = str(raw.get("backend_node_id", ""))

    # Parse nested attributes JSON
    attrs_raw = raw.get("attributes", "{}")
    if isinstance(attrs_raw, str):
        try:
            attrs = json.loads(attrs_raw)
        except (json.JSONDecodeError, TypeError):
            attrs = {}
    else:
        attrs = attrs_raw if isinstance(attrs_raw, dict) else {}

    # If backend_node_id not at top level, try from attributes
    if not backend_node_id:
        backend_node_id = str(attrs.get("backend_node_id", ""))

    # Extract useful text for signature
    text = ""
    for key in ["aria_label", "aria-label", "placeholder", "title", "alt", "value", "name"]:
        v = attrs.get(key, "")
        if v:
            text = str(v)[:120]
            break

    # Inner text from the candidate
    inner_text = str(raw.get("text", ""))[:120]
    if inner_text and not text:
        text = inner_text

    # Build signature: "tag | text | key-attrs"
    attr_parts = []
    for k in ["id", "class", "role", "type", "href"]:
        v = attrs.get(k, "")
        if v:
            attr_parts.append(f'{k}="{str(v)[:60]}"')
    attr_str = " ".join(attr_parts)

    text_short = (text[:80] + "…") if len(text) > 80 else text
    signature = f"{tag} | {text_short} | {attr_str}".strip(" |")

    return {
        "tag": tag,
        "backend_node_id": backend_node_id,
        "signature": signature,
        "text": text,
        "attributes": attrs,
        "is_positive": raw.get("_is_positive", False),
    }


def build_candidate_pool(
    pos_candidates: list,
    neg_candidates: list,
    max_candidates: int = 128,
    shuffle: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Build element pool from pos + neg candidates.
    Returns (pool, target_idx).
    """
    # Parse positive
    positives = []
    for c in pos_candidates:
        parsed = parse_candidate(c)
        if parsed:
            parsed["_is_positive"] = True
            positives.append(parsed)

    # Parse negatives (cap to leave room for positives)
    max_neg = max_candidates - len(positives)
    negatives = []
    for c in neg_candidates[:max_neg * 2]:  # parse extra in case some fail
        if len(negatives) >= max_neg:
            break
        parsed = parse_candidate(c)
        if parsed:
            parsed["_is_positive"] = False
            negatives.append(parsed)

    if not positives:
        return negatives[:max_candidates], -1

    # Combine
    pool = positives + negatives
    pool = pool[:max_candidates]

    if shuffle:
        random.shuffle(pool)

    # Find target index
    target_idx = -1
    for i, c in enumerate(pool):
        if c.get("_is_positive", False):
            target_idx = i
            break

    return pool, target_idx


# ── Candidate → DOMElement conversion (for GCN) ─────────────

def candidates_to_dom_elements(candidates: List[Dict[str, Any]]) -> List[DOMElement]:
    """Convert candidate dicts to DOMElement objects for structural features."""
    elements = []
    for i, c in enumerate(candidates):
        attrs = c.get("attributes", {})
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except:
                attrs = {}

        # Parse bounding box if available
        bbox = None
        bbox_str = attrs.get("bounding_box_rect", "")
        if bbox_str:
            try:
                parts = [float(x) for x in str(bbox_str).split(",")[:4]]
                if len(parts) == 4:
                    bbox = BoundingBox(x=parts[0], y=parts[1], width=parts[2], height=parts[3])
            except (ValueError, TypeError):
                pass

        tag = c.get("tag", "div")
        is_clickable = tag in {"a", "button", "input", "select", "textarea", "option", "label"}

        elem = DOMElement(
            element_id=c.get("backend_node_id", f"cand_{i}"),
            tag=tag,
            text=c.get("text", "")[:256],
            attributes={k: str(v)[:100] for k, v in attrs.items()
                       if k in {"id", "class", "role", "type", "href", "name", "aria_label"}},
            bounding_box=bbox,
            is_clickable=is_clickable,
            is_visible=True,
            parent_id=None,
            depth=0,
        )
        elements.append(elem)
    return elements


# ── PyTorch Dataset ──────────────────────────────────────────

class Mind2WebDataset(Dataset):
    """
    Candidate-based Mind2Web dataset.
    Element pool = pos_candidates + neg_candidates (shuffled).
    Target = index of positive candidate in pool.
    100% target match rate guaranteed.
    """

    def __init__(
        self,
        split: str = "train",
        config: Optional[DataConfig] = None,
        max_samples: Optional[int] = None,
        max_candidates: int = 128,
    ):
        self.config = config or DataConfig()
        self.split = split
        self.max_candidates = max_candidates

        from datasets import load_dataset
        ds = load_dataset(self.config.dataset_name, split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self._dataset = ds

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._dataset[idx]
        return self._process_row(row)

    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Operation
        op_raw = row.get("operation", "") or ""
        operation, value = parse_operation(op_raw)
        op_label = OPERATION_MAP.get(operation, 0)

        # Task
        task = row.get("confirmed_task", "") or row.get("task", "") or ""

        # Build candidate pool
        pos_raw = row.get("pos_candidates", []) or []
        neg_raw = row.get("neg_candidates", []) or []
        if isinstance(pos_raw, str):
            pos_raw = json.loads(pos_raw) if pos_raw else []
        if isinstance(neg_raw, str):
            neg_raw = json.loads(neg_raw) if neg_raw else []

        pool, target_idx = build_candidate_pool(
            pos_raw, neg_raw,
            max_candidates=self.max_candidates,
            shuffle=(self.split == "train"),
        )

        # Element signatures
        element_sigs = [c.get("signature", "") for c in pool]

        # DOM elements for structural features
        dom_elements = candidates_to_dom_elements(pool)

        return {
            "sample_id": str(row.get("action_uid", "")),
            "task": task,
            "element_signatures": element_sigs,
            "num_elements": len(pool),
            "target_element_idx": target_idx,
            "operation_label": op_label,
            "value": value,
            "domain": row.get("domain", "") or "",
            "website": row.get("website", "") or "",
            "dom_elements": dom_elements,
            "trajectory_length": int(row.get("num_steps", 1) or 1),
        }


# ── Collate ──────────────────────────────────────────────────

def vla_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate for variable-length candidate pools."""
    target_idx = torch.tensor(
        [s["target_element_idx"] for s in batch], dtype=torch.long,
    )
    op_labels = torch.tensor(
        [s["operation_label"] for s in batch], dtype=torch.long,
    )
    return {
        "target_element_idx": target_idx,
        "operation_label": op_labels,
        "tasks": [s["task"] for s in batch],
        "element_signatures": [s["element_signatures"] for s in batch],
        "num_elements": [s["num_elements"] for s in batch],
        "sample_ids": [s["sample_id"] for s in batch],
        "dom_elements": [s["dom_elements"] for s in batch],
        "domains": [s.get("domain", "") for s in batch],
        "trajectory_lengths": [s["trajectory_length"] for s in batch],
    }


def build_dataloader(
    split: str = "train",
    config: Optional[DataConfig] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    batch_size: int = 8,
) -> DataLoader:
    """Convenience builder for a DataLoader."""
    cfg = config or DataConfig()
    ds = Mind2WebDataset(split=split, config=cfg, max_samples=max_samples)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=vla_collate_fn, pin_memory=False,
    )
