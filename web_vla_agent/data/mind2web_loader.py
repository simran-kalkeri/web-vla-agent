"""
Multimodal-Mind2Web dataset loader.

Loads from the HuggingFace cache (``osunlp/Multimodal-Mind2Web``).
Wraps the HF dataset rows into a PyTorch Dataset with structured
preprocessing, candidate extraction, and a custom collate function.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data.preprocessing import (
    DOMElement,
    DOMProcessor,
    ScreenshotProcessor,
    crop_element_from_screenshot,
    BoundingBox,
)
from utils.config import DataConfig, VLAConfig, load_config


# ── Structured sample ───────────────────────────────────────

@dataclass
class Mind2WebSample:
    """A single processed Mind2Web training / evaluation sample."""
    # Identifiers
    sample_id: str = ""
    annotation_id: str = ""

    # Task
    task: str = ""
    website: str = ""
    domain: str = ""

    # Observation at this step
    screenshot: Optional[Image.Image] = None
    cleaned_html: str = ""
    url: str = ""

    # Ground-truth action
    action_uid: str = ""  # target element backend_node_id
    operation: str = ""   # CLICK / TYPE / SELECT
    value: str = ""       # typed text or selected option

    # History
    previous_actions: List[str] = field(default_factory=list)

    # Candidate elements
    pos_candidates: List[Dict[str, Any]] = field(default_factory=list)
    neg_candidates: List[Dict[str, Any]] = field(default_factory=list)

    # Preprocessed DOM elements (filled by dataset)
    dom_elements: List[DOMElement] = field(default_factory=list)

    # Number of steps in the full trajectory (for curriculum)
    trajectory_length: int = 1


# ── PyTorch dataset ──────────────────────────────────────────

OPERATION_MAP = {"CLICK": 0, "TYPE": 1, "SELECT": 2, "SCROLL": 3}


class Mind2WebDataset(Dataset):
    """
    PyTorch wrapper around Multimodal-Mind2Web.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test_task"``, ``"test_website"``,
        ``"test_domain"``.
    config : DataConfig | None
        Data-specific config.  Falls back to defaults when *None*.
    max_samples : int | None
        Cap the number of samples (handy for debugging).
    """

    def __init__(
        self,
        split: str = "train",
        config: Optional[DataConfig] = None,
        max_samples: Optional[int] = None,
    ):
        self.config = config or DataConfig()
        self.split = split
        self.dom_processor = DOMProcessor(max_elements=self.config.max_dom_elements)
        self.screenshot_processor = ScreenshotProcessor(
            size=tuple(self.config.screenshot_size)  # type: ignore[arg-type]
        )

        # Lazy import so the top-level module doesn't need `datasets`
        from datasets import load_dataset

        ds = load_dataset(self.config.dataset_name, split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self._dataset = ds

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._dataset[idx]
        sample = self._parse_row(row)
        return self._to_tensors(sample)

    # ── Row parsing ──────────────────────────────────────────
    def _parse_row(self, row: Dict[str, Any]) -> Mind2WebSample:
        """Turn a raw HF row into a :class:`Mind2WebSample`."""
        # Screenshot
        screenshot = None
        if "screenshot" in row and row["screenshot"] is not None:
            img_data = row["screenshot"]
            if isinstance(img_data, Image.Image):
                screenshot = img_data
            elif isinstance(img_data, bytes):
                screenshot = Image.open(BytesIO(img_data))
            elif isinstance(img_data, str):
                # base64 or path — skipped for now
                screenshot = None

        # Parse operation from JSON field
        op_raw = row.get("operation", "") or ""
        operation, value = self._parse_operation(op_raw)

        # Candidates
        pos_candidates = self._safe_json(row.get("pos_candidates", "[]"))
        neg_candidates = self._safe_json(row.get("neg_candidates", "[]"))

        # Extract target element's backend_node_id from pos_candidates
        target_backend_node_id = self._extract_target_node_id(pos_candidates)

        # Previous actions
        prev = row.get("previous_actions", []) or []
        if isinstance(prev, str):
            prev = self._safe_json(prev)
        if not isinstance(prev, list):
            prev = []

        # DOM
        cleaned_html = row.get("cleaned_html", "") or ""
        dom_elements = self.dom_processor.parse(cleaned_html)

        return Mind2WebSample(
            sample_id=str(row.get("action_uid", "")),
            annotation_id=str(row.get("annotation_id", "")),
            task=row.get("confirmed_task", "") or row.get("task", "") or "",
            website=row.get("website", "") or "",
            domain=row.get("domain", "") or "",
            screenshot=screenshot,
            cleaned_html=cleaned_html,
            url=row.get("url", "") or "",
            action_uid=target_backend_node_id,  # Use backend_node_id, NOT action_uid UUID
            operation=operation,
            value=value,
            previous_actions=prev[: self.config.max_action_history],
            pos_candidates=pos_candidates,
            neg_candidates=neg_candidates,
            dom_elements=dom_elements,
            trajectory_length=int(row.get("num_steps", 1) or 1),
        )

    # ── Tensor conversion ────────────────────────────────────
    def _to_tensors(self, sample: Mind2WebSample) -> Dict[str, Any]:
        """Convert a sample into a dict suitable for DataLoader."""
        # Screenshot tensor [C, H, W]
        if sample.screenshot is not None:
            screenshot_np = self.screenshot_processor.to_numpy(sample.screenshot)
            screenshot_tensor = torch.from_numpy(screenshot_np)
        else:
            h, w = self.config.screenshot_size
            screenshot_tensor = torch.zeros(3, h, w, dtype=torch.float32)

        # Element signatures (for text encoder)
        element_sigs = [el.signature for el in sample.dom_elements]
        element_ids = [el.element_id for el in sample.dom_elements]

        # Ground-truth label: index of target element in element list
        target_idx = -1
        for i, eid in enumerate(element_ids):
            if eid == sample.action_uid:
                target_idx = i
                break

        # Operation label
        op_label = OPERATION_MAP.get(sample.operation.upper(), 0)

        return {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "screenshot": screenshot_tensor,
            "element_signatures": element_sigs,
            "element_ids": element_ids,
            "num_elements": len(element_sigs),
            "target_element_idx": target_idx,
            "operation_label": op_label,
            "value": sample.value,
            "previous_actions": sample.previous_actions,
            "cleaned_html": sample.cleaned_html,
            "pos_candidates": sample.pos_candidates,
            "neg_candidates": sample.neg_candidates,
            "trajectory_length": sample.trajectory_length,
            "dom_elements": sample.dom_elements,
        }

    # ── Helpers ──────────────────────────────────────────────
    @staticmethod
    def _parse_operation(op_raw: str) -> Tuple[str, str]:
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

    @staticmethod
    def _parse_action(action_repr: str) -> Tuple[str, str]:
        """Best-effort parse of Mind2Web action strings (fallback)."""
        action_repr = action_repr.strip()
        if not action_repr:
            return ("CLICK", "")
        upper = action_repr.upper()
        if upper.startswith("SELECT"):
            return ("SELECT", action_repr.split(maxsplit=1)[-1] if " " in action_repr else "")
        elif upper.startswith("TYPE"):
            return ("TYPE", action_repr.split(maxsplit=1)[-1] if " " in action_repr else "")
        else:
            return ("CLICK", "")

    @staticmethod
    def _extract_target_node_id(pos_candidates: list) -> str:
        """Extract backend_node_id from the first positive candidate."""
        if not pos_candidates:
            return ""
        cand = pos_candidates[0]
        if isinstance(cand, str):
            try:
                cand = json.loads(cand)
            except (json.JSONDecodeError, TypeError):
                return ""
        # Direct backend_node_id field
        if isinstance(cand, dict):
            bnid = cand.get("backend_node_id", "")
            if bnid:
                return str(bnid)
            # Try inside nested attributes JSON
            attrs = cand.get("attributes", "")
            if isinstance(attrs, str):
                try:
                    attrs_d = json.loads(attrs)
                    return str(attrs_d.get("backend_node_id", ""))
                except (json.JSONDecodeError, TypeError):
                    pass
        return ""

    @staticmethod
    def _safe_json(val: Any) -> list:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        return []


# ── Collate ──────────────────────────────────────────────────

def vla_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate that handles variable-length element lists.

    Returns
    -------
    dict with:
      screenshots : Tensor [B, C, H, W]
      target_element_idx : Tensor [B]
      operation_label : Tensor [B]
      tasks : list[str]
      element_signatures : list[list[str]]   (variable per sample)
      ... and other fields as lists
    """
    screenshots = torch.stack([s["screenshot"] for s in batch])
    target_idx = torch.tensor(
        [s["target_element_idx"] for s in batch], dtype=torch.long
    )
    op_labels = torch.tensor(
        [s["operation_label"] for s in batch], dtype=torch.long
    )

    return {
        "screenshots": screenshots,
        "target_element_idx": target_idx,
        "operation_label": op_labels,
        "tasks": [s["task"] for s in batch],
        "values": [s["value"] for s in batch],
        "element_signatures": [s["element_signatures"] for s in batch],
        "element_ids": [s["element_ids"] for s in batch],
        "num_elements": [s["num_elements"] for s in batch],
        "previous_actions": [s["previous_actions"] for s in batch],
        "sample_ids": [s["sample_id"] for s in batch],
        "trajectory_lengths": [s["trajectory_length"] for s in batch],
        "dom_elements": [s["dom_elements"] for s in batch],
        "pos_candidates": [s["pos_candidates"] for s in batch],
        "neg_candidates": [s["neg_candidates"] for s in batch],
    }


def build_dataloader(
    split: str = "train",
    config: Optional[DataConfig] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """Convenience builder for a DataLoader."""
    cfg = config or DataConfig()
    ds = Mind2WebDataset(split=split, config=cfg, max_samples=max_samples)
    return DataLoader(
        ds,
        batch_size=8,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=vla_collate_fn,
        pin_memory=True,
    )
