"""
Multimodal-Mind2Web Dataset Loader — Full DOM (No Candidate Pools).

Loads the osunlp/Multimodal-Mind2Web dataset and builds training
examples with:
  - Task instruction
  - Full serialized DOM (structured, not candidate-filtered)
  - Screenshot (PIL Image)
  - Ground-truth action JSON targets
  - Action history for multi-step training

NO candidate pool restriction.
NO pos/neg filtered lists.
Full DOM tree serialization.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Mind2Web operation type mapping
_OP_MAP = {
    "CLICK": "CLICK",
    "TYPE": "TYPE",
    "SELECT": "SELECT",
    "HOVER": "CLICK",     # map hover to click
    "ENTER": "CLICK",     # map enter to click
}


@dataclass
class Mind2WebSample:
    """A single training sample from Multimodal-Mind2Web."""
    sample_id: str = ""
    task: str = ""
    website: str = ""
    domain: str = ""
    subdomain: str = ""

    # DOM
    raw_html: str = ""
    cleaned_html: str = ""
    serialized_dom: str = ""

    # Screenshot
    screenshot: Optional[Image.Image] = None

    # Ground-truth action
    action: Dict[str, Any] = field(default_factory=dict)
    action_repr: str = ""

    # Trajectory context (for multi-step training)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    step_index: int = 0
    trajectory_id: str = ""


@dataclass
class Mind2WebTrajectory:
    """A full trajectory (multi-step) from Mind2Web."""
    trajectory_id: str = ""
    task: str = ""
    website: str = ""
    domain: str = ""
    steps: List[Mind2WebSample] = field(default_factory=list)


class Mind2WebLoader:
    """
    Load Multimodal-Mind2Web dataset for VLA training.

    Loads the full dataset (not sample-limited) and builds
    training examples with full DOM serialization.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier.
    max_dom_nodes : int
        Maximum DOM nodes to include in serialization.
    max_text_per_node : int
        Maximum text characters per node.
    """

    def __init__(
        self,
        dataset_name: str = "osunlp/Multimodal-Mind2Web",
        max_dom_nodes: int = 500,
        max_text_per_node: int = 200,
    ):
        self.dataset_name = dataset_name
        self.max_dom_nodes = max_dom_nodes
        self.max_text_per_node = max_text_per_node

        # Lazy import
        self._serializer = None

    @property
    def serializer(self):
        if self._serializer is None:
            from environment.dom_serializer import DOMSerializer
            self._serializer = DOMSerializer(
                max_nodes=self.max_dom_nodes,
                max_text_len=self.max_text_per_node,
            )
        return self._serializer

    def load_dataset(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        streaming: bool = False,
    ):
        """
        Load the raw HuggingFace dataset.

        Parameters
        ----------
        split : str
            Dataset split: 'train', 'test_task', 'test_website', 'test_domain'
        max_samples : int, optional
            Limit number of samples (None = full dataset).
        streaming : bool
            Use streaming mode for large datasets.
        """
        from datasets import load_dataset

        logger.info(f"Loading {self.dataset_name} split={split} streaming={streaming}")

        if streaming:
            ds = load_dataset(self.dataset_name, split=split, streaming=True)
        else:
            ds = load_dataset(self.dataset_name, split=split)

        if max_samples and not streaming:
            ds = ds.select(range(min(max_samples, len(ds))))

        return ds

    def process_sample(
        self,
        raw_sample: Dict[str, Any],
        include_screenshot: bool = True,
    ) -> Mind2WebSample:
        """
        Process a single raw dataset sample into a Mind2WebSample.

        Extracts:
          - Task instruction from confirmed_task
          - Serialized DOM from cleaned_html (full, not candidate-filtered)
          - Screenshot from the screenshot field
          - Ground-truth action from operation + pos_candidates
        """
        sample_id = raw_sample.get("annotation_id", "")
        task = raw_sample.get("confirmed_task", "")
        website = raw_sample.get("website", "")
        domain = raw_sample.get("domain", "")
        subdomain = raw_sample.get("subdomain", "")

        # Get HTML — use cleaned_html for DOM serialization
        cleaned_html = raw_sample.get("cleaned_html", "")
        raw_html = raw_sample.get("raw_html", "")

        # Serialize full DOM (NOT candidate pool)
        serialized_dom = ""
        if cleaned_html:
            try:
                serialized_dom, _ = self.serializer.serialize_from_html(cleaned_html)
            except Exception as e:
                logger.warning(f"DOM serialization failed for {sample_id}: {e}")
                serialized_dom = ""

        # Extract screenshot
        screenshot = None
        if include_screenshot:
            screenshot = self._extract_screenshot(raw_sample)

        # Build ground-truth action
        action = self._build_action(raw_sample)
        action_repr = self._get_action_repr(raw_sample)

        return Mind2WebSample(
            sample_id=sample_id,
            task=task,
            website=website,
            domain=domain,
            subdomain=subdomain,
            raw_html=raw_html,
            cleaned_html=cleaned_html,
            serialized_dom=serialized_dom,
            screenshot=screenshot,
            action=action,
            action_repr=action_repr,
            trajectory_id=sample_id,
        )

    def build_training_examples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        include_screenshot: bool = True,
    ) -> List[Mind2WebSample]:
        """
        Build all training examples from the dataset.

        This loads the FULL dataset (no sample limit by default).
        """
        ds = self.load_dataset(split=split, max_samples=max_samples)
        samples = []

        for i, raw in enumerate(ds):
            try:
                sample = self.process_sample(raw, include_screenshot=include_screenshot)
                if sample.serialized_dom and sample.action:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue

            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} samples, {len(samples)} valid")

        logger.info(f"Built {len(samples)} training examples from {split}")
        return samples

    def build_trajectories(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        include_screenshot: bool = True,
    ) -> List[Mind2WebTrajectory]:
        """
        Group samples into trajectories for multi-step training.

        Samples with the same annotation_id prefix are grouped together.
        Each step gets the action history from previous steps.
        """
        samples = self.build_training_examples(
            split=split,
            max_samples=max_samples,
            include_screenshot=include_screenshot,
        )

        # Group by trajectory_id (annotation_id)
        traj_map: Dict[str, List[Mind2WebSample]] = {}
        for s in samples:
            tid = s.trajectory_id
            if tid not in traj_map:
                traj_map[tid] = []
            traj_map[tid].append(s)

        trajectories = []
        for tid, steps in traj_map.items():
            # Sort by step index
            steps.sort(key=lambda s: s.step_index)

            # Build action history for each step
            history: List[Dict[str, Any]] = []
            for i, step in enumerate(steps):
                step.step_index = i
                step.action_history = list(history)
                history.append({
                    "step": i,
                    "action": step.action.get("action", ""),
                    "element_id": step.action.get("element_id", -1),
                    "value": step.action.get("value", ""),
                    "description": step.action_repr,
                })

            trajectories.append(Mind2WebTrajectory(
                trajectory_id=tid,
                task=steps[0].task if steps else "",
                website=steps[0].website if steps else "",
                domain=steps[0].domain if steps else "",
                steps=steps,
            ))

        logger.info(f"Built {len(trajectories)} trajectories with {sum(len(t.steps) for t in trajectories)} total steps")
        return trajectories

    def _build_action(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ground-truth action JSON from Mind2Web sample.

        Uses operation field + pos_candidates to determine:
          - action type (CLICK/TYPE/SELECT)
          - target element (mapped to DOM node ID)
          - value (for TYPE/SELECT)
        """
        operation = raw_sample.get("operation", {})
        if isinstance(operation, str):
            try:
                operation = json.loads(operation)
            except json.JSONDecodeError:
                return {}

        op_type = operation.get("op", "CLICK").upper()
        action_type = _OP_MAP.get(op_type, "CLICK")
        value = operation.get("value", "")

        # Get target element from pos_candidates
        pos_candidates = raw_sample.get("pos_candidates", [])
        if isinstance(pos_candidates, str):
            try:
                pos_candidates = json.loads(pos_candidates)
            except json.JSONDecodeError:
                pos_candidates = []

        # Build the target element info
        target_element_id = -1
        if pos_candidates:
            # Use the first positive candidate
            first_pos = pos_candidates[0] if isinstance(pos_candidates, list) else pos_candidates
            if isinstance(first_pos, dict):
                target_element_id = first_pos.get("backend_node_id", -1)
            elif isinstance(first_pos, (int, float)):
                target_element_id = int(first_pos)

        action: Dict[str, Any] = {"action": action_type}

        if action_type in ("CLICK", "TYPE", "SELECT"):
            action["element_id"] = target_element_id

        if action_type == "TYPE" and value:
            action["value"] = str(value)
        elif action_type == "SELECT" and value:
            action["value"] = str(value)

        return action

    def _get_action_repr(self, raw_sample: Dict[str, Any]) -> str:
        """Get human-readable action representation."""
        action_reprs = raw_sample.get("action_reprs", [])
        target_idx = raw_sample.get("target_action_index", 0)
        target_reprs = raw_sample.get("target_action_reprs", "")

        if target_reprs:
            return str(target_reprs)
        if action_reprs:
            if isinstance(action_reprs, list) and len(action_reprs) > 0:
                idx = min(target_idx, len(action_reprs) - 1)
                return str(action_reprs[idx])
            return str(action_reprs)
        return ""

    def _extract_screenshot(self, raw_sample: Dict[str, Any]) -> Optional[Image.Image]:
        """Extract screenshot from the dataset sample."""
        screenshot = raw_sample.get("screenshot")
        if screenshot is None:
            return None

        # If it's already a PIL Image
        if isinstance(screenshot, Image.Image):
            return screenshot

        # If it's bytes
        if isinstance(screenshot, bytes):
            from io import BytesIO
            try:
                return Image.open(BytesIO(screenshot))
            except Exception:
                return None

        # If it's a path string
        if isinstance(screenshot, str):
            try:
                return Image.open(screenshot)
            except Exception:
                return None

        return None


class Mind2WebCollator:
    """
    Collate Mind2WebSamples into batches for training.

    Handles tokenization, padding, and label construction
    for the VLA model's causal LM training.
    """

    def __init__(
        self,
        processor,
        prompt_builder,
        max_seq_length: int = 4096,
    ):
        self.processor = processor
        self.prompt_builder = prompt_builder
        self.max_seq_length = max_seq_length

    def __call__(
        self,
        samples: List[Mind2WebSample],
    ) -> Dict[str, Any]:
        """
        Collate samples into a training batch.

        Returns dict with input_ids, attention_mask, labels, and
        optionally pixel_values for screenshots.
        """
        import torch

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_images = []

        for sample in samples:
            # Build training prompt
            training = self.prompt_builder.build_training_prompt(
                task=sample.task,
                serialized_dom=sample.serialized_dom,
                target_action=sample.action,
                action_history=sample.action_history,
                extra_context=f"URL: {sample.website}" if sample.website else "",
            )

            prompt_text = training["prompt"]
            target_text = training["target"]
            full_text = training["full_text"]

            # Tokenize
            prompt_tokens = self.processor.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length - 100,  # leave room for target
                add_special_tokens=False,
            )
            target_tokens = self.processor.tokenizer(
                "\n" + target_text,
                return_tensors="pt",
                add_special_tokens=False,
            )

            # Build input_ids and labels
            input_ids = torch.cat([
                prompt_tokens["input_ids"][0],
                target_tokens["input_ids"][0],
            ])

            # Labels: -100 for prompt tokens, actual tokens for target
            labels = torch.cat([
                torch.full_like(prompt_tokens["input_ids"][0], -100),
                target_tokens["input_ids"][0],
            ])

            # Truncate to max length
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(torch.ones_like(input_ids))

            if sample.screenshot is not None:
                batch_images.append(sample.screenshot)

        # Pad to same length
        max_len = max(ids.shape[0] for ids in batch_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lbls, mask in zip(batch_input_ids, batch_labels, batch_attention_mask):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([lbls, torch.full((pad_len,), -100, dtype=lbls.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        result = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

        return result
