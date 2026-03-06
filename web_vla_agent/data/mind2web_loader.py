"""
Multimodal-Mind2Web Dataset Loader — Candidate-Based.

Loads the osunlp/Multimodal-Mind2Web dataset and builds training
examples with:
  - Task instruction
  - Candidate elements (from pos_candidates + neg_candidates or full DOM)
  - Screenshot (PIL Image)
  - Ground-truth action JSON targets with candidate indices
  - Action history for multi-step training

Candidate-based format: the model picks a candidate index from a
numbered list of DOM elements. The positive candidate is shuffled
among negatives to prevent the model from learning position bias.
"""
from __future__ import annotations

import json
import logging
import random
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

    # Candidates (new)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    target_candidate_index: int = -1   # which candidate is the target

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

    Builds candidate-based training examples where each sample has a
    list of candidate elements and the ground-truth action references
    a candidate index (not a backend_node_id).

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier.
    max_dom_nodes : int
        Maximum DOM nodes to include in serialization.
    max_text_per_node : int
        Maximum text characters per node.
    max_neg_candidates : int
        Maximum number of negative candidates to include per sample.
    """

    def __init__(
        self,
        dataset_name: str = "osunlp/Multimodal-Mind2Web",
        max_dom_nodes: int = 500,
        max_text_per_node: int = 200,
        max_neg_candidates: int = 30,
        scroll_augmentation: bool = True,
        scroll_aug_ratio: float = 0.2,
    ):
        self.dataset_name = dataset_name
        self.max_dom_nodes = max_dom_nodes
        self.max_text_per_node = max_text_per_node
        self.max_neg_candidates = max_neg_candidates
        self.scroll_augmentation = scroll_augmentation
        self.scroll_aug_ratio = scroll_aug_ratio

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

        Builds a candidate list from pos_candidates and neg_candidates,
        shuffles them, and sets the target action to use the candidate index.
        """
        sample_id = raw_sample.get("annotation_id", "")
        task = raw_sample.get("confirmed_task", "")
        website = raw_sample.get("website", "")
        domain = raw_sample.get("domain", "")
        subdomain = raw_sample.get("subdomain", "")

        # Get HTML
        cleaned_html = raw_sample.get("cleaned_html", "")
        raw_html = raw_sample.get("raw_html", "")

        # Serialize full DOM (for backward compat / fallback)
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

        # Build operation info
        operation = raw_sample.get("operation", {})
        if isinstance(operation, str):
            try:
                operation = json.loads(operation)
            except json.JSONDecodeError:
                operation = {}

        op_type = operation.get("op", "CLICK").upper()
        action_type = _OP_MAP.get(op_type, "CLICK")
        value = operation.get("value", "")

        # ── Extract candidates ───────────────────────────────
        candidates, target_idx = self._build_candidates(raw_sample, cleaned_html)

        if not candidates or target_idx < 0:
            # Fallback: try to build from full DOM
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
                candidates=[],
                target_candidate_index=-1,
                action={},
                action_repr=self._get_action_repr(raw_sample),
                trajectory_id=sample_id,
            )

        # Build action with candidate index
        action: Dict[str, Any] = {"action": action_type, "candidate": target_idx}
        if action_type == "TYPE" and value:
            action["value"] = str(value)
        elif action_type == "SELECT" and value:
            action["value"] = str(value)

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
            candidates=candidates,
            target_candidate_index=target_idx,
            action=action,
            action_repr=action_repr,
            trajectory_id=sample_id,
        )

    def _build_candidates(
        self,
        raw_sample: Dict[str, Any],
        cleaned_html: str,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Build the candidate list from pos_candidates and neg_candidates.

        Returns
        -------
        candidates : list of candidate dicts
            Each has: candidate_index, tag, text, attributes, backend_node_id
        target_idx : int
            The candidate_index of the positive candidate (target element)
        """
        pos_candidates = raw_sample.get("pos_candidates", [])
        neg_candidates = raw_sample.get("neg_candidates", [])

        # Parse if string
        if isinstance(pos_candidates, str):
            try:
                pos_candidates = json.loads(pos_candidates)
            except json.JSONDecodeError:
                pos_candidates = []
        if isinstance(neg_candidates, str):
            try:
                neg_candidates = json.loads(neg_candidates)
            except json.JSONDecodeError:
                neg_candidates = []

        if not pos_candidates:
            return [], -1

        # Extract the first positive candidate
        if isinstance(pos_candidates, list) and len(pos_candidates) > 0:
            pos_cand = pos_candidates[0]
        else:
            pos_cand = pos_candidates

        pos_element = self._candidate_to_element(pos_cand, is_positive=True)
        if pos_element is None:
            return [], -1

        # Extract negative candidates (limit count)
        neg_elements = []
        if isinstance(neg_candidates, list):
            for nc in neg_candidates[:self.max_neg_candidates]:
                el = self._candidate_to_element(nc, is_positive=False)
                if el is not None:
                    neg_elements.append(el)

        # If we have no neg candidates from the dataset, try to extract from DOM
        if not neg_elements and cleaned_html:
            neg_elements = self._extract_dom_negatives(
                cleaned_html, pos_element, max_count=self.max_neg_candidates
            )

        # Combine: positive + negatives, then shuffle
        all_elements = [pos_element] + neg_elements
        random.shuffle(all_elements)

        # Find the target index after shuffling
        target_idx = -1
        candidates = []
        for i, el in enumerate(all_elements):
            el["candidate_index"] = i
            candidates.append(el)
            if el.get("_is_positive"):
                target_idx = i

        # Remove internal marker
        for c in candidates:
            c.pop("_is_positive", None)

        return candidates, target_idx

    def _candidate_to_element(
        self,
        candidate: Any,
        is_positive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a Mind2Web candidate (pos or neg) to a candidate element dict.

        Mind2Web candidates can be:
          - a dict with keys like backend_node_id, tag, text, attributes
          - a list [backend_node_id, ...]
          - an int (just backend_node_id)
        """
        if candidate is None:
            return None

        if isinstance(candidate, dict):
            tag = candidate.get("tag", "")
            if not tag:
                tag = candidate.get("tagName", "div")
            tag = tag.lower()

            text = candidate.get("text", "")
            if not text:
                # Try to get from attributes
                text = candidate.get("value", "")

            attributes = candidate.get("attributes", {})
            if isinstance(attributes, str):
                try:
                    attributes = json.loads(attributes)
                except json.JSONDecodeError:
                    # Try parsing as HTML-style attributes
                    attributes = self._parse_attr_string(attributes)

            backend_node_id = candidate.get("backend_node_id", -1)

            return {
                "tag": tag,
                "text": str(text)[:200],
                "attributes": attributes if isinstance(attributes, dict) else {},
                "backend_node_id": backend_node_id,
                "_is_positive": is_positive,
            }

        elif isinstance(candidate, (int, float)):
            return {
                "tag": "div",
                "text": "",
                "attributes": {},
                "backend_node_id": int(candidate),
                "_is_positive": is_positive,
            }

        elif isinstance(candidate, list) and len(candidate) > 0:
            # Try first element as backend_node_id
            return {
                "tag": "div",
                "text": "",
                "attributes": {},
                "backend_node_id": candidate[0] if isinstance(candidate[0], int) else -1,
                "_is_positive": is_positive,
            }

        return None

    def _parse_attr_string(self, attr_str: str) -> Dict[str, str]:
        """Parse HTML-style attribute string into dict."""
        import re
        attrs = {}
        for match in re.finditer(r'(\w[\w-]*)=["\']([^"\']*)["\']', attr_str):
            attrs[match.group(1)] = match.group(2)
        return attrs

    def _extract_dom_negatives(
        self,
        cleaned_html: str,
        pos_element: Dict[str, Any],
        max_count: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Extract negative candidates from the full DOM when dataset
        doesn't provide neg_candidates.
        """
        try:
            _, nodes = self.serializer.serialize_from_html(cleaned_html)
        except Exception:
            return []

        # Filter to interactable nodes that aren't the positive candidate
        pos_text = pos_element.get("text", "").strip().lower()
        pos_tag = pos_element.get("tag", "").lower()

        negatives = []
        for node in nodes:
            if not node.is_interactable:
                continue
            # Skip if it looks like the positive candidate
            if (node.tag == pos_tag and
                    node.text.strip().lower() == pos_text and pos_text):
                continue

            negatives.append({
                "tag": node.tag,
                "text": node.text[:200],
                "attributes": dict(node.attributes),
                "backend_node_id": node.node_id,
                "_is_positive": False,
            })

        random.shuffle(negatives)
        return negatives[:max_count]

    def build_training_examples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        include_screenshot: bool = True,
    ) -> List[Mind2WebSample]:
        """
        Build all training examples from the dataset.

        Includes synthetic scroll augmentation: for a fraction of samples,
        generates a SCROLL action that teaches the model to scroll when
        the target element would be below the viewport.
        """
        ds = self.load_dataset(split=split, max_samples=max_samples)
        samples = []
        scroll_count = 0

        for i, raw in enumerate(ds):
            try:
                sample = self.process_sample(raw, include_screenshot=include_screenshot)
                # Must have candidates and a valid target
                if sample.candidates and sample.target_candidate_index >= 0 and sample.action:
                    samples.append(sample)

                    # Synthetic scroll augmentation
                    if self.scroll_augmentation:
                        scroll_sample = self._generate_scroll_augmentation(sample)
                        if scroll_sample is not None:
                            samples.append(scroll_sample)
                            scroll_count += 1
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue

            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} samples, {len(samples)} valid ({scroll_count} scroll aug)")

        logger.info(f"Built {len(samples)} training examples from {split} ({scroll_count} scroll augmented)")
        return samples

    def _generate_scroll_augmentation(
        self,
        sample: Mind2WebSample,
    ) -> Optional[Mind2WebSample]:
        """
        Generate a synthetic SCROLL training sample from an existing sample.

        Heuristic: if the positive candidate is in the bottom half of the
        candidate list (i.e., likely below the viewport), create a SCROLL
        sample with probability scroll_aug_ratio.

        The scroll sample has the same task/candidates but the target action
        is {"action": "SCROLL", "direction": "down"}. This teaches the model
        to scroll when the target element isn't in the visible candidates.
        """
        # Only augment a fraction of samples
        if random.random() > self.scroll_aug_ratio:
            return None

        # Only create scroll if the positive candidate is in the bottom
        # half of the candidate list (simulating element below viewport)
        target_idx = sample.target_candidate_index
        if target_idx < len(sample.candidates) // 2:
            return None  # Target is near top, no need to scroll

        # Create scroll sample: same task and screenshot,
        # but action is SCROLL and we exclude the positive candidate
        # from the visible candidates (simulating it being off-screen)
        scroll_candidates = [
            c for c in sample.candidates
            if c["candidate_index"] != target_idx
        ]

        # Re-index the candidates
        for i, c in enumerate(scroll_candidates):
            c = dict(c)  # copy
            c["candidate_index"] = i
            scroll_candidates[i] = c

        if not scroll_candidates:
            return None

        return Mind2WebSample(
            sample_id=f"{sample.sample_id}_scroll",
            task=sample.task,
            website=sample.website,
            domain=sample.domain,
            subdomain=sample.subdomain,
            raw_html=sample.raw_html,
            cleaned_html=sample.cleaned_html,
            serialized_dom=sample.serialized_dom,
            candidates=scroll_candidates,
            target_candidate_index=-1,  # no target element (it's off-screen)
            screenshot=sample.screenshot,
            action={"action": "SCROLL", "direction": "down"},
            action_repr="SCROLL down to find target element",
            action_history=sample.action_history,
            step_index=sample.step_index,
            trajectory_id=sample.trajectory_id,
        )

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
                    "candidate": step.action.get("candidate", -1),
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
        """
        Extract screenshot from the dataset sample.

        Always returns an RGB PIL Image. If the sample has no screenshot,
        returns a white placeholder (224x224) so every training sample
        is multimodal and the vision pipeline stays active.
        """
        screenshot = raw_sample.get("screenshot")
        img = None

        # If it's already a PIL Image
        if isinstance(screenshot, Image.Image):
            img = screenshot

        # If it's bytes
        elif isinstance(screenshot, bytes):
            from io import BytesIO
            try:
                img = Image.open(BytesIO(screenshot))
            except Exception:
                img = None

        # If it's a path string
        elif isinstance(screenshot, str):
            try:
                img = Image.open(screenshot)
            except Exception:
                img = None

        # Fallback: white placeholder so every sample is multimodal
        if img is None:
            img = Image.new("RGB", (224, 224), (255, 255, 255))

        # Ensure RGB (Qwen2-VL image processor requires 3-channel input)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img
