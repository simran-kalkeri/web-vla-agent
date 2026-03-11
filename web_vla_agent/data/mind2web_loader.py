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

import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
# Prevent DecompressionBombWarning on large Mind2Web screenshots
# (some are >100MP). PIL warns but still opens them; however a stricter
# env or future PIL version could refuse. Allow up to 200MP safely.
Image.MAX_IMAGE_PIXELS = 200_000_000

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

    # C4 FIX: Ground-truth element identity for evaluation
    gt_backend_node_id: int = -1


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
        max_neg_candidates: int = 63,       # C3/I3: 63 neg + 1 pos = 64
        scroll_augmentation: bool = True,
        scroll_aug_ratio: float = 0.05,
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

        # C2: Capture step_index from raw data
        step_index = raw_sample.get("target_action_index", 0)

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

        # Extract screenshot (S4 FIX: returns None if no screenshot)
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
        candidates, target_idx, gt_backend_node_id = self._build_candidates(raw_sample, cleaned_html)

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
                step_index=step_index,
                gt_backend_node_id=-1,
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
            step_index=step_index,
            gt_backend_node_id=gt_backend_node_id,
        )

    _LOGGED_FIRST_CANDIDATE = False

    def _build_candidates(
        self,
        raw_sample: Dict[str, Any],
        cleaned_html: str,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Build the candidate list from pos_candidates and neg_candidates.

        Handles all observed Mind2Web data formats:
          - list of JSON strings (most common)
          - list of [backend_node_id, dict] pairs
          - list of dicts
          - a single JSON string wrapping a list

        Returns
        -------
        candidates : list of candidate dicts
            Each has: candidate_index, tag, text, attributes, backend_node_id, bbox
        target_idx : int
            The candidate_index of the positive candidate (target element)
        gt_backend_node_id : int
            The backend_node_id of the positive (ground-truth) element (C4 FIX)
        """
        pos_candidates = raw_sample.get("pos_candidates", [])
        neg_candidates = raw_sample.get("neg_candidates", [])

        # Parse top-level if it's a single JSON string wrapping a list
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

        # Ensure list
        if not isinstance(pos_candidates, list):
            pos_candidates = [pos_candidates] if pos_candidates else []
        if not isinstance(neg_candidates, list):
            neg_candidates = [neg_candidates] if neg_candidates else []

        # Diagnostic logging on first sample to surface actual data format
        if not Mind2WebLoader._LOGGED_FIRST_CANDIDATE and pos_candidates:
            Mind2WebLoader._LOGGED_FIRST_CANDIDATE = True
            first = pos_candidates[0]
            print("\n═══ CANDIDATE FORMAT DIAGNOSTICS ═══")
            print(f"  pos_candidates length: {len(pos_candidates)}")
            print(f"  neg_candidates length: {len(neg_candidates)}")
            print(f"  First pos_candidate type: {type(first).__name__}")
            print(f"  First pos_candidate preview: {str(first)[:300]}")
            # Dump raw structure for debugging
            print("\n=== RAW CANDIDATE STRUCTURE ===")
            raw_first = first
            if isinstance(raw_first, str):
                try:
                    raw_first = json.loads(raw_first)
                except json.JSONDecodeError:
                    pass
            if isinstance(raw_first, list) and len(raw_first) == 2:
                raw_first = raw_first[1]
                if isinstance(raw_first, str):
                    try:
                        raw_first = json.loads(raw_first)
                    except json.JSONDecodeError:
                        pass
            print(json.dumps(raw_first, indent=2, default=str)[:1000])
            print("================================")
            print("═════════════════════════════════════\n")

        if not pos_candidates:
            return [], -1, -1

        # Extract the first positive candidate
        pos_cand = pos_candidates[0]
        pos_element = self._candidate_to_element(pos_cand, is_positive=True)
        if pos_element is None:
            logger.debug(
                f"Failed to parse pos_candidate: type={type(pos_cand).__name__}, "
                f"preview={str(pos_cand)[:200]}"
            )
            return [], -1, -1

        # C4 FIX: capture ground-truth backend_node_id
        gt_backend_node_id = pos_element.get("backend_node_id", -1)

        # Extract negative candidates (limit count)
        neg_elements = []
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

        return candidates, target_idx, gt_backend_node_id

    def _candidate_to_element(
        self,
        candidate: Any,
        is_positive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a Mind2Web candidate (pos or neg) to a candidate element dict.

        Mind2Web candidates appear in several formats:
          1. JSON string: '{"tag":"button","attributes":"...","text":"Search"}'
          2. Dict: {"tag":"button", "attributes":"...", "text":"Search"}
          3. Two-element list: [backend_node_id, {"tag":"...", ...}]
          4. Int: just a backend_node_id
        """
        if candidate is None:
            return None

        # ── 1. Handle JSON strings ───────────────────────────────
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                return None

        # ── 2. Handle two-element list: [backend_node_id, dict] ──
        if isinstance(candidate, list):
            if len(candidate) == 2 and isinstance(candidate[1], dict):
                # [backend_node_id, {tag, text, attributes, ...}]
                inner = dict(candidate[1])
                try:
                    inner.setdefault("backend_node_id", int(candidate[0]))
                except (ValueError, TypeError):
                    inner.setdefault("backend_node_id", -1)
                candidate = inner
            elif len(candidate) == 2 and isinstance(candidate[1], str):
                # [backend_node_id, JSON-string]
                try:
                    inner = json.loads(candidate[1])
                    if isinstance(inner, dict):
                        try:
                            inner.setdefault("backend_node_id", int(candidate[0]))
                        except (ValueError, TypeError):
                            inner.setdefault("backend_node_id", -1)
                        candidate = inner
                    else:
                        return self._make_minimal_element(candidate[0], is_positive)
                except json.JSONDecodeError:
                    return self._make_minimal_element(candidate[0], is_positive)
            elif len(candidate) > 0:
                # Generic list — try first element as backend_node_id
                return self._make_minimal_element(candidate[0], is_positive)
            else:
                return None

        # ── 3. Handle dict (standard case) ───────────────────────
        if isinstance(candidate, dict):
            tag = candidate.get("tag", "")
            if not tag:
                tag = candidate.get("tagName", "div")
            tag = tag.lower()

            # Parse attributes FIRST — Mind2Web stores them as a nested JSON string
            # We need these to extract text, so parse early.
            attributes = candidate.get("attributes", {})
            if isinstance(attributes, str):
                try:
                    attributes = json.loads(attributes)
                except json.JSONDecodeError:
                    attributes = self._parse_attr_string(attributes)
            if not isinstance(attributes, dict):
                attributes = {}

            # ── AGGRESSIVE TEXT EXTRACTION ──────────────────────
            # Mind2Web stores text in various places depending on version.
            # Check ALL possible locations in priority order.
            text = (
                # Top-level keys
                candidate.get("text", "")
                or candidate.get("value", "")
                or candidate.get("innerText", "")
                or candidate.get("textContent", "")
                or candidate.get("__text", "")
            )

            # If still empty, check inside parsed attributes dict
            if not text and attributes:
                text = (
                    attributes.get("innerText", "")
                    or attributes.get("textContent", "")
                    or attributes.get("aria-label", "")
                    or attributes.get("placeholder", "")
                    or attributes.get("title", "")
                    or attributes.get("alt", "")
                    or attributes.get("value", "")
                    or attributes.get("name", "")
                    or attributes.get("label", "")
                    or attributes.get("data-label", "")
                    or attributes.get("data-text", "")
                )

            # Last resort: concatenate all non-empty short attribute values
            # to build some textual representation
            if not text and attributes:
                attr_texts = []
                for k, v in attributes.items():
                    if k in ("class", "id", "style", "href", "src",
                             "bounding_box_rect", "backend_node_id",
                             "data-backend-node-id"):
                        continue  # Skip non-semantic attributes
                    sv = str(v).strip()
                    if sv and len(sv) < 100:
                        attr_texts.append(sv)
                if attr_texts:
                    text = " | ".join(attr_texts[:3])  # Take first 3

            # Extract backend_node_id (may be in attributes dict or top-level)
            backend_node_id = candidate.get("backend_node_id", -1)
            if backend_node_id == -1:
                try:
                    backend_node_id = int(attributes.get("backend_node_id", -1))
                except (ValueError, TypeError):
                    backend_node_id = -1

            # I4: Extract bounding_box_rect for spatial grounding
            bbox = None
            bbox_str = attributes.get("bounding_box_rect", "")
            if bbox_str:
                try:
                    parts = [float(x) for x in str(bbox_str).split(",")[:4]]
                    if len(parts) == 4:
                        bbox = parts  # [x, y, w, h]
                except (ValueError, TypeError):
                    pass

            return {
                "tag": tag,
                "text": str(text)[:200],
                "attributes": attributes,
                "backend_node_id": backend_node_id,
                "bbox": bbox,
                "_is_positive": is_positive,
            }

        # ── 4. Handle int/float (just backend_node_id) ───────────
        elif isinstance(candidate, (int, float)):
            return self._make_minimal_element(candidate, is_positive)

        return None

    def _make_minimal_element(
        self,
        backend_node_id: Any,
        is_positive: bool,
    ) -> Optional[Dict[str, Any]]:
        """Create a minimal candidate element from just a backend_node_id."""
        try:
            nid = int(backend_node_id)
        except (ValueError, TypeError):
            return None
        return {
            "tag": "div",
            "text": "",
            "attributes": {},
            "backend_node_id": nid,
            "bbox": None,
            "_is_positive": is_positive,
        }

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
        max_count: int = 63,
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
                "bbox": None,
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
            gt_backend_node_id=-1,
        )

    def build_trajectories(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        include_screenshot: bool = True,
    ) -> List[Mind2WebTrajectory]:
        """
        Group samples into trajectories for multi-step training.

        C2 FIX: The correct grouping key is a hash of
        (confirmed_task + website + action_reprs). The field action_reprs
        is a list of ALL action strings for the whole task.
        Samples with the same (task, website, action_reprs) belong to
        the same trajectory.

        Pass 1: Group raw samples by trajectory key.
        Pass 2: Process each group, sort by target_action_index,
                 inject cumulative action_history.
        """
        ds = self.load_dataset(split=split, max_samples=max_samples)

        # ── Pass 1: Group raw samples by trajectory key ──────────
        traj_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for i, raw in enumerate(ds):
            try:
                confirmed_task = raw.get("confirmed_task", "")
                website = raw.get("website", "")
                action_reprs = raw.get("action_reprs", [])
                if isinstance(action_reprs, str):
                    try:
                        action_reprs = json.loads(action_reprs)
                    except json.JSONDecodeError:
                        action_reprs = [action_reprs]
                if not isinstance(action_reprs, list):
                    action_reprs = [str(action_reprs)]

                # Build trajectory key from task + website + all action representations
                repr_str = "||".join(str(r) for r in action_reprs)
                traj_key_raw = f"{confirmed_task}|||{website}|||{repr_str}"
                traj_key = hashlib.md5(traj_key_raw.encode()).hexdigest()

                traj_groups[traj_key].append(raw)
            except Exception as e:
                logger.warning(f"Failed to group sample {i}: {e}")
                continue

        logger.info(f"Pass 1: {len(traj_groups)} trajectory groups from {split}")

        # ── Pass 2: Process each group ───────────────────────────
        trajectories = []
        total_steps = 0
        skipped_single_step = 0

        for traj_key, raw_steps in traj_groups.items():
            # Sort by target_action_index
            raw_steps.sort(key=lambda r: r.get("target_action_index", 0))

            # Process each step
            processed_steps: List[Mind2WebSample] = []
            for raw in raw_steps:
                try:
                    sample = self.process_sample(raw, include_screenshot=include_screenshot)
                    if sample.candidates and sample.target_candidate_index >= 0 and sample.action:
                        sample.trajectory_id = traj_key
                        sample.step_index = raw.get("target_action_index", 0)
                        processed_steps.append(sample)
                except Exception as e:
                    logger.debug(f"Failed to process step in trajectory {traj_key}: {e}")
                    continue

            # Skip trajectories with fewer than 2 valid steps
            if len(processed_steps) < 2:
                skipped_single_step += 1
                continue

            # Inject cumulative action_history
            history: List[Dict[str, Any]] = []
            for step in processed_steps:
                step.action_history = list(history)  # copy of ALL prior steps' actions
                history.append({
                    "step": step.step_index,
                    "action": step.action.get("action", ""),
                    "candidate": step.action.get("candidate", -1),
                    "value": step.action.get("value", ""),
                    "description": step.action_repr,
                })

            trajectories.append(Mind2WebTrajectory(
                trajectory_id=traj_key,
                task=processed_steps[0].task if processed_steps else "",
                website=processed_steps[0].website if processed_steps else "",
                domain=processed_steps[0].domain if processed_steps else "",
                steps=processed_steps,
            ))
            total_steps += len(processed_steps)

        avg_steps = total_steps / max(len(trajectories), 1)
        logger.info(
            f"Built {len(trajectories)} trajectories with {total_steps} total steps "
            f"(avg {avg_steps:.1f} steps/traj, skipped {skipped_single_step} single-step)"
        )

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

        S4 FIX: Returns None if no screenshot is available, instead of a
        white placeholder image. A 224×224 white image creates spurious
        visual tokens that the model learns to associate with valid states.
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

        # S4 FIX: Return None instead of white placeholder
        if img is None:
            return None

        # Ensure RGB (Qwen2-VL image processor requires 3-channel input)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img
