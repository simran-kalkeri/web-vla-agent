"""
Task Planner — Hierarchical Instruction Decomposer (H1).

Converts a natural-language web task instruction into an ordered list of
structured subgoals, each annotated with intent type, expected action
category, target element type, and optional constraints.

Includes a **SubgoalValidator** that prevents hallucinated subgoals by
checking grounding feasibility against the current DOM graph embeddings.

**Research gap addressed**: Greedy next-action prediction without structured
planning (H1), and hallucinated subgoal decomposition.

**Mathematical formulation**:
    Subgoal validation:
        feasibility(g) = max_i cos(embed(g), h_i^{DOM})
        valid(g) = feasibility(g) > τ_val
    If not valid → trigger replanning or skip subgoal.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import PlannerConfig


# ── Data structures ──────────────────────────────────────────

class IntentType(str, Enum):
    NAVIGATE = "navigate"
    SEARCH = "search"
    FILL_FORM = "fill_form"
    SELECT_OPTION = "select_option"
    CLICK_ELEMENT = "click_element"
    SCROLL = "scroll"
    VERIFY = "verify"
    SUBMIT = "submit"
    EXTRACT = "extract"
    WAIT = "wait"


class SubgoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subgoal:
    """A single structured subgoal in a task decomposition."""
    subgoal_id: str = ""
    description: str = ""
    intent_type: str = IntentType.CLICK_ELEMENT.value
    action_category: str = "click"       # click / type / scroll / select
    expected_element_type: str = "button"  # button / input / link / select / div …
    constraints: Dict[str, Any] = field(default_factory=dict)
    status: str = SubgoalStatus.PENDING.value
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Subgoal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Prompt template ──────────────────────────────────────────

DECOMPOSITION_PROMPT = """\
You are an expert web navigation planner.  Given a user's web task instruction,
decompose it into an ordered list of atomic subgoals.

Each subgoal MUST be a JSON object with these exact keys:
  - "subgoal_id":  sequential id like "sg1", "sg2", …
  - "description": brief natural-language description of the subgoal
  - "intent_type": one of [navigate, search, fill_form, select_option, click_element, scroll, verify, submit, extract, wait]
  - "action_category": one of [click, type, scroll, select]
  - "expected_element_type": one of [button, input, link, select, div, textarea, checkbox, radio, dropdown, form, other]
  - "constraints": optional dict with extra context (e.g. {{"value": "New York"}})

Return ONLY a JSON array of subgoal objects.  No markdown, no explanation.

### Task instruction:
{instruction}

### Subgoals (JSON array):
"""

REPLAN_PROMPT = """\
You are an expert web navigation planner.  The previous plan failed at the
subgoal shown below.  Given the original instruction, the failed subgoal, and
the current page state, generate a REVISED list of remaining subgoals to
complete the task.

### Original instruction:
{instruction}

### Failed subgoal:
{failed_subgoal}

### Current page URL:
{url}

### Current page title:
{page_title}

Return ONLY a JSON array of subgoal objects (same schema as before).

### Revised subgoals (JSON array):
"""


# ── Planner ──────────────────────────────────────────────────

class TaskPlanner:
    """
    LLM-backed hierarchical task decomposer.

    When no *llm_fn* is provided, the planner uses a lightweight rule-based
    fallback so that the rest of the pipeline can be exercised without a GPU.
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        llm_fn: Optional[Any] = None,
    ):
        self.config = config or PlannerConfig()
        self.llm_fn = llm_fn  # callable(prompt: str) → str

    # ── Public API ───────────────────────────────────────────

    def decompose(self, instruction: str) -> List[Subgoal]:
        """Decompose *instruction* into an ordered subgoal sequence."""
        if self.llm_fn is not None:
            return self._llm_decompose(instruction)
        return self._rule_decompose(instruction)

    def replan(
        self,
        instruction: str,
        failed_subgoal: Subgoal,
        url: str = "",
        page_title: str = "",
    ) -> List[Subgoal]:
        """Re-plan remaining subgoals after a failure."""
        if self.llm_fn is not None:
            return self._llm_replan(instruction, failed_subgoal, url, page_title)
        # Fallback: simply retry the same subgoal
        failed_subgoal.status = SubgoalStatus.PENDING.value
        failed_subgoal.retry_count += 1
        return [failed_subgoal]

    # ── LLM path ─────────────────────────────────────────────

    def _llm_decompose(self, instruction: str) -> List[Subgoal]:
        prompt = DECOMPOSITION_PROMPT.format(instruction=instruction)
        for attempt in range(self.config.max_retries):
            raw = self.llm_fn(prompt)  # type: ignore[misc]
            subgoals = self._parse_json_subgoals(raw)
            if subgoals:
                return subgoals
        # All retries failed — fall back
        return self._rule_decompose(instruction)

    def _llm_replan(
        self,
        instruction: str,
        failed_subgoal: Subgoal,
        url: str,
        page_title: str,
    ) -> List[Subgoal]:
        prompt = REPLAN_PROMPT.format(
            instruction=instruction,
            failed_subgoal=json.dumps(failed_subgoal.to_dict()),
            url=url,
            page_title=page_title,
        )
        for attempt in range(self.config.max_retries):
            raw = self.llm_fn(prompt)  # type: ignore[misc]
            subgoals = self._parse_json_subgoals(raw)
            if subgoals:
                return subgoals
        failed_subgoal.status = SubgoalStatus.PENDING.value
        failed_subgoal.retry_count += 1
        return [failed_subgoal]

    # ── JSON parsing ─────────────────────────────────────────

    @staticmethod
    def _parse_json_subgoals(raw: str) -> List[Subgoal]:
        """Robust extraction of a JSON array of subgoals from LLM output."""
        # Try to find JSON array in the output
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned).strip()

        # Find the outermost [ … ]
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []

        try:
            arr = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        if not isinstance(arr, list):
            return []

        subgoals: List[Subgoal] = []
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                continue
            item.setdefault("subgoal_id", f"sg{i + 1}")
            item.setdefault("status", SubgoalStatus.PENDING.value)
            subgoals.append(Subgoal.from_dict(item))

        return subgoals[: 10]  # cap at max_subgoals

    # ── Rule-based fallback ──────────────────────────────────

    @staticmethod
    def _rule_decompose(instruction: str) -> List[Subgoal]:
        """
        Heuristic decomposition for when no LLM is available.

        Splits on sentence boundaries and assigns reasonable defaults.
        """
        # Normalise
        instruction = instruction.strip()
        # Split on periods, semicolons, "then", "and then", numbered steps
        parts = re.split(r"[.;]\s*|\bthen\b|\band then\b|\d+\)\s*", instruction)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            parts = [instruction]

        subgoals: List[Subgoal] = []
        for i, part in enumerate(parts):
            lower = part.lower()
            # Heuristic intent / action guessing
            if any(w in lower for w in ("go to", "navigate", "open", "visit")):
                intent = IntentType.NAVIGATE.value
                action = "click"
                elem = "link"
            elif any(w in lower for w in ("search", "find", "look for")):
                intent = IntentType.SEARCH.value
                action = "type"
                elem = "input"
            elif any(w in lower for w in ("type", "enter", "fill", "write", "input")):
                intent = IntentType.FILL_FORM.value
                action = "type"
                elem = "input"
            elif any(w in lower for w in ("select", "choose", "pick")):
                intent = IntentType.SELECT_OPTION.value
                action = "select"
                elem = "select"
            elif any(w in lower for w in ("click", "press", "tap", "hit")):
                intent = IntentType.CLICK_ELEMENT.value
                action = "click"
                elem = "button"
            elif any(w in lower for w in ("scroll",)):
                intent = IntentType.SCROLL.value
                action = "scroll"
                elem = "div"
            elif any(w in lower for w in ("submit", "confirm", "send")):
                intent = IntentType.SUBMIT.value
                action = "click"
                elem = "button"
            else:
                intent = IntentType.CLICK_ELEMENT.value
                action = "click"
                elem = "other"

            subgoals.append(Subgoal(
                subgoal_id=f"sg{i + 1}",
                description=part,
                intent_type=intent,
                action_category=action,
                expected_element_type=elem,
            ))

        return subgoals


# ── Subgoal Validation Layer ─────────────────────────────────

@dataclass
class ValidationResult:
    """Result of subgoal feasibility check."""
    subgoal_id: str = ""
    is_valid: bool = True
    max_similarity: float = 0.0
    best_matching_element_idx: int = -1
    reason: str = ""


class SubgoalValidator(nn.Module):
    """
    Validates subgoal feasibility against the current DOM state.

    Computes cosine similarity between the subgoal embedding and all
    graph-encoded DOM node embeddings.  If the maximum similarity is
    below a threshold, the subgoal is flagged as ungrounded
    (hallucinated) and should be replanned.

    **Prevents**: hallucinated subgoal steps that reference elements
    not present in the current page.

    Parameters
    ----------
    threshold : float
        Minimum cosine similarity for a subgoal to be considered grounded.
    subgoal_dim : int
        Dimension of subgoal embeddings.
    dom_dim : int
        Dimension of graph-encoded DOM node embeddings.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        subgoal_dim: int = 384,
        dom_dim: int = 256,
    ):
        super().__init__()
        self.threshold = threshold
        # Project subgoal and DOM to shared space for fair comparison
        shared_dim = min(subgoal_dim, dom_dim)
        self.subgoal_proj = nn.Linear(subgoal_dim, shared_dim)
        self.dom_proj = nn.Linear(dom_dim, shared_dim)

    def validate(
        self,
        subgoal_emb: torch.Tensor,   # [1, subgoal_dim] or [subgoal_dim]
        dom_node_embs: torch.Tensor,  # [N, dom_dim]
    ) -> ValidationResult:
        """
        Check if a subgoal embedding has a matching DOM element.

        Returns
        -------
        ValidationResult with is_valid, max_similarity, best_matching_element_idx
        """
        if subgoal_emb.dim() == 1:
            subgoal_emb = subgoal_emb.unsqueeze(0)

        if dom_node_embs.shape[0] == 0:
            return ValidationResult(
                is_valid=False,
                max_similarity=0.0,
                best_matching_element_idx=-1,
                reason="No DOM elements available",
            )

        with torch.no_grad():
            sg = F.normalize(self.subgoal_proj(subgoal_emb), dim=-1)  # [1, D]
            dom = F.normalize(self.dom_proj(dom_node_embs), dim=-1)    # [N, D]
            similarities = torch.matmul(sg, dom.t()).squeeze(0)       # [N]

            max_sim, best_idx = similarities.max(dim=0)

        is_valid = max_sim.item() >= self.threshold
        return ValidationResult(
            is_valid=is_valid,
            max_similarity=max_sim.item(),
            best_matching_element_idx=best_idx.item(),
            reason="" if is_valid else f"Max similarity {max_sim.item():.3f} < threshold {self.threshold}",
        )

    def validate_batch(
        self,
        subgoal_embs: torch.Tensor,   # [K, subgoal_dim]
        dom_node_embs: torch.Tensor,  # [N, dom_dim]
    ) -> List[ValidationResult]:
        """Validate multiple subgoals against the same DOM."""
        results = []
        for i in range(subgoal_embs.shape[0]):
            results.append(self.validate(subgoal_embs[i], dom_node_embs))
        return results

