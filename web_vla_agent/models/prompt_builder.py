"""
Prompt Builder for VLA Web Agent — Candidate-Based.

Constructs multimodal prompts for the Qwen2-VL backbone:
  [SYSTEM] → agent role + action space + grounding rules
  [USER]   → task + candidate list + action history + screenshot

The model generates JSON action as output:
  {"action": "CLICK", "candidate": 3}
  {"action": "TYPE", "candidate": 0, "value": "Brooklyn"}
  {"action": "SCROLL"}
  {"action": "SELECT", "candidate": 8, "value": "Economy"}
  {"action": "STOP"}

Candidate-based approach:
  - Each candidate maps to a real DOM element
  - candidate_index is always a small sequential integer (0, 1, 2, ...)
  - The model picks a candidate index, then the system maps it to the DOM node
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


# FIX S5: Removed "Think about the best element to interact with."
# which contradicted "Return ONLY JSON." — training targets are pure JSON
# with no chain-of-thought.
SYSTEM_PROMPT = """\
You are a web automation agent.

You receive a user task and a list of candidate elements from a webpage.

Each candidate has an index.

Choose the correct element and output ONE JSON action.

Allowed actions:

CLICK
TYPE
SELECT
SCROLL
STOP

Rules:

TYPE must include a "value".
CLICK must include a "candidate".

Examples:

{"action":"CLICK","candidate":2}

{"action":"TYPE","candidate":0,"value":"John"}

{"action":"SCROLL"}

{"action":"STOP"}

Return ONLY JSON.\
"""


class PromptBuilder:
    """
    Build multimodal prompts for the VLA model.

    Uses candidate-based format: the model picks a candidate index
    from a numbered list of DOM elements.
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        max_candidates: int = 64,       # aligned with config (I3)
        max_history_entries: int = 10,
    ):
        self.system_prompt = system_prompt
        self.max_candidates = max_candidates
        self.max_history_entries = max_history_entries

    # Tag display names for cleaner prompt format
    _TAG_DISPLAY = {
        "a": "LINK",
        "button": "BUTTON",
        "input": "INPUT",
        "textarea": "TEXTAREA",
        "select": "SELECT",
        "option": "OPTION",
        "label": "LABEL",
        "summary": "SUMMARY",
        "img": "IMAGE",
        "div": "DIV",
        "span": "SPAN",
    }

    @staticmethod
    def format_candidate(
        index: int,
        tag: str,
        text: str = "",
        attributes: Optional[Dict[str, str]] = None,
        bbox: Optional[List[float]] = None,
    ) -> str:
        """Format a single candidate for prompt display.

        I4: When bbox is available, appends normalized @(x,y) coordinates
        so the model can cross-reference candidate with screenshot.
        """
        display_tag = tag.upper()

        label = text

        if not label:
            attrs = attributes or {}
            label = (
                attrs.get("placeholder")
                or attrs.get("aria-label")
                or attrs.get("name")
                or ""
            )

        # I4: Include normalized bounding box when available
        if bbox and all(v is not None for v in bbox):
            x, y, w, h = bbox
            # Normalize to [0,1] relative to 1280×720 viewport
            nx = round(x / 1280, 2)
            ny = round(y / 720, 2)
            if label:
                return f'[{index}] {display_tag} "{label}" @({nx},{ny})'
            else:
                return f"[{index}] {display_tag} @({nx},{ny})"
        else:
            if label:
                return f'[{index}] {display_tag} "{label}"'
            else:
                return f"[{index}] {display_tag}"

    def format_candidate_list(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Format a list of candidates for the prompt.

        Parameters
        ----------
        candidates : list of dicts
            Each dict must contain:
              - "candidate_index" (int)
              - "tag" (str)
              - "text" (str, optional)
              - "attributes" (dict, optional)
              - "bbox" (list of floats, optional)

        Returns
        -------
        str
            Formatted list of candidates.
        """
        if not candidates:
            return "No candidate elements available."

        lines = []
        for cand in candidates:
            lines.append(
                self.format_candidate(
                    index=cand["candidate_index"],
                    tag=cand["tag"],
                    text=cand.get("text", ""),
                    attributes=cand.get("attributes"),
                    bbox=cand.get("bbox"),
                )
            )

        return "\n".join(lines)

    def build_text_prompt(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
        dom_context: str = "",
    ) -> str:
        """
        Build the text portion of the multimodal prompt.

        Parameters
        ----------
        task : str
            User's task instruction.
        candidates : list of dicts
            Candidate elements with candidate_index, tag, text, attributes.
        action_history : list of dicts, optional
            Previous actions taken.
        extra_context : str, optional
            Additional context (e.g., current URL, page title).
        dom_context : str, optional
            Filtered serialized DOM context (I1: interactable nodes only).
        """
        parts = []

        # Task instruction
        parts.append(f"[USER TASK]\n{task}\n")

        # Extra context
        if extra_context:
            parts.append(f"[CONTEXT]\n{extra_context}\n")

        # I1: DOM structural context (interactable nodes only)
        if dom_context:
            parts.append(f"[PAGE STRUCTURE]\n{dom_context[:2000]}\n")

        # Action history
        if action_history:
            history_text = self._format_history(action_history)
            parts.append(f"[ACTION HISTORY]\n{history_text}\n")
        else:
            parts.append("[ACTION HISTORY]\nNo previous actions.\n")

        # Candidate elements
        truncated = candidates[:self.max_candidates]
        candidates_text = self.format_candidate_list(truncated)
        parts.append(f"[CANDIDATE ELEMENTS]\n{candidates_text}\n")

        # FIX S5: Removed "Think about the best element" instruction
        # which contradicted "Return ONLY JSON" in the system prompt.
        parts.append("Output exactly ONE JSON action:")

        return "\n".join(parts)

    # ── Chat messages ─────────────────────────────────────────

    def build_chat_messages(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        action_history: Optional[List[Dict[str, Any]]] = None,
        screenshot_placeholder: bool = True,
        extra_context: str = "",
        dom_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Build chat-format messages for Qwen2-VL.

        Returns list of message dicts compatible with the chat template.
        """
        user_content = []

        # Add screenshot placeholder
        if screenshot_placeholder:
            user_content.append({
                "type": "image",
                "image": "screenshot_placeholder",
            })

        # Add text prompt
        text_prompt = self.build_text_prompt(
            task=task,
            candidates=candidates,
            action_history=action_history,
            extra_context=extra_context,
            dom_context=dom_context,
        )
        user_content.append({
            "type": "text",
            "text": text_prompt,
        })

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return messages

    # ── Training prompts ──────────────────────────────────────

    def build_training_prompt(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        target_action: Dict[str, Any],
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
        dom_context: str = "",
    ) -> Dict[str, str]:
        """
        Build a training example with prompt + target.

        Returns dict with:
          - "prompt": the input text
          - "target": the target action JSON string
          - "full_text": prompt + target
        """
        prompt = self.build_text_prompt(
            task=task,
            candidates=candidates,
            action_history=action_history,
            extra_context=extra_context,
            dom_context=dom_context,
        )
        target = json.dumps(target_action, ensure_ascii=False)

        return {
            "prompt": prompt,
            "target": target,
            "full_text": f"{prompt}\n{target}",
        }

    def build_training_messages(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        target_action: Dict[str, Any],
        screenshot: Optional[Any] = None,
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
        dom_context: str = "",
    ) -> Dict[str, Any]:
        """
        Build Qwen2-VL chat messages for multimodal training.

        Returns a dict with:
          - "messages_with_target": full conversation including assistant target
          - "messages_prompt_only": conversation WITHOUT assistant turn
          - "target_text": the target action JSON string
        """
        text_prompt = self.build_text_prompt(
            task=task,
            candidates=candidates,
            action_history=action_history,
            extra_context=extra_context,
            dom_context=dom_context,
        )
        target_text = json.dumps(target_action, ensure_ascii=False)

        # User content: image (if available) + text
        user_content = []
        if screenshot is not None:
            user_content.append({
                "type": "image",
                "image": screenshot,
            })
        user_content.append({
            "type": "text",
            "text": text_prompt,
        })

        # Prompt-only messages (no assistant turn)
        messages_prompt_only = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Full messages with target
        messages_with_target = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_text},
        ]

        return {
            "messages_with_target": messages_with_target,
            "messages_prompt_only": messages_prompt_only,
            "target_text": target_text,
        }

    # ── Backward compatibility ────────────────────────────────

    def build_text_prompt_from_dom(
        self,
        task: str,
        serialized_dom: str,
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
    ) -> str:
        """Legacy: build text prompt from serialized DOM string (for eval compat)."""
        parts = []
        parts.append(f"[USER TASK]\n{task}\n")
        if extra_context:
            parts.append(f"[CONTEXT]\n{extra_context}\n")
        if action_history:
            history_text = self._format_history(action_history)
            parts.append(f"[ACTION HISTORY]\n{history_text}\n")
        else:
            parts.append("[ACTION HISTORY]\nNo previous actions.\n")
        dom_text = serialized_dom[:12000]
        if len(serialized_dom) > 12000:
            dom_text += "\n... (DOM truncated)"
        parts.append(f"[CANDIDATE ELEMENTS]\n{dom_text}\n")
        parts.append("[ACTION]\nChoose the correct candidate and action. Output exactly ONE JSON object:")
        return "\n".join(parts)

    # ── History formatting ────────────────────────────────────

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format action history into readable text."""
        entries = history[-self.max_history_entries:]
        lines = []
        for entry in entries:
            desc = entry.get("description", "")
            if desc:
                lines.append(desc)
            else:
                step = entry.get("step", "?")
                action = entry.get("action", "?")
                candidate = entry.get("candidate", entry.get("element_id", "?"))
                line = f"Step {step}: {action} candidate={candidate}"
                value = entry.get("value")
                if value:
                    line += f' value="{value}"'
                lines.append(line)
        return "\n".join(lines)

    # ── Target formatting ─────────────────────────────────────

    @staticmethod
    def format_action_target(
        action: str,
        candidate: int = -1,
        value: str = "",
        direction: str = "",
        amount: int = 0,
    ) -> Dict[str, Any]:
        """
        Build a target action dict for training.

        Uses candidate index instead of element_id.
        """
        d: Dict[str, Any] = {"action": action.upper()}

        if action.upper() in ("CLICK", "TYPE", "SELECT"):
            d["candidate"] = candidate

        if action.upper() == "TYPE" and value:
            d["value"] = value
        elif action.upper() == "SELECT" and value:
            d["value"] = value
        elif action.upper() == "SCROLL":
            d["direction"] = direction or "down"
            d["amount"] = amount or 300

        return d
