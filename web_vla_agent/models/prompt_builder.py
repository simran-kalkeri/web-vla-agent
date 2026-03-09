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


SYSTEM_PROMPT = """\
You are a precise and deterministic web automation agent. Your job is to complete a user task by interacting with elements on a web page.

You are given:

1. The USER TASK
2. A list of DOM CANDIDATE ELEMENTS extracted from the webpage
3. The ACTION HISTORY

Each candidate element corresponds to a real element on the webpage.

Your job is to choose the correct element and perform the correct action.

---

## CRITICAL GROUNDING RULES

You MUST only interact with elements listed in the CANDIDATE list.

Each candidate is mapped to a real DOM node.

Each candidate has:

candidate_index → node_id → real webpage element

You must NEVER invent a candidate_index.

If you choose candidate = N, the system will map it to the corresponding DOM node automatically.

Valid candidate values are ONLY those listed in the candidate list.

---

## ACTION SPACE

You may output only one of the following actions:

CLICK
TYPE
SELECT
SCROLL
STOP

Rules:

CLICK
Click a clickable element (buttons, links, checkboxes, etc.)

TYPE
Enter text into an input field.

TYPE must include a "value" field.

Example:
{"action":"TYPE","candidate":0,"value":"New York"}

SELECT
Choose an option in a dropdown element.

Example:
{"action":"SELECT","candidate":3,"value":"Economy"}

SCROLL
Scroll the page if the needed element is not visible.

STOP
Return STOP only when the task is complete.

---

## ELEMENT SELECTION STRATEGY

To choose the correct candidate:

1. Read the USER TASK carefully.
2. Identify which element is required to progress toward the task.
3. Compare the task with each candidate's:

   * text
   * tag
   * attributes
4. Select the candidate that best matches the intended action.

Prefer elements with tags:

input
button
select
a

Do NOT click decorative or container elements like div unless necessary.

---

## STRICT OUTPUT FORMAT

You must output ONLY valid JSON.

DO NOT output explanations.

DO NOT output multiple actions.

Correct format examples:

{"action":"CLICK","candidate":2}

{"action":"TYPE","candidate":0,"value":"New York"}

{"action":"SELECT","candidate":3,"value":"Economy"}

{"action":"SCROLL"}

{"action":"STOP"}

---

## INVALID OUTPUTS

The following are NOT allowed:

* Missing candidate index
* Outputting text outside JSON
* Multiple actions
* Invalid candidate index

---

## ERROR PREVENTION

If no candidate clearly matches the task:

Use SCROLL instead of guessing.

Never produce invalid candidate indices.

Never output candidate = -1.

---

## OBJECTIVE

Your goal is to select the correct candidate element and perform the correct action to complete the task efficiently.

Always choose the action that most directly advances the task.

Return exactly ONE JSON action.\
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
        max_candidates: int = 50,
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
        """
        Format a single candidate element for the prompt.

        Uses semantic uppercase tag names and includes position.
        Example output:
            [3] BUTTON "Search Flights" at (640,45 120x32) role=tab id=search-btn
        """
        # Semantic tag name
        display_tag = PromptBuilder._TAG_DISPLAY.get(tag.lower(), tag.upper())

        attrs = attributes or {}
        # Keep only useful attributes
        keep = {"role", "type", "placeholder", "aria-label", "href",
                "name", "value", "title", "alt", "id", "for"}
        attr_parts = []
        for k, v in sorted(attrs.items()):
            if k in keep and v:
                attr_parts.append(f'{k}="{str(v)[:60]}"')
        attr_str = " ".join(attr_parts)

        text_str = f' "{text[:80]}"' if text else ""

        # Position info from bounding box
        pos_str = ""
        if bbox and len(bbox) >= 4:
            x, y, w, h = [int(v) for v in bbox[:4]]
            if w > 0 and h > 0:
                pos_str = f" at ({x},{y} {w}x{h})"

        parts = [f"[{index}] {display_tag}{text_str}{pos_str}"]
        if attr_str:
            parts.append(attr_str)
        return " ".join(parts)

    @staticmethod
    def format_candidate_list(candidates: List[Dict[str, Any]]) -> str:
        """
        Format a list of candidate elements for the prompt.

        Each candidate dict should have:
          - candidate_index: int
          - tag: str
          - text: str (optional)
          - attributes: dict (optional)
          - bbox: list (optional) — [x, y, w, h]
        """
        lines = []
        for c in candidates:
            line = PromptBuilder.format_candidate(
                index=c["candidate_index"],
                tag=c.get("tag", "div"),
                text=c.get("text", ""),
                attributes=c.get("attributes"),
                bbox=c.get("bbox"),
            )
            lines.append(line)
        return "\n".join(lines)

    # ── Text prompt ───────────────────────────────────────────

    def build_text_prompt(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
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
        """
        parts = []

        # Task instruction
        parts.append(f"[USER TASK]\n{task}\n")

        # Extra context
        if extra_context:
            parts.append(f"[CONTEXT]\n{extra_context}\n")

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

        # Instruction
        parts.append("[ACTION]\nChoose the correct candidate and action. Output exactly ONE JSON object:")

        return "\n".join(parts)

    # ── Chat messages ─────────────────────────────────────────

    def build_chat_messages(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        action_history: Optional[List[Dict[str, Any]]] = None,
        screenshot_placeholder: bool = True,
        extra_context: str = "",
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
