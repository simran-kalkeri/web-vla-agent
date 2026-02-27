"""
Prompt Builder for VLA Web Agent.

Constructs multimodal prompts for the Qwen2-VL backbone:
  [SYSTEM] → agent role
  [TASK] → user instruction
  [DOM] → serialized DOM tokens (structured, not flat)
  [ACTION_HISTORY] → previous actions
  [IMAGE] → screenshot (via vision processor)

The model generates JSON action as output:
  {"action": "CLICK", "element_id": 32}
  {"action": "TYPE", "element_id": 15, "value": "Brooklyn"}
  {"action": "SCROLL", "direction": "down", "amount": 300}
  {"action": "SELECT", "element_id": 8, "value": "Economy"}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """\
You are an expert web navigation agent. You observe a webpage's DOM structure and screenshot, \
then output a single JSON action to progress toward completing the user's task.

Available actions:
- {"action": "CLICK", "element_id": <id>}
- {"action": "TYPE", "element_id": <id>, "value": "<text>"}
- {"action": "SELECT", "element_id": <id>, "value": "<option>"}
- {"action": "SCROLL", "direction": "<up|down>", "amount": <pixels>}

Rules:
1. Output ONLY a single valid JSON object — no markdown, no explanation.
2. element_id must match a node id from the DOM.
3. For TYPE actions, include the value to type.
4. For SELECT, include the option text to select.
5. For SCROLL, specify direction and amount in pixels.
6. Choose the action that best progresses toward the task goal.
7. If the task appears complete, output: {"action": "DONE"}
"""


class PromptBuilder:
    """
    Build multimodal prompts for the VLA model.

    Handles text prompt construction. Screenshot encoding is done
    separately by the model's vision processor.
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        max_dom_chars: int = 12000,
        max_history_entries: int = 10,
    ):
        self.system_prompt = system_prompt
        self.max_dom_chars = max_dom_chars
        self.max_history_entries = max_history_entries

    def build_text_prompt(
        self,
        task: str,
        serialized_dom: str,
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
    ) -> str:
        """
        Build the text portion of the multimodal prompt.

        Parameters
        ----------
        task : str
            User's task instruction.
        serialized_dom : str
            Structured DOM serialization from DOMSerializer.
        action_history : list of dicts, optional
            Previous actions taken.
        extra_context : str, optional
            Additional context (e.g., current URL, page title).

        Returns
        -------
        str
            Complete text prompt for the model.
        """
        parts = []

        # Task instruction
        parts.append(f"[TASK]\n{task}\n")

        # Extra context
        if extra_context:
            parts.append(f"[CONTEXT]\n{extra_context}\n")

        # Action history
        if action_history:
            history_text = self._format_history(action_history)
            parts.append(f"[ACTION_HISTORY]\n{history_text}\n")
        else:
            parts.append("[ACTION_HISTORY]\nNo previous actions.\n")

        # DOM (may be truncated)
        dom_text = serialized_dom[:self.max_dom_chars]
        if len(serialized_dom) > self.max_dom_chars:
            dom_text += "\n... (DOM truncated)"
        parts.append(f"[DOM]\n{dom_text}\n")

        # Instruction to generate action
        parts.append("[ACTION]\nGenerate the next action as a JSON object:")

        return "\n".join(parts)

    def build_chat_messages(
        self,
        task: str,
        serialized_dom: str,
        action_history: Optional[List[Dict[str, Any]]] = None,
        screenshot_placeholder: bool = True,
        extra_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Build chat-format messages for Qwen2-VL.

        Returns list of message dicts compatible with the chat template.
        The screenshot is represented as an image placeholder that
        the model processor will fill in.
        """
        user_content = []

        # Add screenshot placeholder (will be replaced by actual image)
        if screenshot_placeholder:
            user_content.append({
                "type": "image",
                "image": "screenshot_placeholder",
            })

        # Add text prompt
        text_prompt = self.build_text_prompt(
            task=task,
            serialized_dom=serialized_dom,
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

    def build_training_prompt(
        self,
        task: str,
        serialized_dom: str,
        target_action: Dict[str, Any],
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
    ) -> Dict[str, str]:
        """
        Build a training example with prompt + target.

        Returns
        -------
        dict with keys:
          - "prompt": the input text
          - "target": the target action JSON string
          - "full_text": prompt + target (for causal LM training)
        """
        import json

        prompt = self.build_text_prompt(
            task=task,
            serialized_dom=serialized_dom,
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
        serialized_dom: str,
        target_action: Dict[str, Any],
        screenshot: Optional[Any] = None,
        action_history: Optional[List[Dict[str, Any]]] = None,
        extra_context: str = "",
    ) -> Dict[str, Any]:
        """
        Build Qwen2-VL chat messages for multimodal training.

        Returns a dict with:
          - "messages_with_target": full conversation including assistant
            target (for full tokenization and label masking)
          - "messages_prompt_only": conversation WITHOUT assistant turn
            (for computing prompt length to mask labels)
          - "target_text": the target action JSON string

        These are in Qwen2-VL chat format, compatible with
        ``processor.apply_chat_template()``.
        """
        import json

        text_prompt = self.build_text_prompt(
            task=task,
            serialized_dom=serialized_dom,
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

        # Prompt-only messages (no assistant turn) — for computing prompt length
        messages_prompt_only = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Full messages with target — for tokenizing the complete training sequence
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
                eid = entry.get("element_id", "?")
                line = f"Step {step}: {action} element_id={eid}"
                value = entry.get("value")
                if value:
                    line += f' value="{value}"'
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def format_action_target(
        action: str,
        element_id: int = -1,
        value: str = "",
        direction: str = "",
        amount: int = 0,
    ) -> Dict[str, Any]:
        """
        Build a target action dict for training.

        Constructs the expected model output format.
        """
        d: Dict[str, Any] = {"action": action.upper()}

        if action.upper() in ("CLICK", "TYPE", "SELECT"):
            d["element_id"] = element_id

        if action.upper() == "TYPE" and value:
            d["value"] = value
        elif action.upper() == "SELECT" and value:
            d["value"] = value
        elif action.upper() == "SCROLL":
            d["direction"] = direction or "down"
            d["amount"] = amount or 300

        return d
