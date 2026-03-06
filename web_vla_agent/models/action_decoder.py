"""
Action Decoder — Parse and Validate Generated JSON Actions (Candidate-Based).

Handles:
  - Robust JSON parsing from model output (handles markdown fences, etc.)
  - Action validation against candidate indices
  - Constrained action type checking

Expected formats:
  {"action": "CLICK", "candidate": 3}
  {"action": "TYPE", "candidate": 0, "value": "Brooklyn"}
  {"action": "SELECT", "candidate": 8, "value": "Economy"}
  {"action": "SCROLL", "direction": "down", "amount": 300}
  {"action": "SCROLL"}
  {"action": "STOP"}
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


VALID_ACTIONS = {"CLICK", "TYPE", "SELECT", "SCROLL", "STOP", "DONE"}


class ActionDecoder:
    """
    Parse and validate model-generated action JSON.

    Provides robust parsing with fallbacks and validation
    against the candidate list.
    """

    def __init__(self, valid_actions: Optional[set] = None):
        self.valid_actions = valid_actions or VALID_ACTIONS

    def parse(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse raw model output into an action dict.

        Handles:
          - Clean JSON
          - JSON wrapped in markdown code fences
          - JSON embedded in explanation text
          - Multiple JSON objects (takes first valid one)

        Returns None if parsing fails completely.
        """
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # 1. Try direct parse
        result = self._try_parse(text)
        if result:
            return result

        # 2. Try extracting from markdown code fences
        fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        fences = re.findall(fence_pattern, text, re.DOTALL)
        for fence in fences:
            result = self._try_parse(fence.strip())
            if result:
                return result

        # 3. Try extracting JSON-like patterns
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            result = self._try_parse(match)
            if result:
                return result

        # 4. Try fixing brackets
        fixed = self._fix_brackets(text)
        if fixed != text:
            result = self._try_parse(fixed)
            if result:
                return result

        return None

    def validate(
        self,
        action: Dict[str, Any],
        num_candidates: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Validate a parsed action.

        Parameters
        ----------
        action : dict
            Parsed action dict.
        num_candidates : int, optional
            Number of valid candidates. If provided, validates candidate index.

        Returns (is_valid, error_message).
        """
        action_type = action.get("action", "").upper()

        # Normalize: accept "DONE" as "STOP"
        if action_type == "DONE":
            action_type = "STOP"
            action["action"] = "STOP"

        if action_type not in self.valid_actions:
            return False, f"Invalid action type: {action_type}"

        # STOP/SCROLL don't need a candidate
        if action_type in ("STOP", "DONE", "SCROLL"):
            return True, ""

        # CLICK, TYPE, SELECT need a candidate
        candidate = action.get("candidate")
        if candidate is None:
            # Backward compat: try element_id
            candidate = action.get("element_id")
            if candidate is not None:
                action["candidate"] = candidate

        if candidate is None:
            return False, "Missing candidate index"

        try:
            candidate = int(candidate)
            action["candidate"] = candidate
        except (ValueError, TypeError):
            return False, f"Invalid candidate index: {candidate}"

        if candidate < 0:
            return False, f"Candidate index {candidate} is negative"

        if num_candidates is not None and candidate >= num_candidates:
            return False, f"Candidate index {candidate} >= num_candidates ({num_candidates})"

        # TYPE/SELECT need value
        if action_type == "TYPE" and not action.get("value"):
            return False, "TYPE action missing value"

        if action_type == "SELECT" and not action.get("value"):
            return False, "SELECT action missing value"

        return True, ""

    def parse_and_validate(
        self,
        raw_text: str,
        num_candidates: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool, str]:
        """
        Parse and validate in one call.

        Returns (action_dict, is_valid, error_message).
        """
        action = self.parse(raw_text)
        if action is None:
            return None, False, "Failed to parse JSON"

        action = self.normalize(action)
        is_valid, error = self.validate(action, num_candidates)
        return action, is_valid, error

    def normalize(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize action dict to canonical form.
        """
        result = dict(action)

        # Normalize action type
        action_type = result.get("action", "")
        if not action_type:
            action_type = result.get("type", "")
        result["action"] = action_type.upper()

        # Handle DONE → STOP
        if result["action"] == "DONE":
            result["action"] = "STOP"

        # Normalize candidate: accept both "candidate" and "element_id"
        if "candidate" in result:
            try:
                result["candidate"] = int(result["candidate"])
            except (ValueError, TypeError):
                pass
        elif "element_id" in result:
            try:
                result["candidate"] = int(result["element_id"])
            except (ValueError, TypeError):
                pass

        # Remove element_id if we have candidate
        if "candidate" in result and "element_id" in result:
            del result["element_id"]

        return result

    def _try_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to parse text as JSON and validate basic structure.

        Accepts {"action": "CLICK", "candidate": 3} and
        {"type": "click", "candidate": 3} formats.
        """
        try:
            # Fix bracket issues first
            fixed = self._fix_brackets(text)
            data = json.loads(fixed)
            if isinstance(data, dict):
                # Must have "action" or "type" key
                if "action" in data or "type" in data:
                    return data
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _fix_brackets(text: str) -> str:
        """Fix mismatched brackets from model output.

        The LoRA model sometimes outputs ["type": "click", ...}
        (starts with [ but contains key:value pairs and ends with }).
        This fixes it to {"type": "click", ...}.
        """
        text = text.strip()
        if not text:
            return text

        # Fix [{ → {
        if text.startswith("[") and ":" in text:
            text = "{" + text[1:]

        # Fix }] → }
        if text.endswith("]") and ":" in text:
            text = text[:-1] + "}"

        return text

    @staticmethod
    def action_to_text(action: Dict[str, Any]) -> str:
        """Convert action dict to human-readable text."""
        action_type = action.get("action", "UNKNOWN")
        candidate = action.get("candidate", "?")

        if action_type == "CLICK":
            return f"CLICK candidate={candidate}"
        elif action_type == "TYPE":
            return f'TYPE candidate={candidate} value="{action.get("value", "")}"'
        elif action_type == "SELECT":
            return f'SELECT candidate={candidate} value="{action.get("value", "")}"'
        elif action_type == "SCROLL":
            direction = action.get("direction", "down")
            amount = action.get("amount", 300)
            return f"SCROLL direction={direction} amount={amount}"
        elif action_type in ("STOP", "DONE"):
            return "STOP"
        return f"{action_type} candidate={candidate}"
