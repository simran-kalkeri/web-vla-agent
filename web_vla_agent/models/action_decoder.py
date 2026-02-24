"""
Action Decoder — Parse and Validate Generated JSON Actions.

Handles:
  - Robust JSON parsing from model output (handles markdown fences, etc.)
  - Action validation against DOM node IDs
  - Constrained action type checking
  - Value validation for TYPE/SELECT/SCROLL

Action format:
  {"action": "CLICK", "element_id": 32}
  {"action": "TYPE", "element_id": 15, "value": "Brooklyn"}
  {"action": "SELECT", "element_id": 8, "value": "Economy"}
  {"action": "SCROLL", "direction": "down", "amount": 300}
  {"action": "DONE"}
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


VALID_ACTIONS = {"CLICK", "TYPE", "SELECT", "SCROLL", "DONE"}


class ActionDecoder:
    """
    Parse and validate model-generated action JSON.

    Provides robust parsing with fallbacks and validation
    against the current DOM state.
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

        raw_text = raw_text.strip()

        # Try direct JSON parse first
        action = self._try_parse(raw_text)
        if action:
            return action

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
        cleaned = re.sub(r"```", "", cleaned).strip()
        action = self._try_parse(cleaned)
        if action:
            return action

        # Find JSON object in text  { ... }
        matches = re.findall(r'\{[^{}]*\}', cleaned, re.DOTALL)
        for match in matches:
            action = self._try_parse(match)
            if action:
                return action

        # Try to find nested JSON (with inner braces)
        matches = re.findall(r'\{[^}]*(?:\{[^}]*\}[^}]*)?\}', cleaned, re.DOTALL)
        for match in matches:
            action = self._try_parse(match)
            if action:
                return action

        return None

    def validate(
        self,
        action: Dict[str, Any],
        valid_node_ids: Optional[List[int]] = None,
    ) -> Tuple[bool, str]:
        """
        Validate a parsed action.

        Returns (is_valid, error_message).
        """
        if not action:
            return False, "Empty action"

        # Check action type
        action_type = action.get("action", "").upper()
        if action_type not in self.valid_actions:
            return False, f"Invalid action type: {action_type}"

        # DONE action needs no further validation
        if action_type == "DONE":
            return True, ""

        # SCROLL doesn't require element_id
        if action_type == "SCROLL":
            direction = action.get("direction", "down")
            if direction not in ("up", "down"):
                return False, f"Invalid scroll direction: {direction}"
            return True, ""

        # CLICK, TYPE, SELECT require element_id
        element_id = action.get("element_id")
        if element_id is None:
            return False, f"{action_type} requires element_id"

        try:
            element_id = int(element_id)
        except (ValueError, TypeError):
            return False, f"Invalid element_id: {element_id}"

        # Check against valid node IDs
        if valid_node_ids is not None and element_id not in valid_node_ids:
            return False, f"element_id {element_id} not in DOM"

        # TYPE and SELECT require value
        if action_type == "TYPE":
            value = action.get("value", "")
            if not value:
                return False, "TYPE action requires a value"

        if action_type == "SELECT":
            value = action.get("value", "")
            if not value:
                return False, "SELECT action requires a value"

        return True, ""

    def parse_and_validate(
        self,
        raw_text: str,
        valid_node_ids: Optional[List[int]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool, str]:
        """
        Parse and validate in one call.

        Returns (action_dict, is_valid, error_message).
        """
        action = self.parse(raw_text)
        if action is None:
            return None, False, "Failed to parse JSON from model output"

        is_valid, error = self.validate(action, valid_node_ids)
        return action, is_valid, error

    def normalize(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize action dict to canonical form.
        """
        normalized: Dict[str, Any] = {
            "action": action.get("action", "CLICK").upper(),
        }

        action_type = normalized["action"]

        if action_type in ("CLICK", "TYPE", "SELECT"):
            normalized["element_id"] = int(action.get("element_id", -1))

        if action_type == "TYPE":
            normalized["value"] = str(action.get("value", ""))
        elif action_type == "SELECT":
            normalized["value"] = str(action.get("value", ""))
        elif action_type == "SCROLL":
            normalized["direction"] = action.get("direction", "down")
            normalized["amount"] = int(action.get("amount", 300))

        return normalized

    def _try_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to parse text as JSON and validate basic structure."""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "action" in obj:
                # Normalize action type
                obj["action"] = obj["action"].upper()
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    @staticmethod
    def action_to_text(action: Dict[str, Any]) -> str:
        """Convert action dict to human-readable text."""
        a = action.get("action", "?")
        if a == "CLICK":
            return f"CLICK element_id={action.get('element_id', '?')}"
        elif a == "TYPE":
            return f"TYPE element_id={action.get('element_id', '?')} value=\"{action.get('value', '')}\""
        elif a == "SELECT":
            return f"SELECT element_id={action.get('element_id', '?')} value=\"{action.get('value', '')}\""
        elif a == "SCROLL":
            return f"SCROLL direction={action.get('direction', 'down')} amount={action.get('amount', 300)}"
        elif a == "DONE":
            return "DONE"
        return f"UNKNOWN: {json.dumps(action)}"
