"""
Failure Detector — heuristic + pattern-based failure detection.

Detects:
  - Stale state (page unchanged after action)
  - Error pages
  - Action loops (repeated action sequences)
  - Subgoal timeout

Triggers replanning via the LongTermMemory module.
"""
from __future__ import annotations

import hashlib
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureType(str, Enum):
    NONE = "none"
    STALE_STATE = "stale_state"
    ERROR_PAGE = "error_page"
    ACTION_LOOP = "action_loop"
    TIMEOUT = "timeout"


# Common error page indicators
_ERROR_INDICATORS = [
    "404", "not found", "page not found",
    "500", "internal server error",
    "403", "forbidden", "access denied",
    "error", "something went wrong",
    "oops", "we can't find",
    "this page isn't available",
    "timed out", "connection refused",
]


class FailureDetector:
    """
    Heuristic failure detector for web navigation.

    Parameters
    ----------
    loop_window : int
        Size of the sliding window for loop detection.
    stale_threshold : int
        Number of consecutive unchanged states before declaring stale.
    max_steps : int
        Maximum total steps before timeout.
    """

    def __init__(
        self,
        loop_window: int = 4,
        stale_threshold: int = 3,
        max_steps: int = 30,
    ):
        self.loop_window = loop_window
        self.stale_threshold = stale_threshold
        self.max_steps = max_steps

        self._action_history: deque[str] = deque(maxlen=loop_window * 2)
        self._state_hashes: deque[str] = deque(maxlen=stale_threshold + 1)
        self._total_steps = 0

    # ── Core API ─────────────────────────────────────────────

    def detect(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
        action: Dict[str, Any],
    ) -> FailureType:
        """
        Analyse the transition and return a :class:`FailureType`.

        Parameters
        ----------
        prev_state : state dict before action (can be None on first step)
        curr_state : state dict after action
        action     : the action that was executed
        """
        self._total_steps += 1

        # Record action signature
        action_sig = self._action_signature(action)
        self._action_history.append(action_sig)

        # Record state hash
        state_hash = self._state_hash(curr_state)
        self._state_hashes.append(state_hash)

        # 1. Timeout check
        if self._total_steps >= self.max_steps:
            return FailureType.TIMEOUT

        # 2. Error page check
        if self._is_error_page(curr_state):
            return FailureType.ERROR_PAGE

        # 3. Stale state check
        if self._is_stale():
            return FailureType.STALE_STATE

        # 4. Action loop check
        if self._is_action_loop():
            return FailureType.ACTION_LOOP

        return FailureType.NONE

    def reset(self) -> None:
        """Reset all internal counters for a new task."""
        self._action_history.clear()
        self._state_hashes.clear()
        self._total_steps = 0

    # ── Private checks ───────────────────────────────────────

    def _is_error_page(self, state: Dict[str, Any]) -> bool:
        """Check page title and URL for error indicators."""
        title = (state.get("page_title", "") or "").lower()
        url = (state.get("url", "") or "").lower()
        html = (state.get("html_snippet", "") or "").lower()

        combined = f"{title} {url} {html}"
        return any(indicator in combined for indicator in _ERROR_INDICATORS)

    def _is_stale(self) -> bool:
        """True if the last `stale_threshold` states have the same hash."""
        if len(self._state_hashes) < self.stale_threshold:
            return False
        recent = list(self._state_hashes)[-self.stale_threshold :]
        return len(set(recent)) == 1

    def _is_action_loop(self) -> bool:
        """
        True if a repeating pattern of length ≤ loop_window is found
        in the recent action history.
        """
        history = list(self._action_history)
        n = len(history)
        if n < self.loop_window * 2:
            return False

        for pattern_len in range(1, self.loop_window + 1):
            if n < pattern_len * 2:
                continue
            tail = history[-pattern_len:]
            prev = history[-pattern_len * 2 : -pattern_len]
            if tail == prev:
                return True
        return False

    # ── Hashing ─────────────────────────────────────────────

    @staticmethod
    def _state_hash(state: Dict[str, Any]) -> str:
        """Deterministic hash of state for comparison."""
        # Use URL + DOM length as a lightweight proxy
        content = f"{state.get('url', '')}|{len(state.get('html_snippet', ''))}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _action_signature(action: Dict[str, Any]) -> str:
        """Deterministic string signature for an action."""
        return f"{action.get('type', '?')}|{action.get('element', '?')}|{action.get('value', '')}"
