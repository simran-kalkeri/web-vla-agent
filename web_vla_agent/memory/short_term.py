"""
Short-Term Memory — sliding window of recent states and actions.

Provides immediate context to the Action Policy Network so it can
condition on what just happened (e.g. avoid repeating a failed click).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MemoryEntry:
    """A single (state, action) pair stored in short-term memory."""
    state: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = 0  # step index


class ShortTermMemory:
    """
    Fixed-capacity sliding-window buffer of recent (state, action) pairs.

    Parameters
    ----------
    capacity : int
        Maximum number of entries to keep.  Oldest entries are evicted
        automatically.
    """

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self._buffer: deque[MemoryEntry] = deque(maxlen=capacity)
        self._step = 0

    # ── Core API ─────────────────────────────────────────────

    def push(self, state: Dict[str, Any], action: Dict[str, Any]) -> None:
        """Record a new (state, action) pair."""
        self._buffer.append(MemoryEntry(
            state=state, action=action, timestamp=self._step
        ))
        self._step += 1

    def get_recent(self, n: Optional[int] = None) -> List[MemoryEntry]:
        """Return the last *n* entries (or all if *n* is None)."""
        entries = list(self._buffer)
        if n is not None:
            entries = entries[-n:]
        return entries

    def get_action_sequence(self) -> List[Dict[str, Any]]:
        """Return just the actions in chronological order."""
        return [e.action for e in self._buffer]

    def get_action_strings(self) -> List[str]:
        """Return human-readable action strings for the history encoder."""
        parts: List[str] = []
        for e in self._buffer:
            a = e.action
            action_type = a.get("type", "unknown")
            element = a.get("element", "?")
            value = a.get("value", "")
            s = f"Step {e.timestamp}: {action_type} [{element}]"
            if value:
                s += f' "{value}"'
            parts.append(s)
        return parts

    def get_last_state(self) -> Optional[Dict[str, Any]]:
        """Return the most recent state, or None if empty."""
        if self._buffer:
            return self._buffer[-1].state
        return None

    def get_last_action(self) -> Optional[Dict[str, Any]]:
        """Return the most recent action, or None if empty."""
        if self._buffer:
            return self._buffer[-1].action
        return None

    # ── Housekeeping ─────────────────────────────────────────

    def clear(self) -> None:
        self._buffer.clear()
        self._step = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def current_step(self) -> int:
        return self._step
