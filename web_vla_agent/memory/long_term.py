"""
Long-Term Memory — persistent subgoal progress tracker.

Maintains the state of all subgoals across the lifetime of a task
execution.  Supports serialisation to JSON for checkpoint recovery.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.planner import Subgoal, SubgoalStatus


@dataclass
class SubgoalRecord:
    """Extended subgoal tracking with failure history."""
    subgoal: Subgoal
    attempts: int = 0
    max_attempts: int = 3
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts


class LongTermMemory:
    """
    Track subgoal lifecycle: pending → active → completed | failed.

    Parameters
    ----------
    max_retries : int
        Maximum retry attempts per subgoal before declaring failure.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._records: Dict[str, SubgoalRecord] = {}
        self._order: List[str] = []  # insertion order
        self._current_id: Optional[str] = None

    # ── Subgoal lifecycle ────────────────────────────────────

    def add_subgoal(self, subgoal_id: str, description: str, **kwargs: Any) -> None:
        """Register a new subgoal."""
        sg = Subgoal(subgoal_id=subgoal_id, description=description, **kwargs)
        self._records[subgoal_id] = SubgoalRecord(
            subgoal=sg, max_attempts=self.max_retries
        )
        if subgoal_id not in self._order:
            self._order.append(subgoal_id)

    def add_subgoals(self, subgoals: List[Subgoal]) -> None:
        """Bulk-register subgoals from a planner output."""
        for sg in subgoals:
            self._records[sg.subgoal_id] = SubgoalRecord(
                subgoal=sg, max_attempts=self.max_retries
            )
            if sg.subgoal_id not in self._order:
                self._order.append(sg.subgoal_id)

    def activate(self, subgoal_id: str) -> None:
        """Set a subgoal as the currently active one."""
        if subgoal_id in self._records:
            self._records[subgoal_id].subgoal.status = SubgoalStatus.ACTIVE.value
            self._current_id = subgoal_id

    def mark_complete(self, subgoal_id: str) -> None:
        """Mark a subgoal as successfully completed."""
        if subgoal_id in self._records:
            self._records[subgoal_id].subgoal.status = SubgoalStatus.COMPLETED.value
            if self._current_id == subgoal_id:
                self._current_id = None

    def mark_failed(self, subgoal_id: str, reason: str = "") -> None:
        """Record a failure attempt.  Status set to FAILED if exhausted."""
        if subgoal_id not in self._records:
            return
        rec = self._records[subgoal_id]
        rec.attempts += 1
        if reason:
            rec.failure_reasons.append(reason)
        if rec.exhausted:
            rec.subgoal.status = SubgoalStatus.FAILED.value
        else:
            rec.subgoal.status = SubgoalStatus.PENDING.value  # allow retry

    def skip(self, subgoal_id: str) -> None:
        if subgoal_id in self._records:
            self._records[subgoal_id].subgoal.status = SubgoalStatus.SKIPPED.value

    # ── Queries ──────────────────────────────────────────────

    @property
    def current_subgoal(self) -> Optional[Subgoal]:
        if self._current_id and self._current_id in self._records:
            return self._records[self._current_id].subgoal
        return None

    def next_pending(self) -> Optional[Subgoal]:
        """Return the next pending subgoal in order, or None."""
        for sid in self._order:
            rec = self._records[sid]
            if rec.subgoal.status == SubgoalStatus.PENDING.value and not rec.exhausted:
                return rec.subgoal
        return None

    def should_replan(self) -> bool:
        """True if the current subgoal has exhausted all retries."""
        if self._current_id and self._current_id in self._records:
            return self._records[self._current_id].exhausted
        return False

    def get_progress(self) -> Dict[str, int]:
        """Aggregate progress counters."""
        statuses = {"completed": 0, "failed": 0, "pending": 0, "active": 0, "skipped": 0}
        for rec in self._records.values():
            s = rec.subgoal.status
            if s in statuses:
                statuses[s] += 1
        statuses["total"] = len(self._records)
        return statuses

    def all_done(self) -> bool:
        """True if no pending or active subgoals remain."""
        for rec in self._records.values():
            if rec.subgoal.status in (SubgoalStatus.PENDING.value, SubgoalStatus.ACTIVE.value):
                return False
        return True

    def get_completed_descriptions(self) -> List[str]:
        """Return descriptions of all completed subgoals (for context)."""
        return [
            rec.subgoal.description
            for sid in self._order
            if (rec := self._records[sid]).subgoal.status == SubgoalStatus.COMPLETED.value
        ]

    # ── Serialisation ────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_id": self._current_id,
            "order": self._order,
            "records": {
                sid: {
                    "subgoal": rec.subgoal.to_dict(),
                    "attempts": rec.attempts,
                    "max_attempts": rec.max_attempts,
                    "failure_reasons": rec.failure_reasons,
                }
                for sid, rec in self._records.items()
            },
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "LongTermMemory":
        data = json.loads(Path(path).read_text())
        mem = cls()
        mem._current_id = data.get("current_id")
        mem._order = data.get("order", [])
        for sid, rec_data in data.get("records", {}).items():
            sg = Subgoal.from_dict(rec_data["subgoal"])
            mem._records[sid] = SubgoalRecord(
                subgoal=sg,
                attempts=rec_data.get("attempts", 0),
                max_attempts=rec_data.get("max_attempts", 3),
                failure_reasons=rec_data.get("failure_reasons", []),
            )
        return mem

    # ── Housekeeping ─────────────────────────────────────────

    def clear(self) -> None:
        self._records.clear()
        self._order.clear()
        self._current_id = None

    def __len__(self) -> int:
        return len(self._records)
