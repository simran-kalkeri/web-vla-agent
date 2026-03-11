"""
Task Planner — Lightweight Planning Layer for Multi-Step Web Tasks (I5).

Decomposes a task into ordered subgoals using the VLA model itself.
Runs once at the start of run_task(). Each subgoal is a short
natural language instruction. The agent tracks which subgoal it is
currently executing.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


PLANNING_SYSTEM_PROMPT = """You are a web task planner.
Given a task, decompose it into 2-6 sequential subgoals.
Each subgoal should correspond to one interaction on the webpage.
Output ONLY a JSON array of strings.
Example: ["Click the search box", "Type 'New York'", "Click Search button"]
"""


class TaskPlanner:
    """
    Decomposes a task into ordered subgoals using the VLA model itself.
    Runs once at the start of run_task(). Each subgoal is a short
    natural language instruction. The agent tracks which subgoal it is
    currently executing.
    """

    def __init__(self, model, prompt_builder):
        self.model = model
        self.prompt_builder = prompt_builder

    def plan(self, task: str, url: str) -> List[str]:
        """Generate subgoals for the task. Returns list of subgoal strings."""
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task}\nStarting URL: {url}"},
        ]
        try:
            result = self.model.generate(
                messages=messages,
                image=None,
                temperature=0.1,
                max_new_tokens=128,
            )
            text = result.get("text", "")

            # Extract JSON array from output — non-greedy to avoid
            # matching across multiple bracket pairs in the response
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                subgoals = json.loads(match.group())
                if isinstance(subgoals, list) and all(isinstance(s, str) for s in subgoals):
                    logger.info(f"TaskPlanner: decomposed into {len(subgoals)} subgoals")
                    return subgoals
        except Exception as e:
            logger.warning(f"TaskPlanner: planning failed ({e}), using task as single subgoal")

        # Fallback: treat whole task as single subgoal
        return [task]

    def get_current_subgoal(
        self,
        subgoals: List[str],
        completed_steps: int,
        total_steps: int,
    ) -> str:
        """
        Returns the current active subgoal based on progress.

        Paces subgoal advancement by dividing total_steps evenly across
        subgoals, so with total_steps=30 and 3 subgoals, advancement
        happens at steps 0, 10, 20.
        """
        if not subgoals:
            return ""
        n = len(subgoals)
        # Advance one subgoal every (total_steps / n) steps
        steps_per_subgoal = max(total_steps / n, 1)
        idx = min(int(completed_steps / steps_per_subgoal), n - 1)
        return subgoals[idx]
