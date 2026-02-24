"""
Token-Level Uncertainty Estimation for VLA Web Agent.

NOT entropy of classification logits.

Instead computes:
  1. Average token log probability of generated JSON
  2. Beam disagreement (compare top-K beam outputs)
  3. Ensemble variance (optional)

If confidence < threshold → trigger regeneration or alternative beam.

Threshold calibrated on validation set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class UncertaintyResult:
    """Result of uncertainty estimation."""
    avg_log_prob: float = 0.0
    should_regenerate: bool = False
    beam_agreement: float = 1.0       # 1.0 = all beams agree
    selected_action: Optional[Dict[str, Any]] = None
    reason: str = ""


class TokenUncertainty:
    """
    Token-level uncertainty for generative VLA model.

    Uses average log probability of generated tokens and beam
    disagreement to decide if the model is confident.

    Parameters
    ----------
    min_log_prob : float
        Minimum average log probability threshold. Below this → regenerate.
    beam_width : int
        Number of beams for disagreement check.
    max_regenerations : int
        Maximum number of regeneration attempts.
    """

    def __init__(
        self,
        min_log_prob: float = -2.0,
        beam_width: int = 3,
        max_regenerations: int = 2,
    ):
        self.min_log_prob = min_log_prob
        self.beam_width = beam_width
        self.max_regenerations = max_regenerations
        self._calibrated = False

    def assess(
        self,
        generation_result: Dict[str, Any],
    ) -> UncertaintyResult:
        """
        Assess uncertainty from a single generation.

        Parameters
        ----------
        generation_result : dict
            Must contain "avg_log_prob" and "text".

        Returns
        -------
        UncertaintyResult
        """
        avg_lp = generation_result.get("avg_log_prob", 0.0)
        text = generation_result.get("text", "")

        should_regenerate = avg_lp < self.min_log_prob

        reason = ""
        if should_regenerate:
            reason = f"avg_log_prob={avg_lp:.3f} < threshold={self.min_log_prob}"

        return UncertaintyResult(
            avg_log_prob=avg_lp,
            should_regenerate=should_regenerate,
            reason=reason,
        )

    def assess_beams(
        self,
        beam_results: List[Dict[str, Any]],
    ) -> UncertaintyResult:
        """
        Assess uncertainty from beam search results.

        Checks:
          1. Agreement: do all beams produce the same action type + element_id?
          2. Score spread: how different are the beam scores?

        Parameters
        ----------
        beam_results : list of dicts
            Each with "text" and "score".
        """
        if not beam_results:
            return UncertaintyResult(
                should_regenerate=True,
                reason="No beam results",
            )

        # Parse actions from all beams
        parsed_actions = []
        for beam in beam_results:
            try:
                action = json.loads(beam["text"])
                parsed_actions.append(action)
            except (json.JSONDecodeError, ValueError):
                parsed_actions.append(None)

        # Check agreement on action type and element_id
        valid_actions = [a for a in parsed_actions if a is not None]
        if not valid_actions:
            return UncertaintyResult(
                should_regenerate=True,
                beam_agreement=0.0,
                reason="No valid actions from any beam",
            )

        # Agreement metric: what fraction agree on action + element_id?
        signatures = []
        for a in valid_actions:
            sig = f"{a.get('action', '?')}_{a.get('element_id', '?')}"
            signatures.append(sig)

        most_common = max(set(signatures), key=signatures.count)
        agreement = signatures.count(most_common) / len(signatures)

        # Best action (from highest-scoring beam with most common signature)
        best_action = None
        best_score = float("-inf")
        for i, (beam, action) in enumerate(zip(beam_results, parsed_actions)):
            if action is None:
                continue
            sig = f"{action.get('action', '?')}_{action.get('element_id', '?')}"
            if sig == most_common and beam.get("score", 0) > best_score:
                best_action = action
                best_score = beam["score"]

        should_regenerate = agreement < 0.5

        return UncertaintyResult(
            avg_log_prob=best_score,
            should_regenerate=should_regenerate,
            beam_agreement=agreement,
            selected_action=best_action,
            reason=f"beam_agreement={agreement:.2f}" if should_regenerate else "",
        )

    def calibrate(
        self,
        log_probs: List[float],
        percentile: float = 10.0,
    ) -> float:
        """
        Calibrate threshold from validation set log probabilities.

        Sets threshold at the given percentile of validation log probs.
        """
        if not log_probs:
            return self.min_log_prob

        sorted_lps = sorted(log_probs)
        idx = max(0, int(len(sorted_lps) * percentile / 100.0))
        self.min_log_prob = sorted_lps[idx]
        self._calibrated = True
        return self.min_log_prob

    def statistics(
        self,
        log_probs: List[float],
    ) -> Dict[str, float]:
        """Compute statistics over a batch of log probabilities."""
        if not log_probs:
            return {"mean": 0, "min": 0, "max": 0, "regenerate_fraction": 0}

        return {
            "mean": sum(log_probs) / len(log_probs),
            "min": min(log_probs),
            "max": max(log_probs),
            "regenerate_fraction": sum(
                1 for lp in log_probs if lp < self.min_log_prob
            ) / len(log_probs),
        }
