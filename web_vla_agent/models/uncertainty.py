"""
Entropy-Based Uncertainty Estimation (Refactored).

Stripped to predictive entropy only. No learned confidence head,
no MC-Dropout, no calibration loss.

**Mathematical formulation**:
    Candidate distribution:  p_i = softmax(score_i / τ)
    Predictive entropy:      H(p) = -Σ p_i log p_i
    Replanning trigger:      replan if H(p) > τ_thresh

Entropy threshold τ is auto-computed from validation set
(90th percentile) at the end of Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class UncertaintyResult:
    """Bundle of uncertainty diagnostics for a single step."""
    predictive_entropy: float = 0.0
    max_probability: float = 0.0
    margin: float = 0.0            # gap between top-1 and top-2
    should_replan: bool = False


class EntropyUncertainty:
    """
    Entropy-based uncertainty estimation for grounding decisions.

    Purely functional — no learnable parameters. Uses predictive entropy
    over softmax candidate distribution to trigger replanning.

    Parameters
    ----------
    threshold_tau : float
        Entropy threshold for replanning trigger.
        Auto-calibrated from validation set at end of Phase 2.
    temperature : float
        Temperature for softmax over grounding scores.
    """

    def __init__(
        self,
        threshold_tau: float = 1.5,
        temperature: float = 0.07,
    ):
        self.threshold_tau = threshold_tau
        self.temperature = temperature

    # ── Entropy computation ──────────────────────────────────

    @staticmethod
    def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy: H(p) = -Σ p_i log p_i.

        Parameters
        ----------
        probs : [B, N] — probability distribution over candidates

        Returns
        -------
        [B] — entropy per sample
        """
        log_probs = torch.log(probs.clamp(min=1e-8))
        return -(probs * log_probs).sum(dim=-1)

    def compute_uncertainty(
        self,
        scores: torch.Tensor,  # [B, N] — raw grounding scores
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute uncertainty metrics from grounding scores.

        Returns
        -------
        entropy    : [B] — predictive entropy
        max_prob   : [B] — maximum probability (confidence)
        margin     : [B] — gap between top-1 and top-2 probability
        """
        probs = F.softmax(scores / self.temperature, dim=-1)  # [B, N]
        entropy = self.predictive_entropy(probs)

        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        max_prob = sorted_probs[:, 0]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1] if probs.shape[-1] > 1 else max_prob

        return entropy, max_prob, margin

    def should_replan(
        self,
        scores: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Determine if replanning is needed based on entropy threshold.

        Returns
        -------
        [B] — boolean tensor, True = should replan
        """
        entropy, _, _ = self.compute_uncertainty(scores)
        return entropy > self.threshold_tau

    def calibrate_threshold(
        self,
        all_entropies: torch.Tensor,  # [M] — entropy values from validation
        percentile: float = 90.0,
    ) -> float:
        """
        Auto-compute entropy threshold from validation percentile.

        Sets τ to the given percentile of validation entropy distribution.
        Target: replan_fraction between 5-15%.

        Parameters
        ----------
        all_entropies : [M] — entropy values from validation set
        percentile : float — percentile to use (default 90th)

        Returns
        -------
        float — calibrated threshold
        """
        tau = torch.quantile(all_entropies.float(), percentile / 100.0).item()
        self.threshold_tau = tau
        return tau

    def entropy_statistics(
        self,
        scores: torch.Tensor,  # [B, N]
    ) -> Dict[str, float]:
        """
        Compute full entropy distribution statistics for logging.

        Returns mean, std, and percentiles of entropy distribution.
        """
        entropy, _, _ = self.compute_uncertainty(scores)
        return {
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            "entropy_min": entropy.min().item(),
            "entropy_max": entropy.max().item(),
            "entropy_p50": torch.quantile(entropy.float(), 0.5).item(),
            "entropy_p90": torch.quantile(entropy.float(), 0.9).item(),
            "entropy_p95": torch.quantile(entropy.float(), 0.95).item(),
            "replan_fraction": (entropy > self.threshold_tau).float().mean().item(),
        }

    # ── High-level API ───────────────────────────────────────

    def analyse(
        self,
        scores: torch.Tensor,  # [B, N]
    ) -> List[UncertaintyResult]:
        """Produce per-sample UncertaintyResult for logging/decision making."""
        entropy, max_prob, margin = self.compute_uncertainty(scores)
        replan = self.should_replan(scores)
        B = scores.shape[0]

        results = []
        for i in range(B):
            results.append(UncertaintyResult(
                predictive_entropy=entropy[i].item(),
                max_probability=max_prob[i].item(),
                margin=margin[i].item(),
                should_replan=replan[i].item(),
            ))
        return results
