"""
Candidate Ranker — Semantic Similarity for Element Ranking (I2).

Ranks DOM candidates by semantic similarity to the task instruction
using a lightweight sentence-transformer model. Falls back to word
overlap if sentence-transformers is unavailable.

The ranker runs on CPU to keep the GPU free for Qwen2-VL.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CandidateRanker:
    """
    Ranks DOM candidates by semantic similarity to task instruction.
    Uses a lightweight sentence-transformer for embedding similarity.
    Falls back to word overlap if sentence-transformers unavailable.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # 22M params, 80ms per batch on CPU, good enough for ranking

    def __init__(self, device: str = "cpu"):
        # Load on CPU — keep GPU free for Qwen2-VL
        self._model = None
        self.device = device

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.MODEL_NAME, device=self.device)
                logger.info(f"CandidateRanker loaded: {self.MODEL_NAME} on {self.device}")
            except (ImportError, Exception) as e:
                logger.warning(f"CandidateRanker: sentence-transformers unavailable ({e}), using word overlap fallback")
                self._model = "unavailable"

    def rank(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 64,
    ) -> List[Dict[str, Any]]:
        """
        Return top_n candidates ranked by cosine similarity to task.
        Falls back to word overlap if model unavailable.
        """
        self._load()

        if self._model == "unavailable" or len(candidates) <= top_n:
            # Fall back to word overlap scoring
            return self._word_overlap_rank(task, candidates, top_n)

        # Build text representation of each candidate
        cand_texts = []
        for c in candidates:
            attrs = c.get("attributes", {})
            parts = [
                c.get("text", ""),
                attrs.get("placeholder", ""),
                attrs.get("aria-label", ""),
                attrs.get("title", ""),
                c.get("tag", ""),
            ]
            cand_texts.append(" ".join(p for p in parts if p).strip() or "element")

        try:
            # Encode task and all candidates in one batch
            all_texts = [task] + cand_texts
            embeddings = self._model.encode(
                all_texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self.device,
            )

            task_emb = embeddings[0]
            cand_embs = embeddings[1:]

            # Cosine similarity
            from torch.nn.functional import cosine_similarity
            scores = cosine_similarity(
                task_emb.unsqueeze(0), cand_embs
            ).cpu().tolist()

            # Attach scores and sort
            for c, score in zip(candidates, scores):
                c["semantic_score"] = score

            ranked = sorted(candidates, key=lambda x: x.get("semantic_score", 0), reverse=True)
            return ranked[:top_n]
        except Exception as e:
            logger.warning(f"CandidateRanker encoding failed: {e}, falling back to word overlap")
            return self._word_overlap_rank(task, candidates, top_n)

    def _word_overlap_rank(
        self,
        task: str,
        candidates: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """Fallback ranking using word overlap scoring."""
        _STOP_WORDS = {
            "the", "a", "an", "in", "on", "to", "of", "and", "or", "for",
            "is", "it", "at", "by", "from", "into", "then", "click",
            "type", "select", "scroll", "search", "find", "go", "open",
            "press", "enter", "submit", "field", "button", "page",
        }

        task_words = {
            w.lower()
            for w in task.split()
            if len(w) > 2 and w.lower() not in _STOP_WORDS
        }

        scored = []
        for c in candidates:
            parts = [c.get("text", "")]
            attrs = c.get("attributes", {})
            for key in ("placeholder", "aria-label", "title", "name", "value", "alt"):
                if key in attrs:
                    parts.append(attrs[key])
            el_text = " ".join(parts).lower()
            score = sum(1 for w in task_words if w in el_text)
            tag = c.get("tag", "").lower()
            if tag in ("input", "textarea") and any(
                w in task.lower() for w in ("type", "enter", "write", "input", "fill")
            ):
                score += 3
            c["semantic_score"] = score
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_n]]
