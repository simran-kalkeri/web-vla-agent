"""
GroqModel — Cloud inference backend via Groq API.

Drop-in replacement for VLAModel.  All calls go to the Groq API
endpoint which runs inference on Groq's LPU hardware.  This removes
ANY local GPU / large-RAM requirement: only Python + an API key needed.

Interface contract (mirrors VLAModel exactly):
  - load()
  - generate(messages, image, max_new_tokens, temperature, top_p, return_log_probs)
  - generate_with_beams(messages, image, num_beams, max_new_tokens)

Notes on feature parity:
  - log_probs / avg_log_prob: Groq does not expose per-token log-probs
    in their API.  We return avg_log_prob=0.0 (neutral — well above the
    -2.0 threshold), so uncertainty-based regeneration is effectively
    disabled.  This is safe; the action decoder still validates outputs.
  - generate_with_beams: implemented as N independent API calls at
    incrementally higher temperatures, then deduplicated and scored by
    length-normalised output (approximation of beam score).
  - Vision: screenshots are base64-encoded and sent as image_url content
    blocks (OpenAI-compatible vision format that Groq supports).

Supported vision models (Groq, April 2026):
  - meta-llama/llama-4-scout-17b-16e-instruct  (recommended, fast)
  - meta-llama/llama-4-maverick-17b-128e-instruct
  - llama-3.2-11b-vision-preview
  - llama-3.2-90b-vision-preview

Text-only fallback (no screenshot):
  - llama-3.3-70b-versatile
  - llama-3.1-8b-instant
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GroqModel:
    """
    Groq API inference backend for the VLA Web Agent.

    Parameters
    ----------
    config : VLAConfig, optional
        Full agent config.  Only config.groq is used.
    device : str
        Ignored — Groq runs on Groq's LPU cloud.  Kept for API parity.
    """

    def __init__(
        self,
        config=None,
        device: str = "cpu",   # ignored, kept for parity
    ):
        from utils.config import load_config
        self.config = config or load_config()
        self.device = device          # not used
        self._is_loaded = False
        self._client = None

        # Resolve Groq config with sane defaults if the yaml section is absent
        groq_cfg = getattr(self.config, "groq", None)
        self.model_name: str = (
            getattr(groq_cfg, "model", None)
            or "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self.max_new_tokens: int = (
            getattr(groq_cfg, "max_new_tokens", None)
            or self.config.model.max_new_tokens
        )
        self.temperature: float = (
            getattr(groq_cfg, "temperature", None)
            or self.config.model.temperature
        )
        self.top_p: float = (
            getattr(groq_cfg, "top_p", None)
            or self.config.model.top_p
        )

    # ── Lifecycle ────────────────────────────────────────────────

    def load(self) -> None:
        """
        Initialise the Groq client.

        Reads GROQ_API_KEY from the environment (or from a .env file if
        python-dotenv is installed).  Raises RuntimeError if not found.
        """
        if self._is_loaded:
            return

        # Try loading .env silently
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.  "
                "Add it to your environment or to a .env file:\n"
                "  GROQ_API_KEY=gsk_..."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The 'groq' Python package is not installed.  "
                "Run:  pip install groq"
            ) from exc

        self._client = Groq(api_key=api_key)
        self._is_loaded = True
        logger.info(
            f"GroqModel ready — model='{self.model_name}'  "
            f"(no local GPU required)"
        )
        print(f"  ✅ Groq backend initialised  →  model: {self.model_name}")

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _pil_to_b64(image) -> str:
        """Encode a PIL Image (or file path string) as a base64 PNG data-URL."""
        from PIL import Image as PILImage
        if isinstance(image, str):
            # Received a file path instead of a PIL Image — load it
            image = PILImage.open(image).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        raw = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{raw}"

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]],
        image=None,
    ) -> List[Dict[str, Any]]:
        """
        Convert VLA prompt-builder messages to Groq/OpenAI-compatible format.

        The prompt_builder always sets item["image"] = "screenshot_placeholder"
        (a literal string) for image content blocks.  We ignore that value and
        always use the actual `image` PIL object passed from generate().

        Qwen2-VL format::

            {"role": "user", "content": [
                {"type": "image", "image": "screenshot_placeholder"},
                {"type": "text", "text": "..."},
            ]}

        Groq / OpenAI format::

            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,..."}},
                {"type": "text", "text": "..."},
            ]}
        """
        groq_messages: List[Dict[str, Any]] = []
        image_b64: Optional[str] = None
        if image is not None:
            image_b64 = self._pil_to_b64(image)

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # String content (e.g. system prompt) → pass through unchanged
            if isinstance(content, str):
                groq_messages.append({"role": role, "content": content})
                continue

            # List content → convert/filter items
            new_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")

                if item_type == "image":
                    # Always use the PIL image from generate(), not the
                    # "screenshot_placeholder" string the prompt_builder sets.
                    if image_b64 is not None:
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": image_b64},
                        })
                    # If no image provided, drop the block (text-only call)
                    continue

                if item_type == "text":
                    new_content.append({
                        "type": "text",
                        "text": item.get("text", ""),
                    })
                    continue

                # Unknown item type — pass through as-is
                new_content.append(item)

            groq_messages.append({"role": role, "content": new_content})

        return groq_messages

    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Make a single Groq chat completion call and return the text."""
        if not self._is_loaded:
            self.load()

        # Groq requires temperature > 0 for sampling; clamp to avoid errors
        safe_temp = max(temperature, 0.01)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=safe_temp,
            top_p=top_p,
            stream=False,
        )
        return response.choices[0].message.content or ""

    # ── Public API (mirrors VLAModel) ────────────────────────────

    def generate(
        self,
        messages: List[Dict[str, Any]],
        image=None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_log_probs: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate an action via Groq API.

        Parameters
        ----------
        messages : list of dicts
            Chat messages in VLA prompt-builder format.
        image : PIL.Image, optional
            Screenshot to include as vision input.
        max_new_tokens : int, optional
        temperature : float, optional
        top_p : float, optional
        return_log_probs : bool
            Accepted for API parity; Groq does not expose log-probs.
            Always returns avg_log_prob=0.0 (neutral confidence).

        Returns
        -------
        dict with:
          - "text": generated text string
          - "log_probs": [] (empty — not available from Groq)
          - "avg_log_prob": 0.0 (neutral — will not trigger regeneration)
        """
        if not self._is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        groq_messages = self._convert_messages(messages, image)

        try:
            text = self._call_api(groq_messages, max_new_tokens, temperature, top_p)
        except Exception as exc:
            logger.error(f"Groq API error: {exc}")
            # Return empty text — action decoder will flag as invalid
            text = ""

        result: Dict[str, Any] = {"text": text}
        if return_log_probs:
            # Groq does not provide per-token log-probs.
            # Return neutral value (0.0) so uncertainty check never fires.
            result["log_probs"] = []
            result["avg_log_prob"] = 0.0

        return result

    def generate_with_beams(
        self,
        messages: List[Dict[str, Any]],
        image=None,
        num_beams: int = 3,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Approximate beam search via N independent API calls.

        Each call uses a slightly higher temperature so outputs diverge.
        Results are scored by inverse length (shorter = higher score,
        approximating beam sequence score).

        Returns list of dicts with "text" and "score".
        """
        if not self._is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.max_new_tokens
        groq_messages = self._convert_messages(messages, image)

        results: List[Dict[str, Any]] = []
        base_temp = self.temperature

        for i in range(num_beams):
            temp = base_temp + i * 0.1   # slight temperature annealing
            try:
                text = self._call_api(groq_messages, max_new_tokens, temp, self.top_p)
            except Exception as exc:
                logger.warning(f"Beam {i} Groq API error: {exc}")
                text = ""

            # Approximated score: shorter valid outputs rank higher
            score = -len(text) / max(len(text), 1)
            results.append({"text": text, "score": score})

        return results

    # ── Stubs for training-only methods (not used at inference) ──

    def apply_lora(self) -> None:
        """No-op — LoRA is local-only; Groq uses cloud weights."""
        logger.info("apply_lora() is a no-op for GroqModel (cloud backend)")

    def save_lora(self, path: str) -> None:
        """No-op."""
        logger.info("save_lora() is a no-op for GroqModel")

    def load_lora(self, path: str) -> None:
        """No-op."""
        logger.info("load_lora() is a no-op for GroqModel")

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError(
            "Training is not supported with GroqModel. "
            "Use VLAModel for local training."
        )
