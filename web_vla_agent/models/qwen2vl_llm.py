"""
LLM backend for the TaskPlanner — Qwen2.5-3B-Instruct (text-only).

CPU-optimized: no vision encoder overhead, ~30-60s per inference on CPU.
Uses standard AutoModelForCausalLM + AutoTokenizer.

With 512GB RAM and bfloat16, the model uses ~3.5GB memory.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class PlannerLLM:
    """
    Wraps a text-only causal LM for structured planning output.

    Default model: Qwen/Qwen2.5-3B-Instruct — excellent at
    JSON output and instruction following on CPU.

    Usage:
        llm = PlannerLLM(device="cpu")
        response = llm("Decompose: Book a flight from NYC to LA")

    Callable — pass directly as ``llm_fn`` to :class:`TaskPlanner`.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cpu",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model_name = model_name
        self.torch_dtype = torch_dtype or torch.bfloat16

        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        logger.info(f"Loading {self.model_name} on {self.device} "
                     f"(dtype={self.torch_dtype})...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"Model loaded: {total_params / 1e6:.0f}M params, "
                     f"~{total_params * 2 / 1e9:.1f}GB (bf16)")

    def __call__(self, prompt: str) -> str:
        """
        Generate a response for a planning prompt.

        Parameters
        ----------
        prompt : str
            The planning prompt (decomposition or replanning).

        Returns
        -------
        str
            The model's generated text response.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded.")

        # Build chat messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert web navigation planner. "
                    "You output ONLY valid JSON arrays with no markdown "
                    "formatting, no code fences, and no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)

        logger.debug(f"LLM response ({len(response)} chars): {response[:200]}")
        return response.strip()


def create_planner_llm(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "cpu",
) -> PlannerLLM:
    """Factory to create the planner LLM callable."""
    return PlannerLLM(model_name=model_name, device=device)
