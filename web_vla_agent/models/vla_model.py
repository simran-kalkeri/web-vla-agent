"""
Core VLA Model — Qwen2-VL Multimodal Transformer.

End-to-end multimodal web agent using Qwen2-VL as backbone.
All tokens (task, DOM, action history, screenshot patches) fully cross-attend.

Architecture:
  - Qwen2-VL-2B-Instruct (or 7B) with optional QLoRA (4-bit quantization)
  - Full vision-language cross-attention (no bi-encoder)
  - Autoregressive JSON action generation
  - Token-level log probability extraction for uncertainty

Stability rules:
  - QLoRA via bitsandbytes when config.model.use_qlora is True
  - No manual image resizing (processor handles it)
  - No process_vision_info (direct processor call, matching training)
  - processor.image_processor.min_pixels/max_pixels set to match training
"""
from __future__ import annotations
from PIL import Image
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.config import VLAConfig, load_config

logger = logging.getLogger(__name__)


class VLAModel:
    """
    Multimodal VLA Web Agent backed by Qwen2-VL.

    Supports:
      - GPU inference with optional QLoRA (4-bit) or pure bfloat16
      - LoRA adapters for parameter-efficient fine-tuning
      - Autoregressive JSON action generation
      - Token-level log probability extraction
      - Beam search for uncertainty estimation

    Parameters
    ----------
    config : VLAConfig, optional
    device : str
        Device for inference ('cuda', 'cuda:0', etc.)
    """

    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or load_config()
        self.device = device

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self) -> None:
        """Load model, processor, and tokenizer. Uses QLoRA (4-bit) if configured."""
        if self._is_loaded:
            return

        model_name = self.config.model.name
        use_qlora = getattr(self.config.model, "use_qlora", False)
        quant_bits = getattr(self.config.model, "quantization_bits", 4)

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # Build model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        if use_qlora:
            from transformers import BitsAndBytesConfig

            logger.info(
                f"Loading {model_name} with QLoRA ({quant_bits}-bit quantization)..."
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(quant_bits == 4),
                load_in_8bit=(quant_bits == 8),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            logger.info(f"Loading {model_name} in pure bfloat16 (no quantization)...")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Load processor (handles both text + vision)
        # NOTE: use_fast=False is REQUIRED — the fast Qwen2VLImageProcessor
        # (new default in recent transformers) returns corrupted image_grid_thw
        # values, causing overflow in the vision encoder's rotary positional
        # embedding. The slow processor produces correct grid values.
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        self.tokenizer = self.processor.tokenizer

        # Set image processor resolution limits to MATCH TRAINING exactly.
        # Training (train_supervised.py:265-269) sets these on the processor.
        # Without this, the processor uses different defaults and produces
        # different vision token counts → distribution drift.
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.min_pixels = (
                self.config.model.image_min_pixels
            )
            self.processor.image_processor.max_pixels = (
                self.config.model.image_max_pixels
            )
            logger.info(
                f"Image processor: min_pixels={self.config.model.image_min_pixels}, "
                f"max_pixels={self.config.model.image_max_pixels}"
            )

        # NOTE: Embedding resize is deferred to apply_lora() / load_lora().
        # get_peft_model() can interfere with a resize done here,
        # so we resize AFTER those calls.

        self._is_loaded = True
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model loaded: {total_params/1e6:.0f}M params total, "
            f"{trainable/1e6:.1f}M trainable"
        )

    def apply_lora(self) -> None:
        """
        Apply LoRA adapters for fine-tuning.

        Targets:
          - Language model: q/k/v/o_proj, gate/up/down_proj
          - Vision-language merger: cross-attention projections

        This ensures vision grounding adapts during fine-tuning,
        not just text generation.
        """
        if not self._is_loaded:
            self.load()

        from peft import LoraConfig, get_peft_model

        # Prepare model for quantized training if QLoRA is active
        use_qlora = getattr(self.config.model, "use_qlora", False)
        if use_qlora:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
            logger.info("Model prepared for QLoRA k-bit training")

        # Build target modules list — include vision merger layers
        target_modules = list(self.config.model.lora_target_modules)

        # Discover cross-attention / merger modules in the model.
        # Only collect known-safe leaf names to avoid accidentally targeting
        # unrelated modules that share a generic name like 'fc1' or 'mlp'.
        _VISION_LEAF_NAMES = frozenset({"fc1", "fc2", "q_proj", "k_proj", "v_proj", "o_proj"})
        vision_targets = set()
        for name, _ in self.model.named_modules():
            # Qwen2-VL uses 'visual.merger' for vision-language bridge
            if "merger" in name:
                leaf = name.split(".")[-1]
                if leaf in _VISION_LEAF_NAMES:
                    vision_targets.add(leaf)
            # Also catch cross-attention projections
            if "cross_attn" in name:
                leaf = name.split(".")[-1]
                if leaf in _VISION_LEAF_NAMES:
                    vision_targets.add(leaf)

        # Merge with language model targets and deduplicate
        all_targets = list(set(target_modules) | vision_targets)
        logger.info(f"LoRA target modules: {sorted(all_targets)}")

        lora_config = LoraConfig(
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=all_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)

        # Resize embedding table AFTER prepare_model_for_kbit_training and
        # get_peft_model, which can reset or wrap the embedding layer.
        # Qwen2-VL config.vocab_size (151936) does NOT include vision special
        # tokens that the tokenizer can produce (up to 152064). Any token ID
        # outside the embedding table causes a CUDA assertion failure.
        actual_emb_size = self.model.get_input_embeddings().weight.shape[0]
        tokenizer_size = len(self.tokenizer)
        if tokenizer_size != actual_emb_size:
            logger.info(
                f"Resizing token embeddings: {actual_emb_size} -> {tokenizer_size}"
            )
            self.model.resize_token_embeddings(tokenizer_size)
        else:
            logger.info(f"Token embeddings already correct size: {actual_emb_size}")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"LoRA applied: {trainable/1e6:.1f}M trainable / "
            f"{total/1e6:.0f}M total ({100*trainable/total:.1f}%)"
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        image: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_log_probs: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate an action given multimodal input.

        Memory contract
        ---------------
        All GPU tensors (inputs, outputs, scores) are explicitly deleted before
        this function returns.  Call torch.cuda.empty_cache() + gc.collect()
        *after* this function — NOT inside it — so the caller controls cadence.

        Preprocessing contract
        ----------------------
        Mirrors training exactly:
          - processor(text=..., images=...) — direct call, NO process_vision_info
          - NO manual image resizing — processor.image_processor handles it
          - min_pixels/max_pixels set on processor during load() to match training

        Parameters
        ----------
        messages : list of dicts
            Chat-format messages (system + user with text + image).
        image : PIL.Image, optional
            Screenshot image.
        max_new_tokens : int, optional
        temperature : float, optional
        return_log_probs : bool
            If True, also return token-level log probabilities.

        Returns
        -------
        dict with:
          - "text": generated text
          - "log_probs": list of Python floats (if return_log_probs=True)
          - "avg_log_prob": float (if return_log_probs=True)
        """
        if not self._is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.config.model.max_new_tokens
        temperature = temperature if temperature is not None else self.config.model.temperature

        # Prepare messages — replace image placeholders with actual images
        processed_messages = self._prepare_messages(messages, image)

        # Apply chat template
        text = self.processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs — direct processor call, matching training exactly.
        # Training collator (train_supervised.py:124-131) calls:
        #   processor(text=..., images=..., padding=True, return_tensors="pt")
        # We mirror this. No process_vision_info, no manual resize.
        # The processor.image_processor.min_pixels/max_pixels (set in load())
        # control vision token count, matching training configuration.
        if image is not None:
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        # Move to GPU.  After transfer we delete the CPU-side dict so the CPU
        # copies of pixel_values are freed before the (larger) GPU forward pass.
        device = self.model.device
        gpu_inputs: Dict[str, Any] = {}
        gpu_inputs["input_ids"] = inputs["input_ids"].to(device)
        gpu_inputs["attention_mask"] = inputs["attention_mask"].to(device)
        if "pixel_values" in inputs:
            gpu_inputs["pixel_values"] = inputs["pixel_values"].to(
                device=device, dtype=torch.bfloat16
            )
        if "image_grid_thw" in inputs:
            gpu_inputs["image_grid_thw"] = inputs["image_grid_thw"].to(
                device=device, dtype=torch.int64
            )
        input_len = gpu_inputs["input_ids"].shape[-1]
        del inputs  # free CPU-side tensors immediately

        # Structured JSON decoding: ALWAYS use greedy (no sampling).
        # Sampling causes format drift (arrays, explanations, etc.).
        do_sample = False
        gen_temperature = 1.0

        # torch.inference_mode() is stricter than no_grad:
        # - disables autograd graph construction entirely
        # - prevents accidental version-counter bumps from PEFT hooks
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **gpu_inputs,
                    max_new_tokens=min(max_new_tokens, 128),
                    temperature=gen_temperature,
                    do_sample=do_sample,
                    top_p=1.0,
                    repetition_penalty=self.config.model.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=return_log_probs,
                    return_dict_in_generate=return_log_probs,
                    use_cache=True,
                    return_legacy_cache=True,
                )
        finally:
            del gpu_inputs  # always free GPU inputs, even on exception

        # --- Decode and extract results BEFORE holding the full outputs object ---

        if return_log_probs:
            generated_ids = outputs.sequences[0][input_len:].tolist()
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip()

            # Extract log-probs iteratively — process one score tensor at a time
            # so we never hold the full 256×vocab_size block in memory at once.
            log_probs: List[float] = []
            scores_tuple = outputs.scores
            del outputs  # release sequences + past_key_values NOW

            for i, score in enumerate(scores_tuple):
                if i >= len(generated_ids):
                    break
                score_clamped = torch.clamp(score[0].float(), min=-1e4, max=1e4)
                log_softmax = F.log_softmax(score_clamped, dim=-1)
                log_probs.append(log_softmax[generated_ids[i]].item())
                del score_clamped, log_softmax

            del scores_tuple

            avg_log_prob = sum(log_probs) / max(len(log_probs), 1)
            return {
                "text": generated_text,
                "log_probs": log_probs,
                "avg_log_prob": avg_log_prob,
            }
        else:
            generated_ids = outputs[0][input_len:].tolist()
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip()
            del outputs  # release KV cache + sequences

            return {"text": generated_text}

    def generate_with_beams(
        self,
        messages: List[Dict[str, Any]],
        image: Optional[Any] = None,
        num_beams: int = 3,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple beam hypotheses for uncertainty estimation.

        Returns list of dicts, each with "text" and "score".
        Uses same preprocessing contract as generate() — mirrors training.
        """
        if not self._is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.config.model.max_new_tokens
        processed_messages = self._prepare_messages(messages, image)

        text = self.processor.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True,
        )

        # Direct processor call — matching training, no process_vision_info.
        if image is not None:
            inputs = self.processor(
                text=[text], images=[image],
                padding=True, return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text], padding=True, return_tensors="pt",
            )

        # Move to GPU with correct dtypes
        device = self.model.device
        gpu_inputs: Dict[str, Any] = {}
        gpu_inputs["input_ids"] = inputs["input_ids"].to(device)
        gpu_inputs["attention_mask"] = inputs["attention_mask"].to(device)
        if "pixel_values" in inputs:
            gpu_inputs["pixel_values"] = inputs["pixel_values"].to(
                device=device, dtype=torch.bfloat16
            )
        if "image_grid_thw" in inputs:
            gpu_inputs["image_grid_thw"] = inputs["image_grid_thw"].to(
                device=device, dtype=torch.int64
            )
        input_len = gpu_inputs["input_ids"].shape[-1]
        del inputs

        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **gpu_inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    return_legacy_cache=True,
                )
        finally:
            del gpu_inputs

        results = []
        for i in range(num_beams):
            gen_ids = outputs.sequences[i][input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            score = outputs.sequences_scores[i].item() if hasattr(outputs, 'sequences_scores') else 0.0
            results.append({"text": gen_text, "score": score})

        del outputs
        return results

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute token-level cross-entropy loss for training.

        Labels may contain -100.
        input_ids MUST NOT contain negative values or IDs >= embedding size.
        """

        if not self._is_loaded:
            self.load()

        # Get embedding table size
        emb_size = self.model.get_input_embeddings().weight.shape[0]

        # Compute diagnostics
        min_id = int(input_ids.min())
        max_id = int(input_ids.max())
        neg_count = int((input_ids < 0).sum())
        oob_count = int((input_ids >= emb_size).sum())

        # Log once for debugging
        if not getattr(self, "_ids_debug_printed", False):
            logger.info("========== TOKEN DIAGNOSTICS ==========")
            logger.info(f"Embedding size: {emb_size}")
            logger.info(f"Min input_id: {min_id}")
            logger.info(f"Max input_id: {max_id}")
            logger.info(f"Negative ID count: {neg_count}")
            logger.info(f"Out-of-bounds count: {oob_count}")
            logger.info(f"ATTN MASK MIN: {int(attention_mask.min())}")
            logger.info(f"ATTN MASK MAX: {int(attention_mask.max())}")
            logger.info(f"ATTN MASK DTYPE: {attention_mask.dtype}")
            logger.info(f"pixel_values present: {pixel_values is not None}")
            logger.info(f"image_grid_thw present: {image_grid_thw is not None}")
            if pixel_values is not None:
                logger.info(f"pixel_values shape: {pixel_values.shape}")
            if image_grid_thw is not None:
                logger.info(f"image_grid_thw shape: {image_grid_thw.shape}")
            logger.info("=======================================")
            self._ids_debug_printed = True

        # Hard assertions (no silent corruption)
        assert neg_count == 0, (
            f"Negative token IDs detected in input_ids "
            f"(count={neg_count}, min_id={min_id})"
        )

        assert oob_count == 0, (
            f"Token IDs exceed embedding size "
            f"(count={oob_count}, max_id={max_id}, emb_size={emb_size})"
        )

        # Move tensors to correct device with correct dtypes
        device = self.model.device
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

        if pixel_values is not None:
            # Cast to model compute dtype — processor returns float32,
            # but vision encoder runs in bfloat16 under QLoRA
            kwargs["pixel_values"] = pixel_values.to(
                device=device, dtype=torch.bfloat16
            )

        if image_grid_thw is not None:
            # MUST be int64 (torch.LongTensor) — HF docs specify this.
            # rot_pos_emb uses torch.arange(h) where h comes from grid_thw.
            # int32 causes silent overflow on large grids.
            kwargs["image_grid_thw"] = image_grid_thw.to(
                device=device, dtype=torch.int64
            )

        # Forward pass
        outputs = self.model(**kwargs)

        return outputs.loss

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        image: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Replace image placeholders in messages with actual PIL images."""
        import copy
        processed = copy.deepcopy(messages)

        if image is None:
            # Remove image entries
            for msg in processed:
                if isinstance(msg.get("content"), list):
                    msg["content"] = [
                        c for c in msg["content"]
                        if not (isinstance(c, dict) and c.get("type") == "image")
                    ]
            return processed

        # Replace placeholder with actual image
        for msg in processed:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        item["image"] = image
        return processed

    def save_lora(self, path: str) -> None:
        """Save LoRA adapter weights."""
        if self.model is not None:
            self.model.save_pretrained(path)
            logger.info(f"LoRA adapters saved to {path}")

    def load_lora(self, path: str) -> None:
        """Load LoRA adapter weights."""
        if not self._is_loaded:
            self.load()

        # Resize embeddings to match what was done during training (apply_lora).
        # The base model vocab size (151936) differs from the tokenizer size used
        # at training time, so we must resize before loading the checkpoint or
        # PEFT will raise a size-mismatch RuntimeError.
        actual_emb_size = self.model.get_input_embeddings().weight.shape[0]
        tokenizer_size = len(self.tokenizer)
        if tokenizer_size != actual_emb_size:
            logger.info(
                f"load_lora: resizing token embeddings {actual_emb_size} -> {tokenizer_size}"
            )
            self.model.resize_token_embeddings(tokenizer_size)

        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"LoRA adapters loaded from {path}")
