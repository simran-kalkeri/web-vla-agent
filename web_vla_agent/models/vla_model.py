"""
Core VLA Model — Qwen2-VL Multimodal Transformer.

End-to-end multimodal web agent using Qwen2-VL as backbone.
All tokens (task, DOM, action history, screenshot patches) fully cross-attend.

Architecture:
  - Qwen2-VL-2B-Instruct (or 7B) with QLoRA (4-bit + LoRA adapters)
  - Full vision-language cross-attention (no bi-encoder)
  - Autoregressive JSON action generation
  - Token-level log probability extraction for uncertainty

NOT:
  - A bi-encoder (MiniLM + CLIP)
  - A classification head
  - CPU-optimized
  - Candidate-pool restricted
"""
from __future__ import annotations

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
      - GPU inference with QLoRA (4-bit quantization + LoRA adapters)
      - Autoregressive JSON action generation
      - Token-level log probability extraction
      - Beam search for uncertainty estimation

    Parameters
    ----------
    config : VLAConfig, optional
    device : str
        Device for inference ('cuda', 'cuda:0', etc.)
    load_in_4bit : bool
        Whether to use 4-bit quantization (QLoRA).
    """

    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        device: str = "cuda",
        load_in_4bit: Optional[bool] = None,
    ):
        self.config = config or load_config()
        self.device = device
        self.load_in_4bit = load_in_4bit if load_in_4bit is not None else self.config.model.use_qlora

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self) -> None:
        """Load model, processor, and tokenizer."""
        if self._is_loaded:
            return

        model_name = self.config.model.name
        logger.info(f"Loading {model_name} (4-bit={self.load_in_4bit})...")

        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        # Quantization config
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load model
        # NOTE: attn_implementation="eager" is REQUIRED for training stability.
        # Qwen2-VL + PEFT + gradient checkpointing triggers a CUDA
        # gather index-out-of-bounds assertion when using SDPA attention.
        # Eager attention avoids this. Flash-Attention-2 is also safe if installed.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.load_in_4bit else self.device,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        # Load processor (handles both text + vision)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        # NOTE: Embedding resize is deferred to apply_lora().
        # prepare_model_for_kbit_training() and get_peft_model() can
        # interfere with a resize done here, so we resize AFTER those calls.

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

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.load_in_4bit:
            # use_reentrant=False is required to avoid a hard error in PyTorch >=2.9
            # and suppresses the "use_reentrant parameter should be passed explicitly"
            # warning already visible in training logs.
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

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
          - "log_probs": list of floats (if return_log_probs=True)
          - "avg_log_prob": float (if return_log_probs=True)
        """
        if not self._is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.config.model.max_new_tokens
        temperature = temperature or self.config.model.temperature

        # Prepare messages — replace image placeholders with actual images
        processed_messages = self._prepare_messages(messages, image)

        # Apply chat template
        text = self.processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs (text + image)
        if image is not None:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(processed_messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.01,
                top_p=self.config.model.top_p,
                repetition_penalty=self.config.model.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                output_scores=return_log_probs,
                return_dict_in_generate=return_log_probs,
            )

        if return_log_probs:
            generated_ids = outputs.sequences[0][input_len:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip()

            # Compute token log probabilities
            scores = outputs.scores  # tuple of [1, vocab_size] per step
            log_probs = []
            for i, score in enumerate(scores):
                if i >= len(generated_ids):
                    break
                log_softmax = F.log_softmax(score[0], dim=-1)
                token_log_prob = log_softmax[generated_ids[i]].item()
                log_probs.append(token_log_prob)

            avg_log_prob = sum(log_probs) / max(len(log_probs), 1)

            return {
                "text": generated_text,
                "log_probs": log_probs,
                "avg_log_prob": avg_log_prob,
            }
        else:
            generated_ids = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip()
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
        """
        if not self._is_loaded:
            self.load()

        max_new_tokens = max_new_tokens or self.config.model.max_new_tokens
        processed_messages = self._prepare_messages(messages, image)

        text = self.processor.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True,
        )

        if image is not None:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(processed_messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text], padding=True, return_tensors="pt",
            )

        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        results = []
        for i in range(num_beams):
            gen_ids = outputs.sequences[i][input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            score = outputs.sequences_scores[i].item() if hasattr(outputs, 'sequences_scores') else 0.0
            results.append({"text": gen_text, "score": score})

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

        # Move tensors to correct device
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids.to(self.model.device),
            "attention_mask": attention_mask.to(self.model.device),
            "labels": labels.to(self.model.device),
        }

        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(
                device=self.model.device, dtype=torch.bfloat16
            )

        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw.to(self.model.device)

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
                        if "image" in item and item.get("image") == "screenshot_placeholder":
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
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"LoRA adapters loaded from {path}")
