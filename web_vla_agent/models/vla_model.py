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
        # device_map strategy:
        # - 4-bit (QLoRA): pin to a single GPU.  PEFT's adapter injection
        #   (_move_adapter_to_device_of_base_layer) triggers a CUDA illegal
        #   memory access when the base model is sharded across GPUs with
        #   bitsandbytes 4-bit.  A 4-bit model is small enough (~4 GB for 7B)
        #   that a single 47 GB GPU is more than sufficient.
        # - bfloat16: use device_map="auto" to shard across all visible GPUs.
        if self.load_in_4bit:
            device_map = {"":0}  # single GPU — required for PEFT + 4-bit
        else:
            device_map = "auto"  # multi-GPU shard for bfloat16

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="sdpa",
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
            # Gradient checkpointing saves ~40% VRAM on multimodal forward passes.
            # use_reentrant=False is REQUIRED for PEFT + bitsandbytes compatibility
            # and avoids PyTorch >=2.4 deprecation warnings.
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

        Memory contract
        ---------------
        All GPU tensors (inputs, outputs, scores) are explicitly deleted before
        this function returns.  Call torch.cuda.empty_cache() + gc.collect()
        *after* this function — NOT inside it — so the caller controls cadence.

        Parameters
        ----------
        messages : list of dicts
            Chat-format messages (system + user with text + image).
        image : PIL.Image, optional
            Screenshot image.  Capped to config.model.image_max_pixels before
            being sent to the vision encoder.
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

        # Process inputs (text + image).
        # IMPORTANT: pass min_pixels / max_pixels so the vision encoder never
        # receives more patches than we budget for.  Without this, a raw
        # 1280×720 screenshot produces ~1024 vision tokens; with the cap it
        # stays ≤450.  Each extra vision token adds ~29 KB to the KV cache
        # across 28 layers.
        min_px = getattr(self.config.model, "image_min_pixels", 43904)
        max_px = getattr(self.config.model, "image_max_pixels", 401408)

        if image is not None:
            from qwen_vl_utils import process_vision_info
                # Hard resize to control vision token count
            max_dim = 1024  # or 768 for even more safety
            if max(image.size) > max_dim:
                scale = max_dim / max(image.size)
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size, Image.BICUBIC)
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

        # Greedy vs. sampling.
        # temperature=0 → greedy → no probability tensor → no NaN risk.
        do_sample = temperature > 0.01
        gen_temperature = max(temperature, 0.01) if do_sample else 1.0

        # torch.inference_mode() is stricter than no_grad:
        # - disables autograd graph construction entirely
        # - prevents accidental version-counter bumps from PEFT hooks
        # - allows more aggressive kernel fusion → ~5–15% less activation VRAM
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **gpu_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=gen_temperature,
                    do_sample=do_sample,
                    top_p=self.config.model.top_p if do_sample else 1.0,
                    repetition_penalty=self.config.model.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    output_scores=return_log_probs,
                    return_dict_in_generate=return_log_probs,
                    # return_legacy_cache=True forces the KV cache to be a plain
                    # tuple of tensors instead of a DynamicCache object.  The
                    # DynamicCache holds Python-level references that CPython's
                    # reference-counter cannot always collect promptly (especially
                    # with PEFT reference cycles).  A plain tuple is freed as
                    # soon as outputs is deleted.
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
            # Each tensor is consumed (.item()) and immediately released.
            log_probs: List[float] = []
            scores_tuple = outputs.scores  # local ref to the tuple only
            del outputs  # release sequences + past_key_values NOW

            for i, score in enumerate(scores_tuple):
                if i >= len(generated_ids):
                    break
                # Guard against NaN/inf from bfloat16 instability or
                # mismatched base weights (should not happen in 4-bit mode,
                # but safe to clamp defensively).
                score_clamped = torch.clamp(score[0].float(), min=-1e4, max=1e4)
                log_softmax = F.log_softmax(score_clamped, dim=-1)
                log_probs.append(log_softmax[generated_ids[i]].item())
                del score_clamped, log_softmax  # release each step immediately

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

        # Selective device transfer — preserve dtypes
        device = self.model.device
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                device=device, dtype=torch.bfloat16
            )
        if "image_grid_thw" in inputs:
            inputs["image_grid_thw"] = inputs["image_grid_thw"].to(
                device=device, dtype=torch.int64
            )
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
