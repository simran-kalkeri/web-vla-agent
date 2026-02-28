"""
Supervised Training — Multi-Stage Imitation Learning (Multimodal).

Stage 1: Single-step imitation
  Input: task + DOM + screenshot + empty history
  Target: ground-truth action JSON
  Loss: token-level cross-entropy

Stage 2: Multi-step imitation (teacher forcing)
  Train on full trajectories:
  state_0 → action_0, state_1 → action_1, ...
  Action history accumulated across steps

Data flow:
  Mind2WebSample → PromptBuilder.build_training_messages()
  → processor.apply_chat_template() for all samples
  → processor(text=ALL_TEXTS, images=ALL_IMAGES) — SINGLE CALL
  → {input_ids, attention_mask, pixel_values, image_grid_thw, labels}
  → model.compute_loss()

CRITICAL: The Qwen2-VL processor MUST be called ONCE for the entire batch.
Per-sample processor calls corrupt the internal pixel_values ↔ image_grid_thw
mapping and produce garbage grid values that overflow the vision encoder.
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.config import VLAConfig, load_config

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset wrapper ──────────────────────────────────────────

class Mind2WebVLADataset(Dataset):
    """
    PyTorch Dataset wrapping Mind2Web samples for VLA training.

    Each sample is a Mind2WebSample with .task, .serialized_dom,
    .action, .action_history, .screenshot, .website fields.
    Tokenization happens in the collator.
    """

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Multimodal collate function ──────────────────────────────

def multimodal_collate_fn(batch, processor, prompt_builder, max_seq_length=2048):
    """
    Collate Mind2WebSamples into a multimodal training batch.

    Strategy:
    1. Build ALL chat-template texts for the batch
    2. Collect ALL PIL screenshot images
    3. Call processor() ONCE on the full batch
    4. Find assistant content boundary via token search for label masking

    The processor handles pixel_values batching, image_grid_thw construction,
    image token expansion, and padding internally.

    Returns dict with:
      - input_ids: [batch, seq_len]
      - attention_mask: [batch, seq_len]
      - labels: [batch, seq_len]  (prompt tokens = -100)
      - pixel_values: processor-managed tensor
      - image_grid_thw: processor-managed tensor
    """
    all_full_texts = []
    all_images = []

    for sample in batch:
        # Build multimodal chat messages with target
        training_data = prompt_builder.build_training_messages(
            task=sample.task,
            serialized_dom=sample.serialized_dom,
            target_action=sample.action,
            screenshot=sample.screenshot,
            action_history=sample.action_history,
            extra_context=f"Website: {sample.website}" if sample.website else "",
        )

        messages_full = training_data["messages_with_target"]

        # Render full conversation text (system + user + assistant)
        full_text = processor.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )
        all_full_texts.append(full_text)

        # Collect screenshot image (every sample has one — fallback is a white img)
        if sample.screenshot is not None:
            all_images.append(sample.screenshot)

    # ── SINGLE processor call for the entire batch ──
    batch_inputs = processor(
        text=all_full_texts,
        images=all_images if all_images else None,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    # ── Ensure batch dimension exists for image_grid_thw ──
    # Qwen2-VL processor pixel_values shape is ALWAYS [total_patches, hidden_dim]
    # (no batch dimension) — this is correct, do NOT unsqueeze it.
    # Only image_grid_thw may collapse from [1,3] → [3] when batch_size=1.
    if "image_grid_thw" in batch_inputs:
        if batch_inputs["image_grid_thw"].dim() == 1:
            batch_inputs["image_grid_thw"] = batch_inputs["image_grid_thw"].unsqueeze(0)

    # ── Build labels with prompt masking ──
    # Strategy: find the last <|im_start|> token in each sequence
    # (marks the assistant turn), then mask everything before
    # the assistant's content with -100.
    labels = batch_inputs["input_ids"].clone()

    # Get the token ID for <|im_start|>
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    # Tokenize the assistant role header to measure its length
    # Chat template format: <|im_start|>assistant\n{content}<|im_end|>\n
    assistant_header_ids = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    header_len = len(assistant_header_ids)

    for i in range(labels.shape[0]):
        ids = batch_inputs["input_ids"][i]

        # Find all <|im_start|> positions
        positions = (ids == im_start_id).nonzero(as_tuple=True)[0]

        if len(positions) >= 3:
            # Last <|im_start|> = assistant turn start
            assistant_start = positions[-1].item()

            # Content begins after the header: <|im_start|>assistant\n
            content_start = assistant_start + header_len

            # Mask everything before assistant content (= prompt)
            labels[i, :content_start] = -100
        elif len(positions) >= 1:
            # Fallback: use last <|im_start|> + header
            assistant_start = positions[-1].item()
            content_start = assistant_start + header_len
            labels[i, :content_start] = -100
        else:
            # No <|im_start|> found — mask everything (shouldn't happen)
            print(f"WARNING: No <|im_start|> found in sample {i}, masking all tokens")
            labels[i, :] = -100

        # Mask padding tokens
        pad_mask = batch_inputs["attention_mask"][i] == 0
        labels[i, pad_mask] = -100

    # ── Assemble result ──
    result = {
        "input_ids": batch_inputs["input_ids"],
        "attention_mask": batch_inputs["attention_mask"],
        "labels": labels,
    }

    if "pixel_values" in batch_inputs:
        result["pixel_values"] = batch_inputs["pixel_values"]
    if "image_grid_thw" in batch_inputs:
        result["image_grid_thw"] = batch_inputs["image_grid_thw"]

    # First-batch diagnostics (print to stdout — logger may not be configured)
    global _COLLATE_LOGGED_FIRST
    if not _COLLATE_LOGGED_FIRST:
        _COLLATE_LOGGED_FIRST = True
        print("========== COLLATE DIAGNOSTICS (first batch) ==========")
        print(f"  Batch size: {len(batch)}")
        print(f"  input_ids shape: {result['input_ids'].shape}")
        print(f"  labels shape: {result['labels'].shape}")
        if "pixel_values" in result:
            print(f"  pixel_values shape: {result['pixel_values'].shape}")
            print(f"  pixel_values dtype: {result['pixel_values'].dtype}")
        else:
            print("  pixel_values: MISSING")
        if "image_grid_thw" in result:
            print(f"  image_grid_thw shape: {result['image_grid_thw'].shape}")
            print(f"  image_grid_thw dtype: {result['image_grid_thw'].dtype}")
            print(f"  image_grid_thw values:\n{result['image_grid_thw']}")
        else:
            print("  image_grid_thw: MISSING")
        # Label stats
        num_target = (result["labels"] != -100).sum().item()
        total = result["labels"].numel()
        print(f"  Supervised tokens: {num_target}/{total} ({100*num_target/total:.1f}%)")
        print("=======================================================")

    return result


_COLLATE_LOGGED_FIRST = False


# ── Trainer ──────────────────────────────────────────────────

class VLATrainer:
    """
    Multi-stage imitation learning trainer for VLA Web Agent.

    Parameters
    ----------
    config : VLAConfig
    device : str
    """

    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or load_config()
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def setup(self) -> None:
        """Initialize model, optimizer, and scheduler."""
        from models.vla_model import VLAModel

        logger.info("Setting up VLA model for training...")

        # Load model with QLoRA
        self.model = VLAModel(config=self.config, device=self.device)
        self.model.load()
        self.model.apply_lora()

        # Configure processor image resolution limits
        if hasattr(self.model.processor, "image_processor"):
            self.model.processor.image_processor.min_pixels = (
                self.config.model.image_min_pixels
            )
            self.model.processor.image_processor.max_pixels = (
                self.config.model.image_max_pixels
            )
            logger.info(
                f"Image processor: min_pixels={self.config.model.image_min_pixels}, "
                f"max_pixels={self.config.model.image_max_pixels}"
            )

        # Setup optimizer — only LoRA parameters
        trainable_params = [p for p in self.model.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        logger.info("Training setup complete")

    def _run_training_loop(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        stage: int,
        val_samples: Optional[list] = None,
        prompt_builder=None,
    ) -> Dict[str, Any]:
        """
        Shared training loop for Stage 1 and Stage 2.

        Returns dict with metrics history.
        """
        self.model.model.train()
        total_steps = 0
        grad_accum = self.config.training.gradient_accumulation_steps
        metrics_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Forward pass — pass all multimodal tensors
                    loss = self.model.compute_loss(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        pixel_values=batch.get("pixel_values"),
                        image_grid_thw=batch.get("image_grid_thw"),
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / grad_accum
                    loss.backward()

                    if (batch_idx + 1) % grad_accum == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.model.parameters() if p.requires_grad],
                            self.config.training.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        total_steps += 1

                    epoch_loss += loss.item() * grad_accum
                    epoch_steps += 1

                    if (batch_idx + 1) % self.config.training.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        logger.info(
                            f"  Epoch {epoch+1} Step {batch_idx+1}/{len(dataloader)} "
                            f"Loss={avg_loss:.4f}"
                        )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(
                            f"OOM at batch {batch_idx+1}, skipping. "
                            "Consider reducing max_seq_length or image_max_pixels."
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                        continue
                    raise

            epoch_time = time.time() - start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Loss={avg_epoch_loss:.4f} Time={epoch_time:.1f}s"
            )

            metrics_history.append({
                "epoch": epoch + 1,
                "stage": stage,
                "loss": avg_epoch_loss,
                "time": epoch_time,
            })

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self._save_checkpoint(f"stage{stage}_epoch{epoch+1}")

            # Validation
            if val_samples and prompt_builder:
                val_metrics = self._validate(val_samples, prompt_builder)
                metrics_history[-1].update(val_metrics)
                logger.info(f"  Val: {val_metrics}")

        return {"stage": stage, "metrics": metrics_history}

    def train_stage1(
        self,
        train_samples: list,
        val_samples: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1: Single-step imitation learning.

        Each sample is an independent (state, action) pair with screenshot.
        """
        logger.info(f"=== Stage 1: Single-Step Imitation ({self.config.training.stage1_epochs} epochs) ===")
        logger.info(f"Training samples: {len(train_samples)}")

        from models.prompt_builder import PromptBuilder
        prompt_builder = PromptBuilder()

        dataset = Mind2WebVLADataset(train_samples)

        # Create dataloader with multimodal collate
        from functools import partial
        collate = partial(
            multimodal_collate_fn,
            processor=self.model.processor,
            prompt_builder=prompt_builder,
            max_seq_length=self.config.training.max_seq_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
            drop_last=True,
        )

        return self._run_training_loop(
            dataloader=dataloader,
            num_epochs=self.config.training.stage1_epochs,
            stage=1,
            val_samples=val_samples,
            prompt_builder=prompt_builder,
        )

    def train_stage2(
        self,
        trajectories: list,
        val_samples: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Multi-step imitation with teacher forcing.

        Trains on full trajectories where each step's action history
        includes the ground-truth actions from previous steps.
        """
        logger.info(f"=== Stage 2: Multi-Step Imitation ({self.config.training.stage2_epochs} epochs) ===")
        logger.info(f"Trajectories: {len(trajectories)}")

        from models.prompt_builder import PromptBuilder
        prompt_builder = PromptBuilder()

        # Flatten trajectories into step-level samples (with action history)
        all_steps = []
        for traj in trajectories:
            all_steps.extend(traj.steps)

        logger.info(f"Total training steps: {len(all_steps)}")

        dataset = Mind2WebVLADataset(all_steps)

        from functools import partial
        collate = partial(
            multimodal_collate_fn,
            processor=self.model.processor,
            prompt_builder=prompt_builder,
            max_seq_length=self.config.training.max_seq_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
            drop_last=True,
        )

        return self._run_training_loop(
            dataloader=dataloader,
            num_epochs=self.config.training.stage2_epochs,
            stage=2,
            val_samples=val_samples,
            prompt_builder=prompt_builder,
        )

    def _validate(
        self,
        val_samples: list,
        prompt_builder,
        max_val_samples: int = 100,
    ) -> Dict[str, float]:
        """Run validation on a subset of samples."""
        from models.action_decoder import ActionDecoder

        self.model.model.eval()
        decoder = ActionDecoder()

        val_subset = val_samples[:max_val_samples]
        correct_actions = 0
        correct_elements = 0
        total = 0

        with torch.no_grad():
            for sample in val_subset:
                try:
                    # Build prompt
                    messages = prompt_builder.build_chat_messages(
                        task=sample.task,
                        serialized_dom=sample.serialized_dom,
                        action_history=sample.action_history,
                        screenshot_placeholder=(sample.screenshot is not None),
                    )

                    # Generate
                    result = self.model.generate(
                        messages=messages,
                        image=sample.screenshot,
                        temperature=0.01,
                    )

                    # Parse and compare
                    pred = decoder.parse(result["text"])
                    gt = sample.action

                    if pred:
                        if pred.get("action") == gt.get("action"):
                            correct_actions += 1
                        if pred.get("element_id") == gt.get("element_id"):
                            correct_elements += 1

                    total += 1
                except Exception:
                    total += 1
                    logger.debug("Validation sample failed", exc_info=True)

        self.model.model.train()

        return {
            "val_action_acc": correct_actions / max(total, 1),
            "val_element_acc": correct_elements / max(total, 1),
            "val_total": total,
        }

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        ckpt_dir = Path(self.config.training.checkpoint_dir) / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_lora(str(ckpt_dir))
        logger.info(f"Checkpoint saved: {ckpt_dir}")


# ── Entry point ──────────────────────────────────────────────

def main():
    """Main training entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="VLA Web Agent Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training samples (None = full dataset)")
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    set_seed(config.training.seed)

    from utils.logging import get_logger
    log = get_logger("vla.train", level=config.logging.level, log_dir=config.logging.log_dir)

    # Load data
    from data.mind2web_loader import Mind2WebLoader
    loader = Mind2WebLoader(
        dataset_name=config.data.dataset_name,
        max_dom_nodes=config.data.max_dom_nodes,
    )

    log.info("Loading training data...")
    train_samples = loader.build_training_examples(
        split="train",
        max_samples=args.max_samples,
        include_screenshot=True,
    )

    log.info(f"Loaded {len(train_samples)} training samples")

    # Initialize trainer
    trainer = VLATrainer(config=config, device=args.device)
    trainer.setup()

    # Run training stages
    if args.stage >= 1:
        result = trainer.train_stage1(train_samples)
        log.info(f"Stage 1 complete: {result}")

    if args.stage >= 2:
        log.info("Building trajectories for Stage 2...")
        trajectories = loader.build_trajectories(
            split="train",
            max_samples=args.max_samples,
            include_screenshot=True,
        )
        result = trainer.train_stage2(trajectories)
        log.info(f"Stage 2 complete: {result}")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
