"""
Supervised Training — Multi-Stage Imitation Learning.

Stage 1: Single-step imitation
  Input: task + DOM + screenshot + empty history
  Target: ground-truth action JSON
  Loss: token-level cross-entropy

Stage 2: Multi-step imitation (teacher forcing)
  Train on full trajectories:
  state_0 → action_0, state_1 → action_1, ...
  Action history accumulated across steps

Stage 3 (optional): RL fine-tuning scaffold

NOT:
  - Classification loss
  - InfoNCE contrastive loss
  - Focal loss on action classes
  - GCN/grounder training
"""
from __future__ import annotations

import json
import logging
import os
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

    Each sample is a dict with pre-built prompt + target text.
    Tokenization happens in the collator.
    """

    def __init__(
        self,
        samples: list,
        prompt_builder=None,
    ):
        self.samples = samples
        self.prompt_builder = prompt_builder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Simple collate function ──────────────────────────────────

def collate_fn(batch, tokenizer, prompt_builder, max_seq_length=4096):
    """
    Collate Mind2WebSamples into a training batch.

    Builds prompt + target, tokenizes, constructs labels with -100
    masking on prompt tokens.
    """
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []

    for sample in batch:
        # Build training prompt + target
        training = prompt_builder.build_training_prompt(
            task=sample.task,
            serialized_dom=sample.serialized_dom,
            target_action=sample.action,
            action_history=sample.action_history,
            extra_context=f"Website: {sample.website}" if sample.website else "",
        )

        prompt_text = training["prompt"]
        target_text = "\n" + training["target"]

        # Tokenize prompt and target separately
        prompt_tokens = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length - 100,
            add_special_tokens=True,
        )
        target_tokens = tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        prompt_ids = prompt_tokens["input_ids"][0]
        target_ids = target_tokens["input_ids"][0]

        # Concatenate
        input_ids = torch.cat([prompt_ids, target_ids])[:max_seq_length]

        # Labels: -100 for prompt, actual token ids for target
        labels = torch.cat([
            torch.full_like(prompt_ids, -100),
            target_ids,
        ])[:max_seq_length]

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_attention_mask.append(torch.ones_like(input_ids))

    # Pad to max length in batch
    max_len = max(ids.shape[0] for ids in batch_input_ids)
    pad_token_id = tokenizer.pad_token_id or 0

    padded_ids = []
    padded_labels = []
    padded_masks = []

    for ids, lbls, mask in zip(batch_input_ids, batch_labels, batch_attention_mask):
        pad_len = max_len - ids.shape[0]
        padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)]))
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=lbls.dtype)]))
        padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))

    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_masks),
        "labels": torch.stack(padded_labels),
    }


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

        # Setup optimizer
        trainable_params = [p for p in self.model.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        logger.info("Training setup complete")

    def train_stage1(
        self,
        train_samples: list,
        val_samples: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1: Single-step imitation learning.

        Each sample is an independent (state, action) pair.
        """
        logger.info(f"=== Stage 1: Single-Step Imitation ({self.config.training.stage1_epochs} epochs) ===")
        logger.info(f"Training samples: {len(train_samples)}")

        from models.prompt_builder import PromptBuilder
        prompt_builder = PromptBuilder()

        dataset = Mind2WebVLADataset(train_samples, prompt_builder)

        # Create dataloader with custom collate
        from functools import partial
        collate = partial(
            collate_fn,
            tokenizer=self.model.tokenizer,
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

        # Training loop
        self.model.model.train()
        total_steps = 0
        grad_accum = self.config.training.gradient_accumulation_steps
        metrics_history = []

        for epoch in range(self.config.training.stage1_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(dataloader):
                # Forward pass
                loss = self.model.compute_loss(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
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

            epoch_time = time.time() - start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.stage1_epochs} "
                f"Loss={avg_epoch_loss:.4f} Time={epoch_time:.1f}s"
            )

            metrics_history.append({
                "epoch": epoch + 1,
                "stage": 1,
                "loss": avg_epoch_loss,
                "time": epoch_time,
            })

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self._save_checkpoint(f"stage1_epoch{epoch+1}")

            # Validation
            if val_samples:
                val_metrics = self._validate(val_samples, prompt_builder)
                metrics_history[-1].update(val_metrics)
                logger.info(f"  Val: {val_metrics}")

        return {"stage": 1, "metrics": metrics_history}

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

        # Same training loop as Stage 1, but samples have action history
        dataset = Mind2WebVLADataset(all_steps, prompt_builder)

        from functools import partial
        collate = partial(
            collate_fn,
            tokenizer=self.model.tokenizer,
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

        self.model.model.train()
        total_steps = 0
        grad_accum = self.config.training.gradient_accumulation_steps
        metrics_history = []

        for epoch in range(self.config.training.stage2_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(dataloader):
                loss = self.model.compute_loss(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = loss / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum == 0:
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

            epoch_time = time.time() - start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.stage2_epochs} "
                f"Loss={avg_epoch_loss:.4f} Time={epoch_time:.1f}s"
            )

            metrics_history.append({
                "epoch": epoch + 1,
                "stage": 2,
                "loss": avg_epoch_loss,
                "time": epoch_time,
            })

            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self._save_checkpoint(f"stage2_epoch{epoch+1}")

            if val_samples:
                val_metrics = self._validate(val_samples, prompt_builder)
                metrics_history[-1].update(val_metrics)
                logger.info(f"  Val: {val_metrics}")

        return {"stage": 2, "metrics": metrics_history}

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
        total_loss = 0.0

        with torch.no_grad():
            for sample in val_subset:
                try:
                    # Build prompt
                    messages = prompt_builder.build_chat_messages(
                        task=sample.task,
                        serialized_dom=sample.serialized_dom,
                        action_history=sample.action_history,
                        screenshot_placeholder=False,
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
