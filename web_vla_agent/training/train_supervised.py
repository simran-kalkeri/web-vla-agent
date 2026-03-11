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

S1 FIX: Added cosine LR scheduler with linear warmup.
I6 FIX: Image resolution limits configurable via config.
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


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory-based multi-step training (C2 FIX).

    Each item is a single trajectory (Mind2WebTrajectory), yielding
    all steps in order. The collate function handles interleaving.
    """

    def __init__(self, trajectories: list):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


# ── Multimodal collate function ──────────────────────────────

_PRINTED_SAMPLE = False

def multimodal_collate_fn(batch, processor, prompt_builder, max_seq_length=2048):
    """
    Collate Mind2WebSamples into a multimodal training batch.

    Strategy:
    1. Build ALL chat-template texts for the batch
    2. Collect ALL PIL screenshot images
    3. Call processor() ONCE on the full batch
    4. Find assistant content boundary via token search for label masking

    S4 FIX: Handles samples with screenshot=None by skipping them
    from the images list. The processor handles text-only samples fine.

    I6: Added check for excessive vision tokens from high-res images.

    Returns dict with:
      - input_ids: [batch, seq_len]
      - attention_mask: [batch, seq_len]
      - labels: [batch, seq_len]  (prompt tokens = -100)
      - pixel_values: processor-managed tensor
      - image_grid_thw: processor-managed tensor
    """
    global _PRINTED_SAMPLE

    # ── One-shot diagnostic: verify candidate-label alignment ──
    if not _PRINTED_SAMPLE and len(batch) > 0:
        _PRINTED_SAMPLE = True
        s = batch[0]
        print("\n========== TRAINING SAMPLE DEBUG ==========")
        print(f"TASK: {s.task}")
        print(f"TARGET ACTION: {s.action}")
        print(f"TARGET INDEX: {s.target_candidate_index}")
        print(f"\nCANDIDATES ({len(s.candidates)} total):")
        for c in s.candidates:
            idx = c.get("candidate_index", "?")
            tag = c.get("tag", "")
            text = c.get("text", "")[:60]
            marker = "  <<< TARGET >>>" if idx == s.target_candidate_index else ""
            print(f"  [{idx}] <{tag}> {text}{marker}")
        if hasattr(s, 'action_history') and s.action_history:
            print(f"\nACTION HISTORY ({len(s.action_history)} steps):")
            for h in s.action_history[-3:]:
                print(f"  {h}")
        print("==========================================\n")

    all_full_texts = []
    all_images = []

    for sample in batch:
        # Build multimodal chat messages with target (candidate-based)
        training_data = prompt_builder.build_training_messages(
            task=sample.task,
            candidates=sample.candidates,
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

        # S4 FIX: Only add screenshot if it's not None
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
    if "image_grid_thw" in batch_inputs:
        if batch_inputs["image_grid_thw"].dim() == 1:
            batch_inputs["image_grid_thw"] = batch_inputs["image_grid_thw"].unsqueeze(0)

    # I6: Check for excessive vision tokens
    if "image_grid_thw" in batch_inputs:
        total_vision_tokens = batch_inputs["image_grid_thw"].prod(dim=-1).sum().item()
        if total_vision_tokens > 5000:
            logger.warning(
                f"High vision token count: {total_vision_tokens}. "
                f"Consider lowering image_max_pixels in config."
            )

    # ── Build labels with prompt masking ──
    # Strategy: find the assistant turn in the tokenized output and mask
    # everything BEFORE the assistant's JSON content with -100.
    #
    # Qwen2-VL chat template format:
    #   <|im_start|>system\n...<|im_end|>\n
    #   <|im_start|>user\n...<|im_end|>\n
    #   <|im_start|>assistant\n{"action":...}<|im_end|>\n
    #
    # We need to supervise ONLY the JSON content after "assistant\n".
    labels = batch_inputs["input_ids"].clone()

    # Get special token IDs
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Tokenize just "assistant\n" to find its token IDs for boundary search
    # (Don't include <|im_start|> here — it's a special token that .encode()
    # may not reproduce correctly)
    assistant_role_ids = processor.tokenizer.encode(
        "assistant\n", add_special_tokens=False
    )

    for i in range(labels.shape[0]):
        ids = batch_inputs["input_ids"][i]
        ids_list = ids.tolist()

        # Find all <|im_start|> positions
        positions = (ids == im_start_id).nonzero(as_tuple=True)[0].tolist()

        if len(positions) >= 3:
            # Third <|im_start|> = assistant turn (system=0, user=1, assistant=2)
            assistant_im_start = positions[2]
        elif len(positions) >= 1:
            # Fallback: use the last <|im_start|>
            assistant_im_start = positions[-1]
        else:
            # No <|im_start|> found at all — mask everything
            print(f"ERROR sample {i}: no <|im_start|> tokens found. "
                  f"Token IDs sample: {ids_list[:20]}")
            labels[i, :] = -100
            pad_mask = batch_inputs["attention_mask"][i] == 0
            labels[i, pad_mask] = -100
            continue

        # Now find the exact content start: scan forward from assistant_im_start
        # to find "assistant\n" tokens, then content starts right after.
        content_start = None

        # Method 1: Search for the assistant_role_ids sequence after <|im_start|>
        search_start = assistant_im_start + 1  # skip <|im_start|> itself
        search_end = min(search_start + 10, len(ids_list))  # look within 10 tokens
        for j in range(search_start, search_end):
            # Check if assistant_role_ids matches starting at position j
            if ids_list[j:j + len(assistant_role_ids)] == assistant_role_ids:
                content_start = j + len(assistant_role_ids)
                break

        # Method 2: If pattern match failed, find the newline token after <|im_start|>
        if content_start is None:
            newline_ids = processor.tokenizer.encode("\n", add_special_tokens=False)
            if newline_ids:
                newline_id = newline_ids[0]
                for j in range(search_start, search_end):
                    if ids_list[j] == newline_id:
                        content_start = j + 1
                        break

        # Method 3: Last resort — use a fixed offset of 3 tokens
        # (<|im_start|> + "assistant" + "\n")
        if content_start is None:
            content_start = assistant_im_start + 3
            print(f"WARNING sample {i}: assistant boundary not found precisely, "
                  f"using offset=3 from position {assistant_im_start}")

        # Mask everything before assistant content
        labels[i, :content_start] = -100

        # Also mask everything AFTER <|im_end|> in the assistant turn
        # (the response should end with <|im_end|>\n, don't supervise those)
        end_positions = (ids[content_start:] == im_end_id).nonzero(as_tuple=True)[0]
        if len(end_positions) > 0:
            # First <|im_end|> after content_start = end of assistant response
            assistant_end = content_start + end_positions[0].item()
            labels[i, assistant_end:] = -100

        # Always mask padding
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

    # First-batch diagnostics — verify label masking is correct
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

        # ── Label masking verification ──
        for i in range(min(len(batch), 2)):
            supervised_mask = result["labels"][i] != -100
            supervised_count = supervised_mask.sum().item()
            total_count = result["labels"].shape[1]
            pct = 100 * supervised_count / total_count

            print(f"\n  === LABEL DEBUG sample {i} ===")
            print(f"  Supervised: {supervised_count}/{total_count} ({pct:.1f}%)")

            # Decode the supervised tokens to verify
            supervised_ids = result["labels"][i][supervised_mask]
            supervised_text = processor.tokenizer.decode(
                supervised_ids.tolist(), skip_special_tokens=False
            )
            print(f"  Supervised text: '{supervised_text[:300]}'")

            # Show <|im_start|> positions for debugging
            im_positions = (result["input_ids"][i] == im_start_id).nonzero(as_tuple=True)[0].tolist()
            print(f"  <|im_start|> positions: {im_positions}")
            print(f"  Expected 3 positions (system/user/assistant), got {len(im_positions)}")

            if pct > 10:
                print(f"  ⚠️  WARNING: {pct:.1f}% supervised — label masking likely broken!")
            elif pct < 0.1:
                print(f"  ⚠️  WARNING: {pct:.1f}% supervised — everything may be masked!")
            else:
                print(f"  ✅ Label masking looks correct ({pct:.1f}%)")

        print("=======================================================")

    return result


_COLLATE_LOGGED_FIRST = False


def trajectory_collate_fn(batch, processor, prompt_builder, max_seq_length=2048):
    """
    Collate trajectories into interleaved step-level batches (C2 FIX).

    Each item in batch is a Mind2WebTrajectory. Steps are interleaved
    round-robin across trajectories for balanced gradient signal:
      A0, B0, A1, B1, A2, B2, ...  (not A0, A1, A2, B0, B1, B2)
    """
    # Interleave steps across trajectories
    max_len = max((len(traj.steps) for traj in batch), default=0)
    all_steps = []
    for step_pos in range(max_len):
        for traj in batch:
            if step_pos < len(traj.steps):
                all_steps.append(traj.steps[step_pos])

    # Delegate to the standard multimodal collate
    return multimodal_collate_fn(
        all_steps,
        processor=processor,
        prompt_builder=prompt_builder,
        max_seq_length=max_seq_length,
    )


def validate_trajectories(trajectories: list) -> Dict[str, Any]:
    """
    Pre-training diagnostic for trajectory data quality.

    Reports statistics about trajectory lengths, action distributions,
    and data integrity issues.
    """
    if not trajectories:
        return {"error": "No trajectories provided"}

    step_counts = [len(t.steps) for t in trajectories]
    action_counts = {}
    samples_with_history = 0
    samples_with_screenshot = 0
    total_steps = sum(step_counts)

    for traj in trajectories:
        for step in traj.steps:
            action_type = step.action.get("action", "UNKNOWN")
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            if step.action_history:
                samples_with_history += 1
            if step.screenshot is not None:
                samples_with_screenshot += 1

    stats = {
        "num_trajectories": len(trajectories),
        "total_steps": total_steps,
        "avg_steps_per_trajectory": total_steps / len(trajectories),
        "min_steps": min(step_counts),
        "max_steps": max(step_counts),
        "action_distribution": action_counts,
        "pct_with_history": samples_with_history / max(total_steps, 1),
        "pct_with_screenshot": samples_with_screenshot / max(total_steps, 1),
    }

    print("\n========== TRAJECTORY DIAGNOSTICS ==========")
    print(f"  Trajectories: {stats['num_trajectories']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Avg steps/traj: {stats['avg_steps_per_trajectory']:.1f}")
    print(f"  Step range: [{stats['min_steps']}, {stats['max_steps']}]")
    print(f"  With history: {stats['pct_with_history']:.1%}")
    print(f"  With screenshot: {stats['pct_with_screenshot']:.1%}")
    print(f"  Action distribution: {stats['action_distribution']}")
    print("============================================\n")

    return stats


# ── Trainer ──────────────────────────────────────────────────

class VLATrainer:
    """
    Multi-stage imitation learning trainer for VLA Web Agent.

    S1 FIX: Includes cosine LR scheduler with linear warmup.

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
        self.scheduler = None       # S1: LR scheduler

    def setup(self, skip_lora: bool = False) -> None:
        """Initialize model, optimizer, and scheduler.

        Parameters
        ----------
        skip_lora : bool
            If True, skip apply_lora(). Use when loading a checkpoint
            via load_lora() — that method creates its own PeftModel
            wrapper. Calling both would double-wrap and OOM.
        """
        from models.vla_model import VLAModel

        logger.info("Setting up VLA model for training...")

        # Load model with QLoRA
        self.model = VLAModel(config=self.config, device=self.device)
        self.model.load()
        if not skip_lora:
            self.model.apply_lora()
        else:
            logger.info("Skipping apply_lora() — checkpoint will provide LoRA weights")

        # I6: Configure processor image resolution limits
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

        # S1 FIX: Scheduler is created per-stage in _setup_scheduler()
        # because it needs total_steps which depends on the dataloader.
        self.scheduler = None

        logger.info("Training setup complete")

    def _setup_scheduler(self, num_training_steps: int) -> None:
        """
        S1 FIX: Create cosine LR scheduler with linear warmup.

        Uses transformers.get_cosine_schedule_with_warmup for smooth
        learning rate annealing that prevents catastrophic forgetting
        at the end of training.
        """
        from transformers import get_cosine_schedule_with_warmup

        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(
            f"LR Scheduler: cosine with {num_warmup_steps} warmup steps "
            f"out of {num_training_steps} total"
        )

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

        S1 FIX: Calls scheduler.step() after each optimizer step.

        Returns dict with metrics history.
        """
        self.model.model.train()
        total_steps = 0
        grad_accum = self.config.training.gradient_accumulation_steps
        metrics_history = []
        num_batches = len(dataloader)

        # S1 FIX: Setup scheduler with correct total steps
        total_training_steps = (num_batches // grad_accum) * num_epochs
        if total_training_steps > 0:
            self._setup_scheduler(total_training_steps)

        current_lr = self.config.training.learning_rate

        print(f"\n{'='*60}")
        print(f"  Stage {stage} Training")
        print(f"  Epochs: {num_epochs} | Batches/epoch: {num_batches}")
        print(f"  Batch size: {self.config.training.batch_size} | Grad accum: {grad_accum}")
        print(f"  Effective batch size: {self.config.training.batch_size * grad_accum}")
        print(f"  LR: {self.config.training.learning_rate}")
        if self.scheduler:
            print(f"  Scheduler: cosine warmup ({int(total_training_steps * self.config.training.warmup_ratio)} warmup)")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            nan_count = 0       # consecutive NaN batches
            nan_total = 0       # total NaN batches this epoch
            start_time = time.time()

            self.optimizer.zero_grad()

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

                    # ── NaN detection ────────────────────────────
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_count += 1
                        nan_total += 1
                        if nan_count == 1:
                            print(
                                f"  ⚠️  NaN/Inf loss at batch {batch_idx+1}, "
                                f"skipping (won't backward through NaN)"
                            )
                        if nan_count >= 10:
                            print(
                                f"  ❌ {nan_count} consecutive NaN batches — "
                                f"model weights likely corrupted. "
                                f"Try lowering LR or enabling fp32."
                            )
                            break
                        self.optimizer.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    nan_count = 0  # reset consecutive counter

                    # Scale loss for gradient accumulation
                    scaled_loss = loss / grad_accum
                    scaled_loss.backward()

                    epoch_loss += loss.item()
                    epoch_steps += 1

                    if (batch_idx + 1) % grad_accum == 0:
                        # Gradient clipping + optimizer step
                        trainable = [p for p in self.model.model.parameters() if p.requires_grad]
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable,
                            self.config.training.max_grad_norm,
                        )

                        # Skip step if gradients themselves are NaN
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(
                                f"  ⚠️  NaN grad_norm at step {total_steps+1}, "
                                f"zeroing grads (batch {batch_idx+1})"
                            )
                            self.optimizer.zero_grad()
                            continue

                        self.optimizer.step()

                        # S1 FIX: Step the LR scheduler
                        if self.scheduler:
                            self.scheduler.step()
                            current_lr = self.scheduler.get_last_lr()[0]

                        self.optimizer.zero_grad()
                        total_steps += 1

                        avg_loss = epoch_loss / epoch_steps
                        print(
                            f"  ⚡ Step {total_steps} | "
                            f"Epoch {epoch+1}/{num_epochs} | "
                            f"Batch {batch_idx+1}/{num_batches} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"GradNorm: {grad_norm:.2f} | "
                            f"LR: {current_lr:.2e}"
                        )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(
                            f"  ⚠️  OOM at batch {batch_idx+1}, skipping. "
                            "Reduce max_seq_length or image_max_pixels."
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                        continue
                    raise

            if nan_total > 0:
                print(f"  ℹ️  Epoch {epoch+1}: {nan_total} NaN batches skipped")

            # ── End-of-epoch: flush any remaining accumulated gradients ──
            remaining = epoch_steps % grad_accum
            if remaining > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.model.parameters() if p.requires_grad],
                    self.config.training.max_grad_norm,
                )
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                self.optimizer.zero_grad()
                total_steps += 1

            epoch_time = time.time() - start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

            print(
                f"  ✅ Epoch {epoch+1}/{num_epochs} complete | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Steps so far: {total_steps} | "
                f"LR: {current_lr:.2e}"
            )

            metrics_history.append({
                "epoch": epoch + 1,
                "stage": stage,
                "loss": avg_epoch_loss,
                "time": epoch_time,
                "lr": current_lr,
            })

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self._save_checkpoint(f"stage{stage}_epoch{epoch+1}")

            # Validation
            if val_samples and prompt_builder:
                val_metrics = self._validate(val_samples, prompt_builder)
                metrics_history[-1].update(val_metrics)
                print(
                    f"  📊 Val — Action Acc: {val_metrics['val_action_acc']:.1%} | "
                    f"Element Acc: {val_metrics['val_element_acc']:.1%} | "
                    f"Samples: {val_metrics['val_total']}"
                )

        print(f"\n{'='*60}")
        print(f"  Stage {stage} complete — {total_steps} optimizer steps")
        print(f"  Final loss: {metrics_history[-1]['loss']:.4f}")
        print(f"{'='*60}\n")

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
            num_workers=4,
            drop_last=True,
            pin_memory=True,        # add this
            persistent_workers=True
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
        Stage 2: Multi-step imitation with teacher forcing (C2 FIX).

        Uses TrajectoryDataset and trajectory_collate_fn for proper
        trajectory-aware batching. Each trajectory's steps are kept
        in order with cumulative action history.
        """
        logger.info(f"=== Stage 2: Multi-Step Imitation ({self.config.training.stage2_epochs} epochs) ===")
        logger.info(f"Trajectories: {len(trajectories)}")

        from models.prompt_builder import PromptBuilder
        prompt_builder = PromptBuilder()

        # Pre-training diagnostics
        validate_trajectories(trajectories)

        # Filter: only use trajectories with >= 2 steps
        valid_trajectories = [t for t in trajectories if len(t.steps) >= 2]
        logger.info(
            f"Using {len(valid_trajectories)}/{len(trajectories)} trajectories "
            f"(dropped {len(trajectories) - len(valid_trajectories)} single-step)"
        )

        # C2 FIX: Use TrajectoryDataset with trajectory_collate_fn
        dataset = TrajectoryDataset(valid_trajectories)

        from functools import partial
        collate = partial(
            trajectory_collate_fn,
            processor=self.model.processor,
            prompt_builder=prompt_builder,
            max_seq_length=self.config.training.max_seq_length,
        )

        # batch_size=1 because each trajectory is already multi-step
        # shuffle=True to randomize trajectory order per epoch
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
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
        correct_candidates = 0
        total = 0

        with torch.no_grad():
            for sample in val_subset:
                try:
                    # Build prompt (candidate-based)
                    messages = prompt_builder.build_chat_messages(
                        task=sample.task,
                        candidates=sample.candidates,
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
                        pred = decoder.normalize(pred)
                        if pred.get("action") == gt.get("action"):
                            correct_actions += 1

                        # C4 FIX: compare by backend_node_id, not candidate index
                        pred_idx = pred.get("candidate", -1)
                        if isinstance(pred_idx, int) and 0 <= pred_idx < len(sample.candidates):
                            pred_node_id = sample.candidates[pred_idx].get(
                                "backend_node_id", -2
                            )
                            gt_node_id = getattr(sample, "gt_backend_node_id", -3)
                            if pred_node_id == gt_node_id and gt_node_id >= 0:
                                correct_candidates += 1

                    total += 1
                except Exception:
                    total += 1
                    logger.debug("Validation sample failed", exc_info=True)

        self.model.model.train()

        return {
            "val_action_acc": correct_actions / max(total, 1),
            "val_element_acc": correct_candidates / max(total, 1),
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
    import os
    import sys

    # ── SSL certificate bypass ───────────────────────────────
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    parser = argparse.ArgumentParser(description="VLA Web Agent Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training samples (None = full dataset)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to prior stage checkpoint to resume from "
                             "(e.g. checkpoints_v12/stage1_epoch5 for Stage 2)")
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
        max_neg_candidates=getattr(config.data, 'max_neg_candidates', 63),
    )

    # Initialize trainer
    trainer = VLATrainer(config=config, device=args.device)

    # When loading a checkpoint, skip apply_lora() in setup() to
    # prevent double PeftModel wrapping (which causes OOM).
    trainer.setup(skip_lora=bool(args.checkpoint))

    # Load prior stage checkpoint if specified
    if args.checkpoint:
        log.info(f"Loading checkpoint: {args.checkpoint}")
        trainer.model.load_lora(args.checkpoint)
        log.info("Checkpoint loaded — continuing training from prior stage weights")

        # Re-create optimizer with the newly loaded LoRA parameters
        trainable_params = [p for p in trainer.model.model.parameters() if p.requires_grad]
        trainer.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        log.info(f"Optimizer re-initialized with {len(trainable_params)} trainable params")

    # Run training stages
    if args.stage == 1:
        log.info("Loading training data for Stage 1...")
        train_samples = loader.build_training_examples(
            split="train",
            max_samples=args.max_samples,
            include_screenshot=True,
        )
        log.info(f"Loaded {len(train_samples)} training samples")
        result = trainer.train_stage1(train_samples)
        log.info(f"Stage 1 complete: {result}")

    if args.stage == 2:
        if not args.checkpoint:
            log.warning(
                "Stage 2 requested without --checkpoint. "
                "Consider passing --checkpoint checkpoints_v12/stage1_epoch5 "
                "to continue from Stage 1 weights."
            )
        log.info("Building trajectories for Stage 2...")
        trajectories = loader.build_trajectories(
            split="train",
            max_samples=args.max_samples,
            include_screenshot=True,
        )
        log.info(f"Built {len(trajectories)} trajectories")
        result = trainer.train_stage2(trajectories)
        log.info(f"Stage 2 complete: {result}")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
