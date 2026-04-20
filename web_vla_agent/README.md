# Web VLA Agent

End-to-end multimodal web automation agent built on Qwen2-VL.

This project trains and runs a Vision-Language Action (VLA) agent that reads a web page (DOM + screenshot), understands a natural language task, predicts the next UI action as JSON, executes the action in a browser, and repeats until task completion.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Inference (Run the Agent)](#inference-run-the-agent)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Troubleshooting](#troubleshooting)

## What This Project Does

Given a task like:

- "Search flights from NYC to LA"
- "Open the product page and add item to cart"

The agent:

1. Opens a browser page.
2. Extracts interactable elements from DOM as numbered candidates.
3. Builds a prompt with task + candidates + action history + screenshot.
4. Uses Qwen2-VL to generate one JSON action.
5. Validates and executes that action (`CLICK`, `TYPE`, `SELECT`, or `SCROLL`).
6. Repeats for multiple steps.

It supports:

- Supervised training (single-step + multi-step imitation)
- LoRA/QLoRA fine-tuning
- Uncertainty-based regeneration
- Evaluation metrics (candidate accuracy, action accuracy, per-action F1)
- Playwright-based live browser mode and mock mode

## How It Works

Core pipeline:

1. **Data loader** (`data/mind2web_loader.py`) loads `osunlp/Multimodal-Mind2Web` and prepares candidate-based examples.
2. **Prompt builder** (`models/prompt_builder.py`) creates strict prompts that force JSON action output.
3. **Model wrapper** (`models/vla_model.py`) loads Qwen2-VL, applies LoRA, and runs generation/loss.
4. **Action decoder** (`models/action_decoder.py`) parses model text into valid action JSON.
5. **Browser env** (`environment/playwright_env.py`) executes actions and extracts next state.
6. **Failure + uncertainty modules** (`memory/failure_detector.py`, `models/uncertainty.py`) guard against loops and low-confidence outputs.

## Project Structure

```text
web_vla_agent/
  configs/
    default.yaml              # Main config
  data/
    mind2web_loader.py        # Dataset loader + candidate construction
    preprocessing.py
    augmentation.py
  environment/
    dom_serializer.py         # DOM serialization
    playwright_env.py         # Browser environment
  models/
    vla_model.py              # Qwen2-VL wrapper + LoRA + generation
    prompt_builder.py         # Prompt construction
    action_decoder.py         # JSON parsing/validation
    uncertainty.py            # Confidence and regeneration logic
  training/
    train_supervised.py       # Stage 1 and Stage 2 training
  inference/
    run_agent.py              # End-to-end agent runtime
  evaluation/
    evaluate.py               # Evaluation and metrics report
  memory/
    failure_detector.py       # Loop/staleness/error detection
  utils/
    config.py                 # Dataclass config loader
    logging.py                # Console + JSONL logging
  tests/
    test_smoke.py             # Broad smoke tests
    test_lora_targets.py      # LoRA target discovery tests
  main.py                     # Minimal command index
  pyproject.toml              # Dependencies and CLI entry points
```

## Requirements

Minimum:

- Python 3.10+
- PyTorch-compatible GPU recommended (CUDA)
- Internet access to download:
  - Hugging Face model (`Qwen/Qwen2-VL-2B-Instruct` by default)
  - Dataset (`osunlp/Multimodal-Mind2Web`)
- Playwright browser binaries for live browser mode

Notes:

- CPU-only execution is possible for basic testing, but model inference/training is very slow.
- QLoRA path requires `bitsandbytes` support through Transformers quantization.

## Installation

From project root:

```bash
pip install -e .
```

Install dev dependencies (tests/lint):

```bash
pip install -e .[dev]
```

Install Playwright browsers:

```bash
python -m playwright install chromium
```

Optional environment variables:

- `VLA_CONFIG`: path to custom YAML config
- `WANDB_API_KEY`: only if enabling Weights & Biases logging

## Quick Start

### 1) Run smoke tests (safe, no model download needed)

```bash
python -m tests.test_smoke
```

### 2) Run agent in mock mode

```bash
python -m inference.run_agent --url "https://example.com" --task "Click the most relevant link" --mock --device cuda
```

### 3) Train (limited samples first)

```bash
python -m training.train_supervised --config configs/default.yaml --device cuda --stage 1 --max-samples 200
```

### 4) Evaluate

```bash
python -m evaluation.evaluate --config configs/default.yaml --device cuda --split test_task --max-samples 100
```

## Configuration

Default config file: `configs/default.yaml`

Main sections:

- `model`: base model, generation settings, LoRA/QLoRA setup, image token limits
- `training`: epochs, lr, batching, grad accumulation, checkpoint path
- `data`: dataset id and DOM/action history limits
- `environment`: browser behavior (headless, viewport, timeout, max_steps)
- `uncertainty`: regeneration thresholds
- `evaluation`: default test splits
- `logging`: log level, log dir, wandb toggle

Use a custom config:

```bash
python -m training.train_supervised --config path/to/config.yaml
```

Or set globally:

```bash
# Linux/macOS
export VLA_CONFIG=path/to/config.yaml

# Windows PowerShell
$env:VLA_CONFIG="path\\to\\config.yaml"
```

## Training

Entry point: `training/train_supervised.py`

Stages:

- **Stage 1**: Single-step imitation learning
- **Stage 2**: Multi-step imitation (trajectory-level with action history)

Examples:

```bash
# Stage 1 only
python -m training.train_supervised --stage 1 --device cuda

# Stage 1 + Stage 2
python -m training.train_supervised --stage 2 --device cuda

# Small debug run
python -m training.train_supervised --stage 1 --max-samples 50 --device cuda
```

Checkpoints are saved under `training.checkpoint_dir` (default in YAML: `checkpoints_v2`).

## Inference (Run the Agent)

Entry point: `inference/run_agent.py`

Required args:

- `--url`: starting page
- `--task`: natural language instruction

Common args:

- `--checkpoint`: LoRA checkpoint path
- `--device`: `cuda` or `cpu`
- `--mock`: use mock browser mode
- `--max-steps`: override config max steps

Examples:

```bash
# Live browser mode
python -m inference.run_agent --url "https://www.google.com" --task "Search for weather in New York" --device cuda

# With fine-tuned LoRA checkpoint
python -m inference.run_agent --url "https://example.com" --task "Open the contact page" --checkpoint checkpoints_v2/stage1_epoch5 --device cuda
```

## Evaluation

Entry point: `evaluation/evaluate.py`

Supported splits:

- `train`
- `test_task`
- `test_website`
- `test_domain`

Example:

```bash
python -m evaluation.evaluate --split test_task --max-samples 200 --device cuda
```

Saved artifact:

- `evaluation_results.json`

Key reported metrics:

- parse success rate
- candidate accuracy
- action accuracy
- value accuracy
- full match accuracy
- per-action precision/recall/F1
- latency stats
- failure breakdown

## Testing

Run all unit tests with pytest:

```bash
pytest -q
```

Or run the built-in smoke harness:

```bash
python -m tests.test_smoke
```

LoRA target discovery test:

```bash
python -m tests.test_lora_targets
```

## Outputs and Artifacts

Typical generated files/folders:

- `checkpoints_v2/...` or configured checkpoint directory
- `logs/vla_*.jsonl` structured run logs
- `evaluation_results.json` evaluation summary
- `debug_step_*.png` screenshots during inference

## Troubleshooting

### 1) `playwright` errors or browser not found

Run:

```bash
python -m playwright install chromium
```

### 2) CUDA out-of-memory during training

Reduce these in config:

- `training.max_seq_length`
- `model.image_max_pixels`
- effective batch size (`batch_size * gradient_accumulation_steps`)

### 3) Slow setup due to model/dataset download

First run downloads large assets from Hugging Face. This is expected.

### 4) Agent returns invalid actions repeatedly

- try `--mock` first to validate pipeline
- inspect `debug_step_*.png`
- ensure candidates are being extracted from page
- verify checkpoint path (if using `--checkpoint`)

### 5) Config file not picked up

Pass it explicitly:

```bash
python -m inference.run_agent --config configs/default.yaml ...
```

or set `VLA_CONFIG`.

## CLI Shortcuts

After `pip install -e .`, these console scripts are available:

- `vla-train` -> `training.train_supervised:main`
- `vla-eval` -> `evaluation.evaluate:main`
- `vla-agent` -> `inference.run_agent:main`

Example:

```bash
vla-agent --url "https://example.com" --task "Click login" --mock
```
