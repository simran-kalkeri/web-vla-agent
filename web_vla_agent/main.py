"""
VLA Web Agent — main entry point (Refactored).

Modes:
  train      — 3-phase supervised training
  evaluate   — evaluation on Mind2Web test splits with ablation
  ablation   — run all ablation presets
  run        — full agent pipeline (plan → ground → act → observe)

Usage:
  python main.py train   --config configs/default.yaml
  python main.py evaluate --checkpoint checkpoints/latest.pt --ablation full
  python main.py ablation --checkpoint checkpoints/latest.pt
  python main.py run     --instruction "Book a flight from NYC to LA"
"""
from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from utils.config import load_config, VLAConfig
from utils.logging import get_logger


def cmd_train(args: argparse.Namespace) -> None:
    from training.train_supervised import SupervisedTrainer
    config = load_config(args.config)
    trainer = SupervisedTrainer(config=config, device=args.device)
    trainer.train(max_samples=args.max_samples, resume_checkpoint=args.resume)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluation.evaluate import Evaluator, ABLATION_PRESETS
    config = load_config(args.config)
    ablation = ABLATION_PRESETS.get(args.ablation, ABLATION_PRESETS["full"])
    evaluator = Evaluator(
        config=config, checkpoint_path=args.checkpoint,
        device=args.device, ablation=ablation,
    )
    evaluator.evaluate(
        splits=args.splits, max_samples=args.max_samples, output_path=args.output,
    )


def cmd_ablation(args: argparse.Namespace) -> None:
    from evaluation.evaluate import run_all_ablations
    config = load_config(args.config)
    run_all_ablations(
        config=config, checkpoint_path=args.checkpoint,
        device=args.device, max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """
    Full agent pipeline (evaluation only — planner active here):
      Instruction → Plan → (Ground → Uncertainty → Act → Observe) loop.
    """
    import torch
    config = load_config(args.config)
    logger = get_logger("agent")

    # ── Planner (EVALUATION ONLY) ────────────────────────────
    from models.planner import TaskPlanner
    from models.encoders import TextEncoder
    from models.graph_dom_encoder import GraphDOMEncoder, structural_features, build_edge_index
    from models.uncertainty import EntropyUncertainty
    from data.preprocessing import DOMProcessor

    # Load LLM for task decomposition
    llm_fn = None
    try:
        from models.qwen2vl_llm import create_planner_llm
        logger.info(f"Loading {config.planner.model_name} for task planning ({args.device})...")
        llm_fn = create_planner_llm(
            model_name=config.planner.model_name,
            device=args.device,
        )
        logger.info("Planner LLM loaded ✓")
    except Exception as e:
        logger.warning(f"Failed to load LLM: {e}")
        logger.warning("Falling back to rule-based planner")

    planner = TaskPlanner(config=config.planner, llm_fn=llm_fn)
    subgoals = planner.decompose(args.instruction)

    logger.info(f"Task: {args.instruction}")
    logger.info(f"Decomposed into {len(subgoals)} subgoals:")
    for sg in subgoals:
        logger.info(f"  [{sg.subgoal_id}] {sg.description}")

    # ── Encoders (frozen) ────────────────────────────────────
    text_encoder = TextEncoder(
        model_name=config.model.text_encoder,
        output_dim=config.model.text_dim,
        device=args.device,
    )

    # ── Graph encoder ────────────────────────────────────────
    graph_encoder = GraphDOMEncoder(
        text_dim=config.model.text_dim,
        hidden_dim=config.model.hidden_dim,
    ).to(args.device)
    graph_encoder.eval()

    # ── Uncertainty ──────────────────────────────────────────
    uncertainty = EntropyUncertainty(
        threshold_tau=config.uncertainty.entropy_threshold,
        temperature=config.uncertainty.temperature,
    )

    # ── Memory ───────────────────────────────────────────────
    from memory.long_term import LongTermMemory
    from memory.short_term import ShortTermMemory
    from memory.failure_detector import FailureDetector

    ltm = LongTermMemory(max_retries=config.memory.max_retries_per_subgoal)
    ltm.add_subgoals(subgoals)
    stm = ShortTermMemory(capacity=config.memory.short_term_capacity)
    fd = FailureDetector(
        loop_window=config.memory.loop_detection_window,
        stale_threshold=config.memory.stale_state_threshold,
    )
    dom_proc = DOMProcessor(max_elements=config.data.max_dom_elements)

    # ── Browser (mock) ───────────────────────────────────────
    from environment.playwright_env import BrowserEnvironment, WebAction

    env = BrowserEnvironment(config=config.environment, use_mock=True)

    async def _run_loop():
        await env.start()
        obs = await env.reset(args.url or "about:blank")
        logger.info(f"Environment reset → {obs.url}")

        step = 0
        while not ltm.all_done() and step < 30:
            sg = ltm.next_pending()
            if sg is None:
                break

            ltm.activate(sg.subgoal_id)
            logger.info(f"\nStep {step}: subgoal [{sg.subgoal_id}] {sg.description}")

            # Parse DOM
            dom_elements = dom_proc.parse(obs.dom_tree)
            logger.info(f"  DOM: {len(dom_elements)} elements parsed")

            if dom_elements:
                sigs = [el.signature for el in dom_elements]
                text_embs = text_encoder(sigs)
                with torch.no_grad():
                    graph_embs = graph_encoder.encode_dom(dom_elements, text_embs)

                # Check uncertainty
                scores = graph_embs.unsqueeze(0)  # proxy scores
                replan = uncertainty.should_replan(scores)
                if replan.any():
                    logger.warning(f"  High entropy — replan triggered")
            else:
                logger.warning("  No DOM elements found")

            # Execute action
            action = WebAction(
                action_type=sg.action_category,
                element_selector="mock-element",
                value="",
            )
            prev_state = {"url": obs.url, "html_snippet": obs.dom_tree[:500], "page_title": obs.page_title}
            obs = await env.step(action)
            curr_state = {"url": obs.url, "html_snippet": obs.dom_tree[:500], "page_title": obs.page_title}
            stm.push(prev_state, action.to_dict())

            # Failure detection
            failure = fd.detect(prev_state, curr_state, action.to_dict())
            if failure.value != "none":
                logger.warning(f"  Failure detected: {failure.value}")
                ltm.mark_failed(sg.subgoal_id, failure.value)
                if ltm.should_replan():
                    new_sgs = planner.replan(args.instruction, sg, obs.url, obs.page_title)
                    ltm.add_subgoals(new_sgs)
            else:
                ltm.mark_complete(sg.subgoal_id)
                logger.info(f"  Completed ✓")

            step += 1

        await env.close()
        progress = ltm.get_progress()
        logger.info(f"\nFinal: {progress['completed']}/{progress['total']} subgoals completed")

    asyncio.run(_run_loop())


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add --config and --device to a subparser."""
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")


def main():
    parser = argparse.ArgumentParser(
        description="VLA Web Agent (CPU-Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subs = parser.add_subparsers(dest="command", help="Command")

    # Train
    p = subs.add_parser("train", help="3-phase supervised training")
    _add_common_args(p)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)

    # Evaluate
    p = subs.add_parser("evaluate", help="Evaluation")
    _add_common_args(p)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str, default="results.json")
    p.add_argument("--ablation", type=str, default="full")

    # Ablation
    p = subs.add_parser("ablation", help="Run all ablation presets")
    _add_common_args(p)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="ablation_results")

    # Run (eval only — planner active)
    p = subs.add_parser("run", help="Full agent pipeline (eval only)")
    _add_common_args(p)
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument("--url", type=str, default=None)

    args = parser.parse_args()

    cmds = {"train": cmd_train, "evaluate": cmd_evaluate, "ablation": cmd_ablation, "run": cmd_run}
    fn = cmds.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
