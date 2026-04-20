"""
Sequential Multi-Task Runner for the VLA Web Agent.

Reads a list of tasks from a JSON file and runs them one by one,
printing a summary table at the end.

Tasks file format (tasks.json):
    [
        {"url": "https://duckduckgo.com", "task": "Search for Python tutorials"},
        {"url": "https://en.wikipedia.org", "task": "Search for Machine Learning"},
        {"url": "https://news.ycombinator.com", "task": "Find the top post today"}
    ]

Usage:
    python -m inference.run_tasks --tasks tasks.json --backend groq
    python -m inference.run_tasks --tasks tasks.json --backend groq --max-steps 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def load_tasks(path: str) -> List[Dict[str, Any]]:
    """Load tasks from a JSON file. Each task must have 'url' and 'task' keys."""
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("Tasks file must contain a JSON array of task objects.")
    for i, t in enumerate(tasks):
        if "url" not in t or "task" not in t:
            raise ValueError(
                f"Task #{i} is missing 'url' or 'task' key. Got: {t}"
            )
    return tasks


def print_summary(results: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table of all task results."""
    print("\n" + "=" * 70)
    print("  MULTI-TASK RUN SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r.get("success"))
    print(f"  Tasks: {len(tasks)}  |  Passed: {passed}  |  Failed: {len(tasks) - passed}")
    print("-" * 70)
    for i, (task_def, result) in enumerate(zip(tasks, results)):
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        steps = result.get("total_steps", 0)
        ms = result.get("total_time_ms", 0)
        task_str = task_def["task"][:45].ljust(45)
        print(f"  [{i+1}] {status}  {task_str}  steps={steps}  {ms:.0f}ms")
        if result.get("error"):
            print(f"       ⚠  {result['error']}")
    print("=" * 70 + "\n")


async def run_all(
    tasks: List[Dict[str, Any]],
    backend: str,
    groq_model: str | None,
    device: str,
    config_path: str | None,
    max_steps: int | None,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Run all tasks sequentially, sharing config but fresh browser each time."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.config import load_config
    from inference.run_agent import VLAAgent

    config = load_config(config_path)
    if groq_model:
        config.groq.model = groq_model

    # Load .env for Groq
    if backend == "groq":
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.environ.get("GROQ_API_KEY"):
            print(
                "\n⚠️  GROQ_API_KEY not set — Groq calls will fail.\n"
                "   Add it to your environment or .env file.\n"
            )

    results: List[Dict[str, Any]] = []

    for i, task_def in enumerate(tasks):
        url = task_def["url"]
        task = task_def["task"]
        print(f"\n{'─' * 60}")
        print(f"  Task {i+1}/{len(tasks)}: {task}")
        print(f"  URL: {url}")
        print(f"{'─' * 60}")

        # Each task gets a fresh agent + browser
        agent = VLAAgent(config=config, device=device, backend=backend)

        out = Path(output_dir) / f"task_{i+1:02d}"
        out.mkdir(parents=True, exist_ok=True)
        agent.output_dir = out
        agent.load_model()

        try:
            result = await agent.run_task(url=url, task=task, max_steps=max_steps)
        except Exception as exc:
            result = {
                "success": False,
                "steps": [],
                "total_steps": 0,
                "total_time_ms": 0,
                "final_url": url,
                "error": str(exc),
            }
            print(f"  ❌ Exception: {exc}")

        results.append(result)
        status = "✅ Done" if result["success"] else "❌ Failed"
        print(f"\n  {status} in {result['total_steps']} steps "
              f"({result['total_time_ms']:.0f}ms)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple VLA tasks sequentially from a JSON file."
    )
    parser.add_argument(
        "--tasks", type=str, required=True,
        help="Path to JSON tasks file (array of {url, task} objects).",
    )
    parser.add_argument(
        "--backend", type=str, default="groq", choices=["local", "groq"],
        help="Inference backend (default: groq).",
    )
    parser.add_argument(
        "--groq-model", type=str, default=None,
        help="Override Groq model name.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for local backend.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=str, default="inference_output",
        help="Root directory for per-task screenshot folders.",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    print(f"\n🚀 Running {len(tasks)} task(s) with backend='{args.backend}'")

    results = asyncio.run(run_all(
        tasks=tasks,
        backend=args.backend,
        groq_model=args.groq_model,
        device=args.device,
        config_path=args.config,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    ))

    print_summary(results, tasks)


if __name__ == "__main__":
    main()
