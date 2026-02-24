"""
End-to-End Inference Pipeline for VLA Web Agent.

Reads real webpage → understands instruction → grounds element →
generates action → executes → repeats until done.

Features:
  - Full step loop with environment interaction
  - Uncertainty-based replanning
  - Max steps + failure detection
  - Action validation against DOM
  - Detailed logging
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from models.action_decoder import ActionDecoder
from models.prompt_builder import PromptBuilder
from models.uncertainty import TokenUncertainty
from environment.playwright_env import BrowserEnvironment, BrowserState, WebAction
from utils.config import VLAConfig, load_config

logger = logging.getLogger(__name__)


class VLAAgent:
    """
    End-to-end VLA Web Agent for autonomous web navigation.

    Parameters
    ----------
    config : VLAConfig, optional
    device : str
    use_mock : bool
        If True, use mock browser environment.
    """

    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        device: str = "cuda",
        use_mock: bool = False,
    ):
        self.config = config or load_config()
        self.device = device
        self.use_mock = use_mock

        self.model = None
        self.prompt_builder = PromptBuilder()
        self.action_decoder = ActionDecoder()
        self.uncertainty = TokenUncertainty(
            min_log_prob=self.config.uncertainty.min_log_prob_threshold,
            beam_width=self.config.uncertainty.beam_width,
            max_regenerations=self.config.uncertainty.max_regenerations,
        )

        self.env = BrowserEnvironment(
            max_steps=self.config.environment.max_steps,
            timeout=self.config.environment.timeout_ms,
            headless=self.config.environment.headless,
            viewport_width=self.config.environment.viewport_width,
            viewport_height=self.config.environment.viewport_height,
            use_mock=use_mock,
        )

    def load_model(self, checkpoint: Optional[str] = None) -> None:
        """Load the VLA model and optionally LoRA weights."""
        from models.vla_model import VLAModel

        self.model = VLAModel(config=self.config, device=self.device)
        self.model.load()

        if checkpoint:
            self.model.load_lora(checkpoint)
            logger.info(f"Loaded checkpoint: {checkpoint}")

    async def run_task(
        self,
        url: str,
        task: str,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a full web task autonomously.

        Parameters
        ----------
        url : str
            Starting URL.
        task : str
            Natural language task instruction.
        max_steps : int, optional
            Override max steps.

        Returns
        -------
        dict with:
          - "success": bool
          - "steps": list of step results
          - "total_steps": int
          - "total_time_ms": float
          - "final_url": str
          - "error": str (if any)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_steps = max_steps or self.config.environment.max_steps
        start_time = time.time()
        step_results = []

        # Start browser and navigate
        await self.env.start()
        try:
            state = await self.env.reset(url)
            logger.info(f"Task: {task}")
            logger.info(f"URL: {url}")
            logger.info(f"Page: {state.page_title}")

            for step in range(max_steps):
                logger.info(f"\n--- Step {step + 1}/{max_steps} ---")

                # Check error/done conditions
                if state.error:
                    logger.warning(f"Error detected: {state.error}")
                    return self._make_result(
                        success=False,
                        steps=step_results,
                        start_time=start_time,
                        url=state.url,
                        error=state.error,
                    )

                if state.done:
                    logger.info("Task marked as done by environment")
                    return self._make_result(
                        success=True,
                        steps=step_results,
                        start_time=start_time,
                        url=state.url,
                    )

                # Generate action
                action_result = await self._generate_action(state, task)
                step_results.append(action_result)

                action = action_result.get("action")
                if action is None:
                    logger.warning("Failed to generate valid action")
                    continue

                # Check DONE
                if action.get("action") == "DONE":
                    logger.info("Model predicted DONE — task complete")
                    return self._make_result(
                        success=True,
                        steps=step_results,
                        start_time=start_time,
                        url=state.url,
                    )

                # Execute action
                web_action = WebAction.from_dict(action)
                logger.info(f"Executing: {web_action.to_history_string()}")

                state = await self.env.step(web_action)
                logger.info(f"  → Page: {state.page_title} URL: {state.url}")

            # Max steps reached
            return self._make_result(
                success=False,
                steps=step_results,
                start_time=start_time,
                url=state.url,
                error="Max steps reached",
            )

        finally:
            await self.env.close()

    async def _generate_action(
        self,
        state: BrowserState,
        task: str,
    ) -> Dict[str, Any]:
        """
        Generate and validate an action for the current state.

        Uses uncertainty estimation and regeneration if needed.
        """
        # Build prompt
        extra_context = f"URL: {state.url}\nPage: {state.page_title}"
        messages = self.prompt_builder.build_chat_messages(
            task=task,
            serialized_dom=state.serialized_dom,
            action_history=state.action_history,
            screenshot_placeholder=state.screenshot is not None,
            extra_context=extra_context,
        )

        # Get valid node IDs from current DOM
        valid_ids = [el.get("node_id", -1) for el in state.dom_elements]

        # Generate with log probabilities
        start = time.time()
        result = self.model.generate(
            messages=messages,
            image=state.screenshot,
            return_log_probs=True,
        )
        latency_ms = (time.time() - start) * 1000

        logger.info(f"  Generated: {result['text'][:200]}")
        logger.info(f"  Avg log prob: {result.get('avg_log_prob', 0):.3f}")

        # Parse and validate
        action, is_valid, error = self.action_decoder.parse_and_validate(
            result["text"], valid_ids,
        )

        # Check uncertainty — regenerate if low confidence
        uncertainty = self.uncertainty.assess(result)
        if uncertainty.should_regenerate and not is_valid:
            logger.info(f"  Low confidence ({uncertainty.reason}) — regenerating...")
            for attempt in range(self.uncertainty.max_regenerations):
                result = self.model.generate(
                    messages=messages,
                    image=state.screenshot,
                    temperature=0.3 + attempt * 0.1,
                    return_log_probs=True,
                )
                action, is_valid, error = self.action_decoder.parse_and_validate(
                    result["text"], valid_ids,
                )
                if is_valid:
                    logger.info(f"  Regeneration {attempt + 1} succeeded")
                    break

        if not is_valid:
            logger.warning(f"  Invalid action: {error}")

        return {
            "action": action if is_valid else None,
            "raw_output": result["text"],
            "avg_log_prob": result.get("avg_log_prob", 0),
            "latency_ms": latency_ms,
            "is_valid": is_valid,
            "error": error if not is_valid else "",
        }

    def _make_result(
        self,
        success: bool,
        steps: list,
        start_time: float,
        url: str,
        error: str = "",
    ) -> Dict[str, Any]:
        """Build the final result dict."""
        total_time = (time.time() - start_time) * 1000
        return {
            "success": success,
            "steps": steps,
            "total_steps": len(steps),
            "total_time_ms": total_time,
            "final_url": url,
            "error": error,
        }


# ── Entry point ──────────────────────────────────────────────

def main():
    """Run the VLA agent on a task."""
    import argparse

    parser = argparse.ArgumentParser(description="VLA Web Agent Inference")
    parser.add_argument("--url", type=str, required=True, help="Starting URL")
    parser.add_argument("--task", type=str, required=True, help="Task instruction")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Use mock browser")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    from utils.logging import get_logger
    log = get_logger("vla.agent")

    config = load_config(args.config)
    agent = VLAAgent(
        config=config,
        device=args.device,
        use_mock=args.mock,
    )
    agent.load_model(checkpoint=args.checkpoint)

    result = asyncio.run(agent.run_task(
        url=args.url,
        task=args.task,
        max_steps=args.max_steps,
    ))

    print(f"\n{'=' * 60}")
    print(f"  Task: {args.task}")
    print(f"  Success: {result['success']}")
    print(f"  Steps: {result['total_steps']}")
    print(f"  Time: {result['total_time_ms']:.0f}ms")
    print(f"  Final URL: {result['final_url']}")
    if result['error']:
        print(f"  Error: {result['error']}")
    print(f"{'=' * 60}")

    # Print step details
    for i, step in enumerate(result['steps']):
        action = step.get('action')
        print(f"\n  Step {i + 1}:")
        print(f"    Action: {action}")
        print(f"    Valid: {step.get('is_valid', False)}")
        print(f"    Latency: {step.get('latency_ms', 0):.0f}ms")


if __name__ == "__main__":
    main()
