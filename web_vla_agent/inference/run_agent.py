"""
End-to-End Inference Pipeline for VLA Web Agent — Candidate-Based.

Reads real webpage → understands instruction → presents candidates →
generates action with candidate index → executes → repeats until done.

Features:
  - Full step loop with environment interaction
  - Candidate-based element selection
  - Uncertainty-based replanning
  - Max steps + failure detection
  - Action validation against candidate list
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
from memory.failure_detector import FailureDetector, FailureType
from utils.config import VLAConfig, load_config

logger = logging.getLogger(__name__)


class VLAAgent:
    """
    End-to-end VLA Web Agent for autonomous web navigation.

    Uses candidate-based element selection: DOM elements are presented
    as numbered candidates, and the model picks a candidate index.

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
        self.failure_detector = FailureDetector(
            loop_window=4,
            stale_threshold=3,
            max_steps=self.config.environment.max_steps,
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
            self.failure_detector.reset()
            prev_state_dict: dict | None = None

            # Save initial screenshot for debugging
            if state.screenshot:
                state.screenshot.save("debug_step_0.png")

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

                # Build candidate list from current DOM elements
                candidates = self._build_candidates_from_state(state)

                # Generate action
                action_result = await self._generate_action(
                    state, task, candidates
                )
                step_results.append(action_result)

                action = action_result.get("action")

                # ── Auto-scroll fallback ─────────────────────────
                # If model fails to generate valid action, try scrolling
                # to reveal more elements, then re-prompt
                if action is None:
                    auto_scrolled = await self._auto_scroll_fallback(state)
                    if auto_scrolled:
                        state = await self.env.extract_state()
                        logger.info("  Auto-scrolled (invalid action fallback)")
                    else:
                        logger.warning("Failed to generate valid action")
                    continue

                # Check STOP/DONE
                if action.get("action") in ("STOP", "DONE"):
                    logger.info("Model predicted STOP — task complete")
                    return self._make_result(
                        success=True,
                        steps=step_results,
                        start_time=start_time,
                        url=state.url,
                    )

                # ── Handle SCROLL action ─────────────────────────
                if action.get("action") == "SCROLL":
                    direction = action.get("direction", "down")
                    amount = int(action.get("amount", 400))
                    scroll_action = WebAction(
                        action="SCROLL",
                        direction=direction,
                        amount=amount,
                    )
                    logger.info(f"Executing: SCROLL {direction} {amount}px")
                    new_state = await self.env.step(scroll_action)
                    state = new_state
                    continue

                # Map candidate index → node_id for execution
                web_action = self._candidate_to_web_action(action, candidates)
                if web_action is None:
                    # Can't map candidate — try auto-scroll as fallback
                    auto_scrolled = await self._auto_scroll_fallback(state)
                    if auto_scrolled:
                        state = await self.env.extract_state()
                        logger.info("  Auto-scrolled (bad candidate fallback)")
                    else:
                        logger.warning("Could not map candidate to DOM element")
                    continue

                logger.info(f"Executing: {web_action.to_history_string()}")

                new_state = await self.env.step(web_action)
                logger.info(
                    f"  → Page: {new_state.page_title} URL: {new_state.url}"
                )

                # Save step screenshot for debugging
                if new_state.screenshot:
                    new_state.screenshot.save(f"debug_step_{step+1}.png")

                # Failure detection
                curr_state_dict = {
                    "url": new_state.url,
                    "page_title": new_state.page_title,
                    "html_snippet": (
                        new_state.dom_tree[:500] if new_state.dom_tree else ""
                    ),
                }
                failure = self.failure_detector.detect(
                    prev_state=prev_state_dict,
                    curr_state=curr_state_dict,
                    action=action,
                )
                if failure != FailureType.NONE:
                    logger.warning(
                        f"Failure detected: {failure.value} — stopping"
                    )
                    return self._make_result(
                        success=False,
                        steps=step_results,
                        start_time=start_time,
                        url=new_state.url,
                        error=f"Failure: {failure.value}",
                    )

                prev_state_dict = curr_state_dict
                state = new_state

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

    def _build_candidates_from_state(
        self,
        state: BrowserState,
    ) -> List[Dict[str, Any]]:
        """
        Build a candidate list from the browser state's DOM elements.

        Each candidate gets a sequential index (0, 1, 2, ...) and
        maps to the original DOM node_id for action execution.
        """
        candidates = []

        # Prefer interactable elements, but include all visible ones
        elements = state.dom_elements or []

        # Sort: interactable first
        interactable = [e for e in elements if e.get("is_interactable")]
        non_interactable = [e for e in elements if not e.get("is_interactable")]
        sorted_elements = interactable + non_interactable

        # Limit to prevent prompt overflow
        max_candidates = self.prompt_builder.max_candidates
        for i, el in enumerate(sorted_elements[:max_candidates]):
            candidates.append({
                "candidate_index": i,
                "tag": el.get("tag", "div"),
                "text": str(el.get("text", ""))[:200],
                "attributes": el.get("attributes", {}),
                "node_id": el.get("node_id", -1),
            })

        return candidates

    def _candidate_to_web_action(
        self,
        action: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[WebAction]:
        """
        Map a candidate-based action to a WebAction with the real node_id.
        """
        candidate_idx = action.get("candidate", -1)

        if candidate_idx < 0 or candidate_idx >= len(candidates):
            return None

        node_id = candidates[candidate_idx].get("node_id", -1)
        if node_id < 0:
            return None

        return WebAction(
            action=action.get("action", "CLICK").upper(),
            element_id=node_id,
            value=action.get("value", ""),
            direction=action.get("direction", "down"),
            amount=int(action.get("amount", 300)),
        )

    async def _auto_scroll_fallback(self, state: BrowserState) -> bool:
        """
        Auto-scroll the page as a fallback when the model can't find
        a valid candidate.

        Returns True if scroll was executed, False if max attempts reached.
        """
        max_scroll_attempts = getattr(
            self.config.environment, "max_scroll_attempts", 3
        )

        if not hasattr(self, "_scroll_attempts"):
            self._scroll_attempts = 0

        if self._scroll_attempts >= max_scroll_attempts:
            logger.info(
                f"  Max scroll attempts ({max_scroll_attempts}) reached"
            )
            self._scroll_attempts = 0  # Reset for next step
            return False

        self._scroll_attempts += 1
        scroll_action = WebAction(
            action="SCROLL", direction="down", amount=400
        )
        try:
            await self.env.step(scroll_action)
            logger.info(
                f"  Auto-scroll attempt {self._scroll_attempts}/"
                f"{max_scroll_attempts}"
            )
            return True
        except Exception as e:
            logger.warning(f"  Auto-scroll failed: {e}")
            return False

    async def _generate_action(
        self,
        state: BrowserState,
        task: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate and validate an action for the current state.

        Uses uncertainty estimation and regeneration if needed.
        """
        # Build prompt with candidate list
        extra_context = f"URL: {state.url}\nPage: {state.page_title}"
        messages = self.prompt_builder.build_chat_messages(
            task=task,
            candidates=candidates,
            action_history=state.action_history,
            screenshot_placeholder=state.screenshot is not None,
            extra_context=extra_context,
        )

        # Debug
        num_cands = len(candidates)
        img_size = state.screenshot.size if state.screenshot else None
        print(
            f"  [DEBUG] Candidates: {num_cands}, "
            f"Screenshot: {img_size}"
        )

        # Generate with log probabilities
        start = time.time()
        result = self.model.generate(
            messages=messages,
            image=state.screenshot,
            return_log_probs=True,
        )
        latency_ms = (time.time() - start) * 1000

        logger.info(f"  Generated: {result['text'][:200]}")
        print(f"  [DEBUG] Raw model output: {result['text'][:300]}")
        logger.info(f"  Avg log prob: {result.get('avg_log_prob', 0):.3f}")

        # Parse and validate
        action, is_valid, error = self.action_decoder.parse_and_validate(
            result["text"], num_candidates=num_cands,
        )

        # Check uncertainty — regenerate if low confidence
        uncertainty = self.uncertainty.assess(result)
        if uncertainty.should_regenerate and not is_valid:
            logger.info(
                f"  Low confidence ({uncertainty.reason}) — regenerating..."
            )
            for attempt in range(self.uncertainty.max_regenerations):
                result = self.model.generate(
                    messages=messages,
                    image=state.screenshot,
                    temperature=0.3 + attempt * 0.1,
                    return_log_probs=True,
                )
                action, is_valid, error = (
                    self.action_decoder.parse_and_validate(
                        result["text"], num_candidates=num_cands,
                    )
                )
                if is_valid:
                    logger.info(
                        f"  Regeneration {attempt + 1} succeeded"
                    )
                    break

        if not is_valid:
            logger.warning(f"  Invalid action: {error}")
            print(f"  Invalid action: {error}")
            print(f"  Raw output was: {result['text'][:200]}")

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
