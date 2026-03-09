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
from PIL import ImageDraw

logger = logging.getLogger(__name__)

# ── Interactable element filters ─────────────────────────────

INTERACTABLE_TAGS = {
    "button", "input", "textarea", "select", "a",
    "option", "label", "summary",
}
INTERACTABLE_ROLES = {
    "button", "link", "menuitem", "tab", "checkbox", "radio",
    "switch", "option", "combobox", "searchbox", "textbox",
    "listbox", "slider", "spinbutton", "treeitem",
}

# Common stop words to ignore when matching task to candidates
_STOP_WORDS = {
    "the", "a", "an", "in", "on", "to", "of", "and", "or", "for",
    "is", "it", "at", "by", "from", "into", "then", "click",
    "type", "select", "scroll", "search", "find", "go", "open",
    "press", "enter", "submit", "field", "button", "page",
}


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
        self.output_dir = None   # Set externally for screenshot saving

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
        previous_action_str = ""    # Track duplicate actions
        duplicate_count = 0

        # Start browser and navigate
        await self.env.start()
        try:
            state = await self.env.reset(url)
            self.failure_detector.reset()
            prev_state_dict: dict | None = None

            # Save initial screenshot
            if state.screenshot and self.output_dir:
                path = self.output_dir / "step_00_initial.png"
                state.screenshot.save(str(path))
                print(f"  📸 Saved: {path}")

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
                candidates = self._build_candidates_from_state(state, task=task)

                # NOTE: Visual highlighting is DISABLED.
                # The model was trained on raw Mind2Web screenshots
                # (no bounding box overlays). Annotating the screenshot
                # introduces a vision distribution shift that causes the
                # model to default to SCROLL.

                # Generate action (use raw screenshot — matches training)
                action_result = await self._generate_action(
                    state, task, candidates,
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

                # ── Duplicate action detection ───────────────────
                # If the model generates the same action twice in a
                # row, the task is likely complete. Auto-DONE.
                action_str = json.dumps(action, sort_keys=True)
                if action_str == previous_action_str:
                    duplicate_count += 1
                    if duplicate_count >= 2:
                        # Only report success if at least one meaningful
                        # action (CLICK/TYPE/SELECT) was taken.  Scrolling
                        # 3× and auto-completing is NOT a success.
                        meaningful = any(
                            s.get("action", {}).get("action")
                            in ("CLICK", "TYPE", "SELECT")
                            for s in step_results
                        )
                        label = "✅" if meaningful else "⚠️"
                        logger.info(
                            f"Same action repeated {duplicate_count+1} times — "
                            f"auto-completing task (meaningful={meaningful})"
                        )
                        print(
                            f"  {label} Auto-DONE: same action repeated "
                            f"{duplicate_count+1} times"
                        )
                        return self._make_result(
                            success=meaningful,
                            steps=step_results,
                            start_time=start_time,
                            url=state.url,
                        )
                else:
                    duplicate_count = 0
                previous_action_str = action_str

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

                # Save step screenshot
                if new_state.screenshot and self.output_dir:
                    action_type = action.get('action', 'UNK')
                    candidate = action.get('candidate', '?')
                    fname = f"step_{step+1:02d}_{action_type}_c{candidate}.png"
                    path = self.output_dir / fname
                    new_state.screenshot.save(str(path))
                    print(f"  📸 Saved: {path}")

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
        task: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Build a filtered, scored candidate list from browser state.

        Pipeline:
        1. Filter to interactable + visible elements only
        2. Score by task-text similarity (word overlap)
        3. Sort by score (best matches first)
        4. Keep top N candidates
        5. Enrich with bounding box data
        """
        elements = state.dom_elements or []

        # ── Step 1: Filter to interactable only ──────────────
        interactable = []
        for el in elements:
            if not el.get("is_visible", True):
                continue

            tag = el.get("tag", "").lower()
            role = el.get("attributes", {}).get("role", "").lower()

            is_ok = (
                tag in INTERACTABLE_TAGS
                or role in INTERACTABLE_ROLES
                or el.get("is_interactable", False)
            )
            if is_ok:
                interactable.append(el)

        # ── Step 2: Score by task similarity ──────────────────
        task_words = {
            w.lower()
            for w in task.split()
            if len(w) > 2 and w.lower() not in _STOP_WORDS
        }

        scored = []
        for el in interactable:
            # Build searchable text from element
            parts = [el.get("text", "")]
            attrs = el.get("attributes", {})
            for key in ("placeholder", "aria-label", "title", "name", "value", "alt"):
                if key in attrs:
                    parts.append(attrs[key])
            el_text = " ".join(parts).lower()

            # Word overlap score
            score = sum(1 for w in task_words if w in el_text)

            # Boost input/textarea for TYPE tasks
            tag = el.get("tag", "").lower()
            if tag in ("input", "textarea") and any(
                w in task.lower() for w in ("type", "enter", "write", "input", "fill")
            ):
                score += 3

            scored.append((score, el))

        # ── Step 3: Sort and limit ────────────────────────────
        scored.sort(key=lambda x: x[0], reverse=True)
        max_candidates = min(self.prompt_builder.max_candidates, 20)
        selected = scored[:max_candidates]

        # ── Step 4: Build candidate dicts ────────────────────
        # NOTE: bbox is intentionally set to None.
        # The model was trained on Mind2Web candidates which have NO
        # bounding box data. Including bbox causes the prompt builder
        # to append "at (x,y WxH)" text the model never learned,
        # creating a format mismatch that causes SCROLL-only output.
        candidates = []
        for i, (score, el) in enumerate(selected):
            # Semantic text extraction: fall back to common DOM
            # attributes when inner text is empty.  This matches
            # Mind2Web training format where candidates always
            # carried meaningful labels.
            attrs = el.get("attributes", {})
            text = (
                el.get("text", "")
                or attrs.get("placeholder", "")
                or attrs.get("aria-label", "")
                or attrs.get("title", "")
                or attrs.get("name", "")
                or attrs.get("value", "")
            )
            text = str(text)[:200]

            candidates.append({
                "candidate_index": i,
                "tag": el.get("tag", "div"),
                "text": text,
                "attributes": attrs,
                "node_id": el.get("node_id", -1),
                "bbox": None,
                "score": score,
            })

        if not candidates and interactable:
            # Fallback: keep first 20 interactable without scoring
            for i, el in enumerate(interactable[:20]):
                attrs = el.get("attributes", {})
                text = (
                    el.get("text", "")
                    or attrs.get("placeholder", "")
                    or attrs.get("aria-label", "")
                    or attrs.get("title", "")
                    or attrs.get("name", "")
                    or attrs.get("value", "")
                )
                text = str(text)[:200]

                candidates.append({
                    "candidate_index": i,
                    "tag": el.get("tag", "div"),
                    "text": text,
                    "attributes": attrs,
                    "node_id": el.get("node_id", -1),
                    "bbox": None,
                    "score": 0,
                })

        # Debug: print candidate summary for verification
        print("\n===== CANDIDATES =====")
        for c in candidates:
            attrs = c.get("attributes", {})
            print(
                f"[{c['candidate_index']}] "
                f"{c['tag']} "
                f"text='{c['text']}' "
                f"placeholder='{attrs.get('placeholder', '')}' "
                f"name='{attrs.get('name', '')}' "
                f"aria='{attrs.get('aria-label', '')}'"
            )
        print("======================\n")

        return candidates

    # ── Visual highlighting ──────────────────────────────────

    def _highlight_candidates(
        self,
        screenshot: Image.Image,
        candidates: List[Dict[str, Any]],
        viewport_info: Dict[str, Any],
    ) -> Image.Image:
        """
        Draw bounding boxes and index labels on the screenshot
        for each candidate element. Returns a new annotated image.
        """
        img = screenshot.copy()
        draw = ImageDraw.Draw(img)

        sx = viewport_info.get("scrollX", 0)
        sy = viewport_info.get("scrollY", 0)

        # Color palette for different element types
        tag_colors = {
            "input": "#FF4444",
            "textarea": "#FF4444",
            "button": "#4444FF",
            "a": "#44AA44",
            "select": "#FF8800",
        }

        for c in candidates:
            bbox = c.get("bbox", [0, 0, 0, 0])
            x, y, w, h = bbox

            # Adjust for scroll offset (bbox is page-relative)
            x1 = x - sx
            y1 = y - sy
            x2 = x1 + w
            y2 = y1 + h

            # Skip elements outside viewport
            if x2 < 0 or y2 < 0 or x1 > img.width or y1 > img.height:
                continue

            tag = c.get("tag", "").lower()
            color = tag_colors.get(tag, "#FF0000")

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw index label
            idx = c.get("candidate_index", "?")
            label = str(idx)
            label_y = max(0, y1 - 14)
            draw.rectangle([x1, label_y, x1 + 18, label_y + 14], fill=color)
            draw.text((x1 + 2, label_y), label, fill="white")

        return img

    # ── Action masking ───────────────────────────────────────

    @staticmethod
    def _mask_action(
        action: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Mask impossible actions based on available candidates.

        - TYPE without input/textarea → CLICK
        - SELECT without select → CLICK
        - Candidate index out of range → clamp
        """
        action_type = action.get("action", "")
        input_tags = {"input", "textarea"}
        has_inputs = any(c.get("tag", "").lower() in input_tags for c in candidates)
        has_selects = any(c.get("tag", "").lower() == "select" for c in candidates)

        if action_type == "TYPE" and not has_inputs:
            logger.info("  Action mask: TYPE → CLICK (no input fields)")
            action["action"] = "CLICK"
            action.pop("value", None)

        if action_type == "SELECT" and not has_selects:
            logger.info("  Action mask: SELECT → CLICK (no select fields)")
            action["action"] = "CLICK"
            action.pop("value", None)

        # Clamp candidate index
        if "candidate" in action and candidates:
            idx = action["candidate"]
            if isinstance(idx, int) and idx >= len(candidates):
                action["candidate"] = len(candidates) - 1
                logger.info(f"  Action mask: clamped candidate {idx} → {action['candidate']}")

        return action

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
        screenshot_override: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Generate and validate an action for the current state.

        Uses uncertainty estimation and regeneration if needed.
        screenshot_override: annotated screenshot with visual highlights.
        """
        # Use annotated screenshot if provided (has bounding boxes drawn)
        screenshot_to_use = screenshot_override or state.screenshot

        # Build prompt with candidate list
        extra_context = f"URL: {state.url}\nPage: {state.page_title}"
        messages = self.prompt_builder.build_chat_messages(
            task=task,
            candidates=candidates,
            action_history=state.action_history,
            screenshot_placeholder=screenshot_to_use is not None,
            extra_context=extra_context,
        )

        # Debug
        num_cands = len(candidates)
        img_size = screenshot_to_use.size if screenshot_to_use else None
        print(
            f"  [DEBUG] Candidates: {num_cands}, "
            f"Screenshot: {img_size}"
        )

        # Generate with log probabilities
        start = time.time()
        result = self.model.generate(
            messages=messages,
            image=screenshot_to_use,
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
                    image=screenshot_to_use,
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

        # Apply action masking (prevent impossible actions)
        if is_valid and action is not None:
            action = self._mask_action(action, candidates)

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
    parser.add_argument("--output-dir", type=str, default="inference_output",
                        help="Directory to save step screenshots (default: inference_output)")
    args = parser.parse_args()

    from utils.logging import get_logger
    log = get_logger("vla.agent")

    config = load_config(args.config)

    # Setup output directory for screenshots
    output_dir = None
    if args.output_dir:
        from pathlib import Path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Screenshots will be saved to: {output_dir}")

    agent = VLAAgent(
        config=config,
        device=args.device,
        use_mock=args.mock,
    )
    agent.output_dir = output_dir
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
