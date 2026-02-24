"""
Smoke Tests for VLA Web Agent (Multimodal Sequential Architecture).

Validates that all new modules import, instantiate, and work correctly
WITHOUT requiring GPU or model downloads.
"""
from __future__ import annotations

import asyncio
import json
import sys


def test_imports():
    """All new modules import without error."""
    print("  [1] Testing imports...", end=" ")
    from environment.dom_serializer import DOMSerializer, SerializedNode
    from environment.playwright_env import BrowserEnvironment, BrowserState, WebAction
    from models.prompt_builder import PromptBuilder
    from models.action_decoder import ActionDecoder
    from models.uncertainty import TokenUncertainty, UncertaintyResult
    from utils.config import VLAConfig, load_config
    from utils.logging import get_logger
    from memory.failure_detector import FailureDetector, FailureType
    print("PASS ✓")


def test_config():
    """Config loads and has correct new structure."""
    print("  [2] Testing config...", end=" ")
    from utils.config import load_config
    cfg = load_config()
    assert cfg.model.name == "Qwen/Qwen2-VL-2B-Instruct"
    assert cfg.model.use_qlora is True
    assert cfg.model.lora_r == 16
    assert cfg.training.stage1_epochs == 5
    assert cfg.training.stage2_epochs == 10
    assert cfg.training.bf16 is True
    assert cfg.data.max_dom_nodes == 500
    assert cfg.uncertainty.min_log_prob_threshold == -2.0
    assert cfg.environment.max_steps == 30
    print("PASS ✓")


def test_dom_serializer():
    """DOM serializer produces structured output."""
    print("  [3] Testing DOM serializer...", end=" ")
    from environment.dom_serializer import DOMSerializer

    serializer = DOMSerializer(max_nodes=500)
    html = '''<html><body>
    <button id="search" role="tab">Search and reserve a car</button>
    <input type="text" placeholder="Enter city" id="city"/>
    <a href="/results">View Results</a>
    <div class="container"><span>Label</span></div>
    </body></html>'''

    text, nodes = serializer.serialize_from_html(html)
    assert len(nodes) > 0, "No nodes extracted"
    interactable = [n for n in nodes if n.is_interactable]
    assert len(interactable) >= 3, f"Expected >=3 interactable, got {len(interactable)}"
    assert "button" in text
    assert "Search and reserve a car" in text
    assert "node id=" in text
    assert "</node>" in text

    # Interactable nodes should be first
    first_non_interactable_idx = None
    for i, n in enumerate(nodes):
        if not n.is_interactable:
            first_non_interactable_idx = i
            break
    if first_non_interactable_idx is not None:
        for n in nodes[:first_non_interactable_idx]:
            assert n.is_interactable, "Interactable nodes should be sorted first"

    ids = DOMSerializer.get_node_ids(nodes)
    assert len(ids) == len(nodes)
    print(f"PASS ✓ ({len(nodes)} nodes, {len(interactable)} interactable)")


def test_browser_environment_mock():
    """Browser environment mock mode returns correct state."""
    print("  [4] Testing browser environment (mock)...", end=" ")
    from environment.playwright_env import BrowserEnvironment, WebAction

    env = BrowserEnvironment(use_mock=True, max_steps=10)

    async def run():
        await env.start()
        state = await env.reset("http://test.com")

        # Check state has all required fields
        assert state.dom_tree, "Missing dom_tree"
        assert state.screenshot is not None, "Missing screenshot"
        assert isinstance(state.viewport_info, dict), "Missing viewport_info"
        assert isinstance(state.action_history, list), "Missing action_history"
        assert state.serialized_dom, "Missing serialized_dom"
        assert len(state.dom_elements) > 0, "No DOM elements"
        assert state.page_title, "Missing page_title"
        assert state.step_number == 0

        # Take action
        a = WebAction(action="CLICK", element_id=1)
        state2 = await env.step(a)
        assert len(state2.action_history) == 1
        assert state2.step_number == 1

        # Test action history text
        history = env.get_action_history_text()
        assert "CLICK" in history

        # TYPE action
        a2 = WebAction(action="TYPE", element_id=3, value="NYC")
        state3 = await env.step(a2)
        assert len(state3.action_history) == 2

        # SCROLL action
        a3 = WebAction(action="SCROLL", direction="down", amount=300)
        state4 = await env.step(a3)
        assert len(state4.action_history) == 3

        await env.close()
        return True

    result = asyncio.run(run())
    assert result
    print("PASS ✓")


def test_prompt_builder():
    """Prompt builder produces correct format."""
    print("  [5] Testing prompt builder...", end=" ")
    from models.prompt_builder import PromptBuilder

    builder = PromptBuilder()

    # Text prompt
    text = builder.build_text_prompt(
        task="Click the search button",
        serialized_dom='<node id=1 tag=button>Search</node>',
        action_history=[],
    )
    assert "[TASK]" in text
    assert "Click the search button" in text
    assert "[DOM]" in text
    assert "Search" in text
    assert "[ACTION_HISTORY]" in text

    # Chat messages
    msgs = builder.build_chat_messages(
        task="Click search",
        serialized_dom='<node id=1 tag=button>Search</node>',
    )
    assert len(msgs) == 2  # system + user
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"

    # Training prompt
    training = builder.build_training_prompt(
        task="Click search",
        serialized_dom='<node id=1 tag=button>Search</node>',
        target_action={"action": "CLICK", "element_id": 1},
    )
    assert "prompt" in training
    assert "target" in training
    assert "full_text" in training
    assert '"action": "CLICK"' in training["target"]

    # Format action target
    target = PromptBuilder.format_action_target("TYPE", element_id=5, value="NYC")
    assert target["action"] == "TYPE"
    assert target["element_id"] == 5
    assert target["value"] == "NYC"
    print("PASS ✓")


def test_action_decoder():
    """Action decoder parses and validates correctly."""
    print("  [6] Testing action decoder...", end=" ")
    from models.action_decoder import ActionDecoder

    decoder = ActionDecoder()

    # Clean JSON
    action = decoder.parse('{"action": "CLICK", "element_id": 32}')
    assert action is not None
    assert action["action"] == "CLICK"
    assert action["element_id"] == 32

    # JSON in markdown fences
    action2 = decoder.parse('```json\n{"action": "TYPE", "element_id": 15, "value": "Brooklyn"}\n```')
    assert action2 is not None
    assert action2["action"] == "TYPE"
    assert action2["value"] == "Brooklyn"

    # JSON embedded in text
    action3 = decoder.parse('I think the best action is {"action": "SCROLL", "direction": "down", "amount": 300}')
    assert action3 is not None
    assert action3["action"] == "SCROLL"

    # Validation
    valid, err = decoder.validate({"action": "CLICK", "element_id": 5}, [1, 5, 10])
    assert valid, f"Should be valid: {err}"

    valid, err = decoder.validate({"action": "CLICK", "element_id": 99}, [1, 5, 10])
    assert not valid, "Should be invalid (bad element_id)"

    valid, err = decoder.validate({"action": "TYPE", "element_id": 5}, [5])
    assert not valid, "TYPE without value should be invalid"

    valid, err = decoder.validate({"action": "DONE"})
    assert valid, "DONE should always be valid"

    # Normalize
    norm = decoder.normalize({"action": "click", "element_id": "32"})
    assert norm["action"] == "CLICK"
    assert norm["element_id"] == 32

    # Parse and validate
    action, is_valid, error = decoder.parse_and_validate(
        '{"action": "SELECT", "element_id": 8, "value": "Economy"}',
        valid_node_ids=[1, 5, 8, 20],
    )
    assert is_valid, f"Should be valid: {error}"
    assert action["value"] == "Economy"
    print("PASS ✓")


def test_uncertainty():
    """Token uncertainty estimates correctly."""
    print("  [7] Testing uncertainty...", end=" ")
    from models.uncertainty import TokenUncertainty

    uc = TokenUncertainty(min_log_prob=-2.0, beam_width=3)

    # High confidence (should NOT regenerate)
    result = uc.assess({"avg_log_prob": -0.5, "text": '{"action": "CLICK"}'})
    assert not result.should_regenerate, "Should not regenerate on confident output"

    # Low confidence (should regenerate)
    result = uc.assess({"avg_log_prob": -3.0, "text": '{"action": "CLICK"}'})
    assert result.should_regenerate, "Should regenerate on low confidence"

    # Beam agreement
    beams = [
        {"text": '{"action": "CLICK", "element_id": 5}', "score": -0.3},
        {"text": '{"action": "CLICK", "element_id": 5}', "score": -0.5},
        {"text": '{"action": "CLICK", "element_id": 5}', "score": -0.8},
    ]
    result = uc.assess_beams(beams)
    assert result.beam_agreement == 1.0, "All beams agree"
    assert not result.should_regenerate

    # Beam disagreement
    beams_disagree = [
        {"text": '{"action": "CLICK", "element_id": 5}', "score": -0.3},
        {"text": '{"action": "TYPE", "element_id": 10, "value": "x"}', "score": -1.5},
        {"text": '{"action": "SCROLL", "direction": "down"}', "score": -2.0},
    ]
    result = uc.assess_beams(beams_disagree)
    assert result.beam_agreement < 1.0, "Beams should disagree"

    # Calibration
    lps = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -0.3, -0.1]
    threshold = uc.calibrate(lps, percentile=10.0)
    assert threshold < 0, f"Threshold should be negative: {threshold}"

    # Statistics
    stats = uc.statistics(lps)
    assert "mean" in stats
    assert "regenerate_fraction" in stats
    print("PASS ✓")


def test_failure_detector():
    """Failure detector works correctly."""
    print("  [8] Testing failure detector...", end=" ")
    from memory.failure_detector import FailureDetector, FailureType

    fd = FailureDetector(loop_window=3, stale_threshold=3, max_steps=5)

    # Normal transition
    ft = fd.detect(None, {"url": "a.com", "html_snippet": "hello"}, {"type": "CLICK"})
    assert ft == FailureType.NONE

    # Error page
    ft = fd.detect(
        {"url": "a.com"}, {"url": "a.com/404", "page_title": "Page Not Found", "html_snippet": ""},
        {"type": "CLICK"}
    )
    assert ft == FailureType.ERROR_PAGE

    fd.reset()
    print("PASS ✓")


def test_data_loader_imports():
    """Data loader imports and instantiates."""
    print("  [9] Testing data loader...", end=" ")
    from data.mind2web_loader import Mind2WebLoader, Mind2WebSample, Mind2WebCollator

    loader = Mind2WebLoader(max_dom_nodes=500)
    assert loader.dataset_name == "osunlp/Multimodal-Mind2Web"

    sample = Mind2WebSample(
        task="Click search",
        serialized_dom="<node id=1 tag=button>Search</node>",
        action={"action": "CLICK", "element_id": 1},
    )
    assert sample.task == "Click search"
    print("PASS ✓")


def test_old_files_removed():
    """Old architecture files have been deleted."""
    print("  [10] Testing old files removed...", end=" ")
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_files = [
        "models/encoders.py",
        "models/grounder.py",
        "models/graph_dom_encoder.py",
        "models/planner.py",
        "models/qwen2vl_llm.py",
        "data/precompute_embeddings.py",
        "memory/long_term.py",
        "memory/short_term.py",
    ]
    for f in old_files:
        path = os.path.join(base, f)
        assert not os.path.exists(path), f"Old file still exists: {f}"
    print("PASS ✓")


def test_new_files_exist():
    """New architecture files exist."""
    print("  [11] Testing new files exist...", end=" ")
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_files = [
        "models/vla_model.py",
        "models/prompt_builder.py",
        "models/action_decoder.py",
        "models/uncertainty.py",
        "environment/dom_serializer.py",
        "environment/playwright_env.py",
        "data/mind2web_loader.py",
        "training/train_supervised.py",
        "evaluation/evaluate.py",
        "inference/run_agent.py",
    ]
    for f in new_files:
        path = os.path.join(base, f)
        assert os.path.exists(path), f"New file missing: {f}"
    print("PASS ✓")


def main():
    print("=" * 60)
    print("  VLA Web Agent — Smoke Tests (New Architecture)")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        test_imports,
        test_config,
        test_dom_serializer,
        test_browser_environment_mock,
        test_prompt_builder,
        test_action_decoder,
        test_uncertainty,
        test_failure_detector,
        test_data_loader_imports,
        test_old_files_removed,
        test_new_files_exist,
    ]

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL ✗ — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
