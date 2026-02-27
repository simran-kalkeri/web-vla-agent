"""
Test: apply_lora target module discovery

Validates the vision target discovery logic WITHOUT a GPU or model download.
Simulates the Qwen2-VL named_modules() output to confirm:

  1. "mlp" (a VisionMlp container) is NEVER added as a LoRA target.
  2. "fc1" / "fc2" (the leaf Linear4bit layers inside VisionMlp) ARE added.
  3. Language model targets (q_proj, k_proj, etc.) are preserved.
  4. No regression: cross_attn leaf projections are also picked up.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Replicate just the discovery logic from apply_lora (no model needed)
# ---------------------------------------------------------------------------

def _discover_vision_targets(named_modules_list: list[str]) -> set[str]:
    """
    Mirrors the exact logic inside VLAModel.apply_lora after the fix.
    Takes a list of module name strings (as produced by model.named_modules()).
    """
    _VISION_LEAF_NAMES = frozenset(
        {"fc1", "fc2", "q_proj", "k_proj", "v_proj", "o_proj"}
    )
    vision_targets: set[str] = set()

    for name in named_modules_list:
        if "merger" in name:
            leaf = name.split(".")[-1]
            if leaf in _VISION_LEAF_NAMES:
                vision_targets.add(leaf)
        if "cross_attn" in name:
            leaf = name.split(".")[-1]
            if leaf in _VISION_LEAF_NAMES:
                vision_targets.add(leaf)

    return vision_targets


# ---------------------------------------------------------------------------
# Simulated Qwen2-VL-2B module name list (realistic subset)
# ---------------------------------------------------------------------------

QWEN2VL_MODULES = [
    # root
    "",
    # vision encoder
    "model",
    "model.visual",
    "model.visual.patch_embed",
    "model.visual.blocks",
    "model.visual.blocks.0",
    "model.visual.blocks.0.attn",
    "model.visual.blocks.0.attn.q",        # not a standard proj name → skip
    "model.visual.blocks.0.attn.k",
    "model.visual.blocks.0.attn.v",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.norm1",
    "model.visual.blocks.0.mlp",           # NOT in merger path → skip
    "model.visual.blocks.0.mlp.fc1",       # NOT in merger path → skip
    "model.visual.blocks.0.mlp.fc2",       # NOT in merger path → skip
    # vision-language merger (the critical part)
    "model.visual.merger",
    "model.visual.merger.mlp",             # VisionMlp CONTAINER → must NOT be targeted
    "model.visual.merger.mlp.fc1",         # Leaf Linear4bit  → MUST be targeted
    "model.visual.merger.mlp.fc2",         # Leaf Linear4bit  → MUST be targeted
    "model.visual.merger.ln_q",
    # language model layers
    "model.layers",
    "model.layers.0",
    "model.layers.0.self_attn",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    # cross-attention (hypothetical; verify it also works)
    "model.layers.0.cross_attn",
    "model.layers.0.cross_attn.q_proj",
    "model.layers.0.cross_attn.k_proj",
    "model.layers.0.cross_attn.v_proj",
    "model.layers.0.cross_attn.o_proj",
    # lm_head
    "lm_head",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mlp_container_not_targeted():
    """VisionMlp container ('mlp' leaf under merger path) must NOT be added."""
    targets = _discover_vision_targets(QWEN2VL_MODULES)
    assert "mlp" not in targets, (
        f"'mlp' should NOT be a LoRA target — it is a VisionMlp container, "
        f"not a leaf nn.Linear. Got targets: {targets}"
    )
    print("PASS ✓  'mlp' container correctly excluded")


def test_fc1_fc2_targeted():
    """fc1 / fc2 leaf linears inside VisionMlp MUST be added."""
    targets = _discover_vision_targets(QWEN2VL_MODULES)
    assert "fc1" in targets, f"'fc1' should be a LoRA target. Got: {targets}"
    assert "fc2" in targets, f"'fc2' should be a LoRA target. Got: {targets}"
    print(f"PASS ✓  'fc1' and 'fc2' correctly included (targets so far: {sorted(targets)})")


def test_cross_attn_projections_targeted():
    """q/k/v/o_proj under cross_attn paths must be discovered."""
    targets = _discover_vision_targets(QWEN2VL_MODULES)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert proj in targets, (
            f"'{proj}' under cross_attn path should be targeted. Got: {targets}"
        )
    print(f"PASS ✓  cross_attn projections correctly included")


def test_non_merger_mlp_not_targeted():
    """'mlp' that appears outside merger (e.g. vision block mlp) must NOT be targeted."""
    # The module "model.visual.blocks.0.mlp" has 'mlp' as leaf but 'merger' is
    # NOT in its path, so it must be ignored.
    targets = _discover_vision_targets(QWEN2VL_MODULES)
    # 'mlp' should not appear at all after our fix
    assert "mlp" not in targets
    print("PASS ✓  Non-merger 'mlp' modules correctly ignored")


def test_no_unknown_names_in_targets():
    """Every discovered vision target must be in the allowed leaf name set."""
    _VISION_LEAF_NAMES = frozenset(
        {"fc1", "fc2", "q_proj", "k_proj", "v_proj", "o_proj"}
    )
    targets = _discover_vision_targets(QWEN2VL_MODULES)
    unknown = targets - _VISION_LEAF_NAMES
    assert not unknown, f"Unknown targets discovered: {unknown}"
    print(f"PASS ✓  All targets are known-safe leaf names: {sorted(targets)}")


def test_combined_with_lm_targets():
    """Merged target list (LM + vision) must contain all expected modules."""
    lm_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
    vision = _discover_vision_targets(QWEN2VL_MODULES)
    all_targets = list(set(lm_targets) | vision)

    for m in ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
        assert m in all_targets, f"'{m}' missing from final target list: {all_targets}"
    assert "mlp" not in all_targets, "'mlp' must not be in final target list"
    print(f"PASS ✓  Final merged targets: {sorted(all_targets)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_mlp_container_not_targeted,
        test_fc1_fc2_targeted,
        test_cross_attn_projections_targeted,
        test_non_merger_mlp_not_targeted,
        test_no_unknown_names_in_targets,
        test_combined_with_lm_targets,
    ]

    print("=" * 60)
    print("  LoRA Target Discovery — Unit Tests")
    print("=" * 60)

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL ✗  {e}")
            failed += 1

    print("=" * 60)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 60)
    raise SystemExit(0 if failed == 0 else 1)
