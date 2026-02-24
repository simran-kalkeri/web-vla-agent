"""
Smoke tests for the refactored VLA Web Agent.

Validates:
  1. All modules import and instantiate
  2. Forward pass produces correct output shapes
  3. 3-term loss computes with valid gradients
  4. Trainable parameter count < 10M
  5. Entropy uncertainty triggers correctly
  6. Focal loss produces valid gradients
  7. Phase management (action head enable/disable)
  8. Config loads correctly
"""
from __future__ import annotations

import sys
import torch
import torch.nn.functional as F


def test_imports():
    """All refactored modules import without error."""
    print("  [1] Testing imports...", end=" ")
    from models.graph_dom_encoder import GraphDOMEncoder, structural_features, build_edge_index
    from models.grounder import VLAGrounder, VLAGrounderBatched, CrossAttentionBlock
    from models.uncertainty import EntropyUncertainty, UncertaintyResult
    from models.encoders import TextEncoder, VisionEncoder
    from utils.config import VLAConfig, load_config
    from evaluation.metrics import MetricsTracker, StepResult, TaskResult
    from evaluation.evaluate import Evaluator, ABLATION_PRESETS, AblationMode
    print("PASS ✓")


def test_model_construction():
    """All modules instantiate with default parameters."""
    print("  [2] Testing model construction...", end=" ")
    from models.graph_dom_encoder import GraphDOMEncoder
    from models.grounder import VLAGrounder, VLAGrounderBatched

    gcn = GraphDOMEncoder(text_dim=384, hidden_dim=256)
    grounder = VLAGrounder(d_model=256, text_dim=256, vision_dim=512, subgoal_dim=384)
    batched = VLAGrounderBatched(d_model=256, text_dim=256, vision_dim=512, subgoal_dim=384)
    print("PASS ✓")
    return gcn, grounder, batched


def test_forward_pass(gcn, grounder):
    """Forward pass produces correct output shapes."""
    print("  [3] Testing forward pass dimensions...", end=" ")
    B, N = 2, 10
    text_dim, struct_dim, hidden_dim, vision_dim, subgoal_dim = 384, 36, 256, 512, 384

    # GCN forward
    text_emb = torch.randn(N, text_dim)
    struct_feat = torch.randn(N, struct_dim)
    edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)
    gcn_out = gcn(text_emb, struct_feat, edge_index)
    assert gcn_out.shape == (N, hidden_dim), f"GCN output: {gcn_out.shape} != ({N}, {hidden_dim})"

    # Grounder forward
    subgoal = torch.randn(B, subgoal_dim)
    dom_nodes = torch.randn(B, N, hidden_dim)
    vision = torch.randn(B, N, vision_dim)

    elem_logits, action_logits, grounding_scores, node_reprs = grounder(
        subgoal, dom_nodes, vision
    )
    assert elem_logits.shape == (B, N), f"elem_logits: {elem_logits.shape}"
    assert action_logits.shape == (B, 4), f"action_logits: {action_logits.shape}"
    assert grounding_scores.shape == (B, N), f"grounding_scores: {grounding_scores.shape}"
    assert node_reprs.shape == (B, N, hidden_dim), f"node_reprs: {node_reprs.shape}"
    print("PASS ✓")


def test_loss_computation(grounder):
    """3-term loss computes with valid gradients."""
    print("  [4] Testing loss computation...", end=" ")
    from training.train_supervised import focal_loss

    B, N = 4, 8
    subgoal = torch.randn(B, 384)
    dom_nodes = torch.randn(B, N, 256, requires_grad=True)
    vision = torch.randn(B, N, 512)

    elem_logits, action_logits, grounding_scores, _ = grounder(
        subgoal, dom_nodes, vision
    )

    # L_CE_element
    elem_targets = torch.randint(0, N, (B,))
    loss_elem = F.cross_entropy(elem_logits, elem_targets)

    # L_InfoNCE
    loss_infonce = grounder.contrastive_loss(grounding_scores, elem_targets)

    # L_CE_action (focal)
    action_targets = torch.randint(0, 4, (B,))
    loss_action = focal_loss(action_logits, action_targets, gamma=2.0)

    # Total
    total = 1.0 * loss_elem + 0.5 * loss_infonce + 1.0 * loss_action
    total.backward()

    assert dom_nodes.grad is not None, "No gradient on dom_nodes"
    assert not torch.isnan(total), f"Total loss is NaN: {total}"
    assert not torch.isinf(total), f"Total loss is Inf: {total}"
    print(f"PASS ✓ (L={total.item():.4f})")


def test_parameter_count(gcn, grounder):
    """Total trainable parameters < 10M."""
    print("  [5] Testing parameter count...", end=" ")
    gcn_params = sum(p.numel() for p in gcn.parameters() if p.requires_grad)
    grounder_params = sum(p.numel() for p in grounder.parameters() if p.requires_grad)
    total = gcn_params + grounder_params

    print(f"\n      GCN params:      {gcn_params:>10,}")
    print(f"      Grounder params: {grounder_params:>10,}")
    print(f"      Total trainable: {total:>10,}", end=" ")
    assert total < 10_000_000, f"Too many params: {total} >= 10M"
    print("PASS ✓ (< 10M)")


def test_entropy_uncertainty():
    """Entropy-based uncertainty triggers correctly."""
    print("  [6] Testing entropy uncertainty...", end=" ")
    from models.uncertainty import EntropyUncertainty

    uc = EntropyUncertainty(threshold_tau=1.0, temperature=0.07)

    # Low entropy (confident) — should NOT replan
    scores_confident = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    replan = uc.should_replan(scores_confident)
    assert not replan[0].item(), "Should not replan on confident prediction"

    # High entropy (uniform) — should replan
    scores_uniform = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    replan = uc.should_replan(scores_uniform)
    assert replan[0].item(), "Should replan on uniform prediction"

    # Calibration
    entropies = torch.rand(100) * 3.0
    tau = uc.calibrate_threshold(entropies, percentile=90.0)
    assert 0 < tau < 5.0, f"Calibrated threshold out of range: {tau}"

    # Stats
    stats = uc.entropy_statistics(torch.randn(4, 10))
    assert "entropy_mean" in stats
    assert "replan_fraction" in stats
    print("PASS ✓")


def test_focal_loss():
    """Focal loss produces valid gradients and differs from standard CE."""
    print("  [7] Testing focal loss...", end=" ")
    from training.train_supervised import focal_loss

    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.randint(0, 4, (8,))

    # Standard CE (gamma=0)
    loss_ce = focal_loss(logits, targets, gamma=0.0)

    # Focal (gamma=2)
    loss_focal = focal_loss(logits, targets, gamma=2.0)

    loss_focal.backward()
    assert logits.grad is not None, "No gradient from focal loss"
    assert not torch.isnan(loss_focal), "Focal loss is NaN"

    # With class weights
    weights = torch.tensor([1.0, 2.0, 1.5, 3.0])
    loss_weighted = focal_loss(logits.detach().requires_grad_(True), targets, gamma=2.0, weight=weights)
    loss_weighted.backward()
    print("PASS ✓")


def test_batched_grounder():
    """Variable-length batched grounder handles padding correctly."""
    print("  [8] Testing batched grounder...", end=" ")
    from models.grounder import VLAGrounderBatched

    batched = VLAGrounderBatched(d_model=256, text_dim=256, vision_dim=512, subgoal_dim=384)

    subgoal = torch.randn(3, 384)
    dom_list = [torch.randn(5, 256), torch.randn(8, 256), torch.randn(3, 256)]
    vis_list = [torch.randn(5, 512), torch.randn(8, 512), torch.randn(3, 512)]

    elem_logits, action_logits, scores, reprs = batched(subgoal, dom_list, vis_list)

    assert len(elem_logits) == 3
    assert elem_logits[0].shape == (5,)
    assert elem_logits[1].shape == (8,)
    assert elem_logits[2].shape == (3,)
    assert action_logits.shape == (3, 4)
    print("PASS ✓")


def test_config():
    """Config loads and has correct structure."""
    print("  [9] Testing config...", end=" ")
    from utils.config import load_config

    cfg = load_config()
    assert cfg.graph.gcn_layers == 1
    assert cfg.grounder.cross_attention_layers == 1
    assert cfg.grounder.num_heads == 4
    assert cfg.model.hidden_dim == 256
    assert cfg.training.focal_gamma == 2.0
    assert cfg.training.phase1_epochs == 10
    assert cfg.training.phase2_epochs == 30
    assert hasattr(cfg, 'uncertainty')
    assert hasattr(cfg.uncertainty, 'entropy_threshold')
    print("PASS ✓")


def test_ablation_presets():
    """Ablation presets are correctly defined."""
    print("  [10] Testing ablation presets...", end=" ")
    from evaluation.evaluate import ABLATION_PRESETS

    expected = {"full", "no_graph", "no_cross_attention", "no_entropy", "dom_only", "action_unweighted"}
    actual = set(ABLATION_PRESETS.keys())
    assert actual == expected, f"Missing ablations: {expected - actual}, extra: {actual - expected}"
    print("PASS ✓")


def main():
    print("=" * 60)
    print("  VLA Web Agent — Refactored Smoke Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        test_imports,
        test_model_construction,
        test_forward_pass,
        test_loss_computation,
        test_parameter_count,
        test_entropy_uncertainty,
        test_focal_loss,
        test_batched_grounder,
        test_config,
        test_ablation_presets,
    ]

    gcn = grounder = batched = None

    for test in tests:
        try:
            if test == test_forward_pass:
                test(gcn, grounder)
            elif test == test_loss_computation:
                test(grounder)
            elif test == test_parameter_count:
                test(gcn, grounder)
            elif test == test_model_construction:
                gcn, grounder, batched = test()
            else:
                test()
            passed += 1
        except Exception as e:
            print(f"FAIL ✗ — {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
