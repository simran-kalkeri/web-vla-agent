"""
Diagnose Mind2Web dataset loading — prints actual data structure.

Run:
    cd /home/hashira/genai/vla/web-vla-agent/web_vla_agent
    python scripts/diagnose_data.py
"""
from __future__ import annotations

import json
import os
import ssl
import sys

# SSL bypass for corporate environments
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def main():
    from datasets import load_dataset

    print("=" * 60)
    print("  Mind2Web Dataset — Format Diagnostic")
    print("=" * 60)

    print("\n[1] Loading dataset (first 5 samples)...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train", streaming=True)

    for i, sample in enumerate(ds):
        if i >= 5:
            break

        print(f"\n{'─' * 60}")
        print(f"  Sample {i}")
        print(f"{'─' * 60}")

        # Print all top-level keys and their types
        print(f"  Keys: {list(sample.keys())}")

        for key in ["confirmed_task", "website", "domain", "annotation_id"]:
            val = sample.get(key, "MISSING")
            print(f"  {key}: {str(val)[:100]}")

        # Operation
        op = sample.get("operation", "MISSING")
        print(f"  operation type: {type(op).__name__}")
        print(f"  operation: {str(op)[:200]}")

        # Positive candidates
        pos = sample.get("pos_candidates", [])
        print(f"\n  pos_candidates type: {type(pos).__name__}")
        print(f"  pos_candidates length: {len(pos) if isinstance(pos, list) else 'N/A'}")
        if isinstance(pos, list) and len(pos) > 0:
            first = pos[0]
            print(f"  pos_candidates[0] type: {type(first).__name__}")
            print(f"  pos_candidates[0] preview: {str(first)[:300]}")
            # If it's a string, try parsing it
            if isinstance(first, str):
                try:
                    parsed = json.loads(first)
                    print(f"  pos_candidates[0] parsed type: {type(parsed).__name__}")
                    if isinstance(parsed, dict):
                        print(f"  pos_candidates[0] parsed keys: {list(parsed.keys())}")
                    elif isinstance(parsed, list):
                        print(f"  pos_candidates[0] parsed length: {len(parsed)}")
                        if len(parsed) > 0:
                            print(f"  pos_candidates[0] parsed[0] type: {type(parsed[0]).__name__}")
                            print(f"  pos_candidates[0] parsed[0]: {str(parsed[0])[:200]}")
                except json.JSONDecodeError:
                    print(f"  pos_candidates[0] is NOT valid JSON")

        # Negative candidates
        neg = sample.get("neg_candidates", [])
        print(f"\n  neg_candidates type: {type(neg).__name__}")
        print(f"  neg_candidates length: {len(neg) if isinstance(neg, list) else 'N/A'}")
        if isinstance(neg, list) and len(neg) > 0:
            first_neg = neg[0]
            print(f"  neg_candidates[0] type: {type(first_neg).__name__}")
            print(f"  neg_candidates[0] preview: {str(first_neg)[:300]}")

        # Screenshot
        ss = sample.get("screenshot")
        print(f"\n  screenshot type: {type(ss).__name__ if ss is not None else 'None'}")
        if hasattr(ss, "size"):
            print(f"  screenshot size: {ss.size}")

        # Other fields
        for key in ["action_reprs", "target_action_index", "target_action_reprs"]:
            val = sample.get(key, "MISSING")
            print(f"  {key}: {str(val)[:200]}")

    # Now test the loader
    print(f"\n{'=' * 60}")
    print("  Testing Mind2WebLoader...")
    print(f"{'=' * 60}")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.mind2web_loader import Mind2WebLoader

    loader = Mind2WebLoader(max_dom_nodes=500)
    samples = loader.build_training_examples(
        split="train", max_samples=10, include_screenshot=False,
    )

    print(f"\n  Valid training samples: {len(samples)} / 10")
    for s in samples[:3]:
        print(f"\n  Sample: {s.sample_id[:30]}...")
        print(f"    Task: {s.task[:80]}")
        print(f"    Candidates: {len(s.candidates)}")
        print(f"    Target idx: {s.target_candidate_index}")
        print(f"    Action: {s.action}")

    if not samples:
        print("\n  ❌ ZERO valid samples — candidate parsing is still broken!")
    else:
        print(f"\n  ✅ {len(samples)} valid samples loaded successfully!")


if __name__ == "__main__":
    main()
