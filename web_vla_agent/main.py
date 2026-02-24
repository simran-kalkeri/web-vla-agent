"""VLA Web Agent — Main entry point."""
from __future__ import annotations

import sys


def main():
    """Dispatch to appropriate subcommand."""
    print("VLA Web Agent v2.0.0")
    print()
    print("Commands:")
    print("  python -m training.train_supervised  — Train the model")
    print("  python -m evaluation.evaluate        — Evaluate the model")
    print("  python -m inference.run_agent         — Run the agent on a URL")
    print()
    print("Run with PYTHONPATH set to the web_vla_agent directory:")
    print("  export PYTHONPATH=/path/to/web_vla_agent")


if __name__ == "__main__":
    main()
