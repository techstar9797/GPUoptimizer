"""Run generations of the policy-writing agent against the current harness.

Usage: python scripts/evolve.py --generations 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill import observe  # noqa: E402
from backfill.agent import run_generation  # noqa: E402
from backfill.config import SETTINGS  # noqa: E402
from backfill.harness import create_root, latest_active  # noqa: E402
from backfill.store import get_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not SETTINGS.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set. Add it to .env in the repo root.")
    store = get_store()
    h = latest_active(store) or create_root(store)
    start = len(store.find("runs", {"kind": "candidate"}))
    print(f"Store: {store.backend}. Raindrop: {'on' if observe.ENABLED else 'off'}. "
          f"Model: {SETTINGS.openai_model}. Harness v{h['version']}.")
    try:
        for g in range(start + 1, start + a.generations + 1):
            run_generation(h, g, seed=a.seed, store=store)
    finally:
        observe.shutdown()


if __name__ == "__main__":
    main()
