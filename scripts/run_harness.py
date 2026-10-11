"""The long run: the harness improves itself epoch after epoch.

Usage:
  python scripts/run_harness.py --epochs 10 --gens 3
  python scripts/run_harness.py --epochs 10 --gens 3 --no-memory --db backfill_ablation

--no-memory switches off and locks lesson recall and duplicate checks (Atlas Vector
Search), for the ablation that shows what Atlas memory is worth.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gens", type=int, default=3, help="inner generations per harness version")
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = no limit)")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--db", default=None, help="MongoDB database name (overrides MONGODB_DB)")
    ap.add_argument("--fresh", action="store_true", help="clear runs, lessons and harness versions in this database first")
    a = ap.parse_args()
    if a.db:
        os.environ["MONGODB_DB"] = a.db

    from backfill import observe
    from backfill.config import SETTINGS
    from backfill.harness import DEFAULT_SPEC, create_root, latest_active
    from backfill.meta import MEMORY_KEYS, run_epoch
    from backfill.store import get_store

    if not SETTINGS.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set. Add it to .env in the repo root.")
    store = get_store()
    if a.fresh:
        if store.backend == "atlas":
            for c in ("harness_versions", "runs", "lessons", "ledger"):
                store.db[c].delete_many({})
        else:
            store.clear()
        print("Cleared runs, lessons and harness versions. Workload fingerprints are kept.")
    h = latest_active(store)
    locked = None
    if a.no_memory:
        spec = {**DEFAULT_SPEC, "tools": {**DEFAULT_SPEC["tools"], "duplicate_check": False, "lesson_recall": False},
                "context_policy": {**DEFAULT_SPEC["context_policy"], "recall_lessons": 0}}
        locked = MEMORY_KEYS
        h = h or create_root(store, spec, note="Initial harness, memory off (ablation)")
    h = h or create_root(store)
    print(f"Store: {store.backend} ({SETTINGS.mongodb_db}). Raindrop: {'on' if observe.ENABLED else 'off'}. "
          f"Model: {SETTINGS.openai_model}. Start: harness v{h['version']}. Memory: {'off' if a.no_memory else 'on'}.")
    t0 = time.time()
    try:
        for e in range(1, a.epochs + 1):
            if a.hours and time.time() - t0 > a.hours * 3600:
                print("Time budget reached.")
                break
            h = run_epoch(h, e, a.gens, locked=locked, store=store)
            observe.flush()
    finally:
        observe.shutdown()
    print(f"Done. Active harness: v{h['version']}.")


if __name__ == "__main__":
    main()
