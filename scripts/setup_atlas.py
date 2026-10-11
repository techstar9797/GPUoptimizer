"""Create Backfill's collections and Atlas Vector Search indexes, and load workload fingerprints.

Usage: python scripts/setup_atlas.py [--reset]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill.config import HELDOUT_WORKLOADS, SETTINGS, TRAIN_WORKLOADS  # noqa: E402
from backfill.fingerprint import FEATURES, fingerprint  # noqa: E402
from backfill.harness import create_root, latest_active  # noqa: E402
from backfill.store import family_of, get_store  # noqa: E402
from backfill.trace import list_workloads, load_pods  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete all documents first")
    a = ap.parse_args()
    missing = SETTINGS.missing()
    if missing:
        print(f"Not set in .env: {', '.join(missing)}")
    store = get_store()
    print(f"Store: {store.backend}" + (f" (database '{SETTINGS.mongodb_db}')" if store.backend == "atlas" else ""))
    if a.reset:
        store.clear()
        print("Cleared all collections.")
    for line in store.setup(wait=True):
        print(" ", line)
    for w in list_workloads():
        split = "train" if w in TRAIN_WORKLOADS else "heldout" if w in HELDOUT_WORKLOADS else "other"
        store.upsert("workloads", w, {"name": w, "family": family_of(w), "split": split,
                                      "fingerprint": fingerprint(load_pods(w)), "features": FEATURES,
                                      "best_harness_id": None})
    print(f"Loaded fingerprints for {len(list_workloads())} workloads.")
    if not latest_active(store):
        h = create_root(store)
        print(f"Created harness v1 ({h['_id']}).")
    print("Ready.")


if __name__ == "__main__":
    main()
