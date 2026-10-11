"""Save the run from Atlas to results/snapshot.json, so the console can replay it offline.

Usage: python scripts/snapshot.py            (main run plus the no-memory ablation)
Then:  python -m backfill.server --replay
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill.config import SETTINGS  # noqa: E402
from backfill.store import COLLECTIONS, get_store  # noqa: E402

DROP = {"embedding", "code_embedding"}


def dump(db: str) -> dict:
    store = get_store(db)
    out = {}
    for c in COLLECTIONS:
        docs = store.find(c)
        out[c] = [{k: v for k, v in d.items() if k not in DROP} for d in docs]
    return out


def main():
    snap = {"created_at": time.time(), "databases": {}}
    for db in (SETTINGS.mongodb_db, "backfill_ablation"):
        data = dump(db)
        if data["runs"] or data["harness_versions"]:
            snap["databases"][db] = data
            print(f"{db}: {len(data['harness_versions'])} harness versions, {len(data['runs'])} runs, "
                  f"{len(data['lessons'])} lessons")
    out = Path(__file__).resolve().parent.parent / "results" / "snapshot.json"
    out.write_text(json.dumps(snap, default=str))
    print(f"Saved {out} ({out.stat().st_size // 1024} KB). Replay with: python -m backfill.server --replay")
    # The dashboard's offline fallback (also what Vercel serves if Atlas is unreachable)
    from backfill.demo_data import build
    site = Path(__file__).resolve().parent.parent / "dashboard" / "public" / "data.json"
    site.write_text(json.dumps(build(get_store(SETTINGS.mongodb_db), get_store("backfill_ablation"), source="recorded"),
                               default=str))
    print(f"Saved {site} for the dashboard fallback.")


if __name__ == "__main__":
    main()
