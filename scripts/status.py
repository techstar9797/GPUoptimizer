"""Where do the runs stand? Summarizes the main and memory-off databases in Atlas.

Usage: python scripts/status.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill.agent import baseline_scores  # noqa: E402
from backfill.config import SETTINGS, TRAIN_WORKLOADS  # noqa: E402
from backfill.store import get_store  # noqa: E402


def ago(t):
    if not t:
        return "never"
    s = int(time.time() - t)
    return f"{s}s ago" if s < 120 else f"{s // 60} min ago"


def summarize(db: str, fgd: float | None):
    st = get_store(db)
    runs = st.find("runs", {"kind": "candidate"}, projection={"code": 0, "code_embedding": 0, "curves": 0})
    vers = st.find("harness_versions", sort=[("version", 1)])
    ok = [r for r in runs if r.get("ok") and not r.get("duplicate_of")]
    last = max((r.get("created_at") or 0 for r in runs), default=0)
    best = min(ok, key=lambda r: r["mean_unalloc"]) if ok else None
    print(f"\n== {db}")
    if not runs and not vers:
        print("   empty: this run has not started")
        return
    print(f"   generations: {len(runs)}  (scored {len(ok)}, repeats {sum(1 for r in runs if r.get('duplicate_of'))}, "
          f"failed {sum(1 for r in runs if not r.get('ok'))})")
    print(f"   harness versions: {len(vers)}  " + ", ".join(f"v{v['version']} {v['status']}" for v in vers[-6:]))
    if best:
        vs = f", FGD {fgd:.1f} ({'beats' if best['mean_unalloc'] < fgd else 'behind'} by {abs(fgd - best['mean_unalloc']):.1f})" if fgd else ""
        print(f"   best: {best['mean_unalloc']:.1f} GPUs idle at gen {best['generation']} (harness v{best['harness_version']}){vs}")
    print(f"   last generation written: {ago(last)}" + ("  <- still running" if last and time.time() - last < 180 else ""))


def main():
    b = baseline_scores(0).get("fgd", {})
    fgd = sum(b[w] for w in TRAIN_WORKLOADS) / len(TRAIN_WORKLOADS) if all(w in b for w in TRAIN_WORKLOADS) else None
    summarize(SETTINGS.mongodb_db, fgd)
    summarize("backfill_ablation", fgd)


if __name__ == "__main__":
    main()
