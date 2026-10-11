"""One compact JSON document with everything the demo dashboard shows.

The same shape is produced by api/data.js on Vercel, and saved to site/data.json
by scripts/snapshot.py as the offline fallback.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import HELDOUT_WORKLOADS, SETTINGS, TRAIN_WORKLOADS

ROOT = Path(__file__).resolve().parent.parent
RUN_FIELDS = ["_id", "generation", "harness_version", "harness_id", "created_at", "ok", "mean_unalloc",
              "per_workload", "heldout", "signals", "recalled_lessons", "duplicate_of", "idea", "error",
              "raindrop_event_id"]
VERSION_FIELDS = ["_id", "version", "parent_id", "status", "rationale", "score", "heldout_score",
                  "created_at", "decided_at", "author"]


def _run(r: dict) -> dict:
    d = {k: r.get(k) for k in RUN_FIELDS}
    d["recalled_lessons"] = len(r.get("recalled_lessons") or [])
    d["idea"] = (r.get("idea") or "").split("\n")[0][:160]
    d["error"] = (r.get("error") or "")[:160] or None
    return d


def _db(store) -> dict:
    runs = store.find("runs", {"kind": "candidate"}, projection={"code": 0, "code_embedding": 0, "curves": 0})
    versions = store.find("harness_versions")
    ledger = store.find("ledger")
    return {
        "runs": sorted((_run(r) for r in runs), key=lambda r: r["created_at"] or 0),
        "versions": sorted(({k: v.get(k) for k in VERSION_FIELDS} | {"rationale": (v.get("rationale") or "")[:400]}
                            for v in versions), key=lambda v: v["version"]),
        "lessons": len(store.find("lessons", projection={"embedding": 0})),
        "tokens": sum(l.get("tokens") or 0 for l in ledger),
    }


def baselines() -> list[dict]:
    rows = []
    for name in ("baselines.json", "baselines_heldout.json"):
        f = ROOT / "results" / name
        if f.exists():
            rows += [{k: r[k] for k in ("policy", "workload", "seed", "unalloc_gpus")} for r in json.loads(f.read_text())]
    return rows


def build(main_store, ablation_store=None, source: str | None = None) -> dict:
    out = {
        "source": source or main_store.backend,
        "generated_at": time.time(),
        "recorded_at": getattr(main_store, "recorded_at", None),
        "model": SETTINGS.openai_model,
        "train": TRAIN_WORKLOADS,
        "heldout": HELDOUT_WORKLOADS,
        "baselines": baselines(),
        "main": _db(main_store),
        "ablation": None,
    }
    if ablation_store is not None:
        try:
            ab = _db(ablation_store)
            out["ablation"] = ab if ab["runs"] else None
        except Exception:
            out["ablation"] = None
    return out
