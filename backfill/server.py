"""Backfill console: HTTP API plus the web UI.

Run: python -m backfill.server  (then open http://localhost:8000)
"""
from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import observe
from .agent import baseline_scores
from .config import HELDOUT_WORKLOADS, SETTINGS, TRAIN_WORKLOADS
from .policies.baselines import BASELINES
from .sandbox import load_policy
from .sim import simulate
from .store import get_store
from .trace import GPU_MILLI, list_workloads, load_nodes, load_pods

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
RESULTS = ROOT / "results"

NODES = load_nodes()

POLICY_LABELS = {
    "random": "Random",
    "dot_product": "Dot-product",
    "gpu_packing": "GPU packing",
    "best_fit": "Best-fit",
    "fgd": "FGD (ATC '23)",
}


def _layout():
    """Node order for the cluster map: largest GPU models first, then biggest nodes first."""
    totals: dict = {}
    for m, g in zip(NODES.model, NODES.n_gpu):
        totals[m] = totals.get(m, 0) + int(g)
    order = sorted(range(len(NODES.names)), key=lambda i: (-totals[NODES.model[i]], NODES.model[i], -NODES.n_gpu[i], i))
    node_idx, slot_idx = [], []
    for i in order:
        for s in range(NODES.n_gpu[i]):
            node_idx.append(i)
            slot_idx.append(s)
    groups, cur = [], None
    for i in order:
        if cur is None or cur["model"] != NODES.model[i]:
            cur = {"model": str(NODES.model[i]), "nodes": []}
            groups.append(cur)
        cur["nodes"].append(int(NODES.n_gpu[i]))
    return np.array(node_idx), np.array(slot_idx), groups


NODE_IDX, SLOT_IDX, GROUPS = _layout()


def _gpu_free_flat(gpu_free: np.ndarray) -> list[int]:
    return gpu_free[NODE_IDX, SLOT_IDX].astype(int).tolist()


def _agent_policies(limit: int = 3) -> list[dict]:
    try:
        best = get_store().find("runs", {"ok": True, "kind": "candidate"}, sort=[("mean_unalloc", 1)], limit=limit)
    except Exception:
        return []
    return [{"id": r["_id"], "label": f"Agent: gen {r['generation']} (harness v{r['harness_version']})"} for r in best]


async def meta(_: Request):
    return JSONResponse({
        "nodes": len(NODES.names),
        "gpus": int(NODES.n_gpu.sum()),
        "models": len(GROUPS),
        "groups": GROUPS,
        "workloads": list_workloads(),
        "policies": [{"id": k, "label": POLICY_LABELS.get(k, k)} for k in BASELINES] + _agent_policies(),
    })


async def baselines(_: Request):
    f = RESULTS / "baselines.json"
    if not f.exists():
        return JSONResponse({"rows": []})
    return JSONResponse({"rows": json.loads(f.read_text()), "labels": POLICY_LABELS})


async def run_stream(request: Request):
    """Server-sent events: one frame per 2% of arrived demand, then a final result."""
    q = request.query_params
    policy, workload = q.get("policy", "fgd"), q.get("workload", "default")
    seed = int(q.get("seed", "0"))
    if policy.startswith("run_"):
        doc = get_store().get("runs", policy)
        if not doc or not doc.get("ok"):
            return JSONResponse({"error": f"No working agent policy '{policy}'."}, status_code=400)
        policy_fn = load_policy(doc["code"])
    elif policy in BASELINES:
        policy_fn = BASELINES[policy]
    else:
        return JSONResponse({"error": f"Unknown policy '{policy}'."}, status_code=400)
    if workload not in list_workloads():
        return JSONResponse({"error": f"Unknown workload '{workload}'."}, status_code=400)

    out: queue.Queue = queue.Queue(maxsize=256)

    def on_frame(cl, stats):
        out.put({"type": "frame", **stats, "gpu_free": _gpu_free_flat(cl.gpu_free)})

    def worker():
        try:
            r = simulate(policy_fn, NODES, load_pods(workload), seed=seed,
                         name=policy, on_frame=on_frame, frame_every=0.02)
            out.put({"type": "done", **r.row(), "capacity_gpus": r.capacity_gpus})
        except Exception as e:  # surfaced to the UI as an error event
            out.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        out.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def events():
        while (item := out.get()) is not None:
            yield f"data: {json.dumps(item, separators=(',', ':'))}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------- Harness: read from the store (Atlas, or the local fallback) ----------
def _clean(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in ("code_embedding", "embedding", "curves")}


async def harness(_: Request):
    store = get_store()
    versions = [_clean(v) for v in store.find("harness_versions", sort=[("version", 1)])]
    runs = [_clean(r) for r in store.find("runs", {"kind": "candidate"}, sort=[("generation", 1)])]
    lessons = {l["_id"]: {"text": l["text"], "family": l["family"], "generation": l.get("generation")}
               for l in store.find("lessons", projection={"embedding": 0})}
    ledger = store.find("ledger")
    base = baseline_scores()
    ref = {p: sum(base[p][w] for w in TRAIN_WORKLOADS) / len(TRAIN_WORKLOADS)
           for p in ("fgd", "gpu_packing", "best_fit") if p in base and all(w in base[p] for w in TRAIN_WORKLOADS)}
    return JSONResponse({
        "backend": store.backend,
        "database": SETTINGS.mongodb_db if store.backend == "atlas" else None,
        "raindrop": observe.ENABLED,
        "model": SETTINGS.openai_model,
        "train": TRAIN_WORKLOADS, "heldout": HELDOUT_WORKLOADS,
        "reference": ref,
        "versions": versions, "runs": runs, "lessons": lessons,
        "tokens": sum(l.get("tokens") or 0 for l in ledger),
    })


ABLATION_DB = os.environ.get("BACKFILL_ABLATION_DB", "backfill_ablation")


def _best_curve(store) -> dict:
    runs = sorted(store.find("runs", {"kind": "candidate"}, projection={"code_embedding": 0, "curves": 0, "code": 0}),
                  key=lambda r: r.get("generation") or 0)
    best, pts = None, []
    for r in runs:
        if r.get("ok") and r.get("mean_unalloc") is not None:
            best = r["mean_unalloc"] if best is None else min(best, r["mean_unalloc"])
        pts.append({"g": r["generation"], "best": best, "score": r.get("mean_unalloc") if r.get("ok") else None})
    tokens = sum(l.get("tokens") or 0 for l in store.find("ledger"))
    return {"points": pts, "best": best, "generations": len(runs),
            "failed": sum(1 for r in runs if not r.get("ok")),
            "duplicates": sum(1 for r in runs if r.get("duplicate_of")), "tokens": tokens}


async def ablation(_: Request):
    """Best-so-far curves with Atlas memory on (main database) and off (ablation database)."""
    out = {"memory_on": _best_curve(get_store())}
    try:
        off = get_store(ABLATION_DB)
        out["memory_off"] = _best_curve(off) if off.find("runs", limit=1) else None
    except Exception as e:
        out["memory_off"], out["error"] = None, str(e)
    base = baseline_scores()
    if "fgd" in base and all(w in base["fgd"] for w in TRAIN_WORKLOADS):
        out["fgd"] = sum(base["fgd"][w] for w in TRAIN_WORKLOADS) / len(TRAIN_WORKLOADS)
    for k in ("memory_on", "memory_off"):
        c = out.get(k)
        if c and out.get("fgd") is not None:
            c["beat_fgd_at"] = next((p["g"] for p in c["points"] if p["best"] is not None and p["best"] < out["fgd"]), None)
    return JSONResponse(out)


async def events(request: Request):
    """Push a 'changed' event when harness data changes: Atlas change streams, or polling locally."""
    store = get_store()

    def atlas_stream():
        pipeline = [{"$match": {"ns.coll": {"$in": ["runs", "harness_versions", "lessons"]}}}]
        with store.db.watch(pipeline, max_await_time_ms=15000) as stream:
            yield "data: {\"type\":\"ready\",\"via\":\"atlas-change-stream\"}\n\n"
            while True:
                change = stream.try_next()
                if change is None:
                    yield ": keepalive\n\n"
                    continue
                coll = change["ns"]["coll"]
                yield f"data: {json.dumps({'type': 'changed', 'collection': coll, 'op': change['operationType']})}\n\n"

    def local_poll():
        import time as _t
        yield "data: {\"type\":\"ready\",\"via\":\"polling\"}\n\n"
        last = None
        while True:
            sig = tuple(len(store._load(c)) for c in ("runs", "harness_versions", "lessons"))
            if last is not None and sig != last:
                yield "data: {\"type\":\"changed\"}\n\n"
            last = sig
            _t.sleep(2)
            yield ": keepalive\n\n"

    gen = atlas_stream() if store.backend == "atlas" else local_poll()
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def index(_: Request):
    return FileResponse(WEB / "index.html")


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/baselines", baselines),
    Route("/api/run", run_stream),
    Route("/api/harness", harness),
    Route("/api/events", events),
    Route("/api/ablation", ablation),
    Mount("/static", StaticFiles(directory=WEB), name="static"),
])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="warning")
