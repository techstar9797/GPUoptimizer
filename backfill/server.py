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

from .policies.baselines import BASELINES
from .sim import simulate
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


async def meta(_: Request):
    return JSONResponse({
        "nodes": len(NODES.names),
        "gpus": int(NODES.n_gpu.sum()),
        "models": len(GROUPS),
        "groups": GROUPS,
        "workloads": list_workloads(),
        "policies": [{"id": k, "label": POLICY_LABELS.get(k, k)} for k in BASELINES],
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
    if policy not in BASELINES:
        return JSONResponse({"error": f"Unknown policy '{policy}'."}, status_code=400)
    if workload not in list_workloads():
        return JSONResponse({"error": f"Unknown workload '{workload}'."}, status_code=400)

    out: queue.Queue = queue.Queue(maxsize=256)

    def on_frame(cl, stats):
        out.put({"type": "frame", **stats, "gpu_free": _gpu_free_flat(cl.gpu_free)})

    def worker():
        try:
            r = simulate(BASELINES[policy], NODES, load_pods(workload), seed=seed,
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


async def index(_: Request):
    return FileResponse(WEB / "index.html")


app = Starlette(routes=[
    Route("/", index),
    Route("/api/meta", meta),
    Route("/api/baselines", baselines),
    Route("/api/run", run_stream),
    Mount("/static", StaticFiles(directory=WEB), name="static"),
])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="warning")
