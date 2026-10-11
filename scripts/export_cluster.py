"""Record cluster fill-ups for the dashboard's Cluster tab (played back in the browser).

Simulates the default production workload with FGD, GPU packing, best-fit and the
agent's best policy from Atlas, saving the state of every GPU at each 2% of demand.

Usage: python scripts/export_cluster.py [--workload default] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill.policies.baselines import BASELINES  # noqa: E402
from backfill.sandbox import load_policy  # noqa: E402
from backfill.server import GROUPS, NODE_IDX, NODES, SLOT_IDX  # noqa: E402
from backfill.sim import simulate  # noqa: E402
from backfill.trace import load_pods  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "dashboard" / "public" / "cluster"
LEVELS = 20  # free capacity per GPU, quantized to 5% steps


def encode(gpu_free) -> str:
    """One character per GPU in map order: 'A' = fully used ... 'U' = idle."""
    flat = gpu_free[NODE_IDX, SLOT_IDX]
    q = (flat.clip(0, 1000) * LEVELS + 500) // 1000
    return "".join(chr(65 + int(v)) for v in q)


def record(name: str, label: str, policy, workload: str, seed: int) -> dict:
    frames = []

    def on_frame(cl, stats):
        frames.append({"p": round(stats["progress"], 3), "alloc": round(stats["alloc"], 1),
                       "failed": round(stats["failed_gpus"], 1), "g": encode(cl.gpu_free)})

    r = simulate(policy, NODES, load_pods(workload), seed=seed, name=name, on_frame=on_frame, frame_every=0.02)
    print(f"  {label:28s} {r.unalloc_gpus:7.1f} GPUs idle at full demand ({len(frames)} frames)")
    return {"id": name, "label": label, "workload": workload, "seed": seed, "unalloc": round(r.unalloc_gpus, 1),
            "failed_pods": r.failed_pods, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-agent", action="store_true", help="skip reading the agent's best policy from Atlas")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    policies = [("fgd", "FGD (USENIX ATC '23)", BASELINES["fgd"]),
                ("gpu_packing", "GPU packing", BASELINES["gpu_packing"]),
                ("best_fit", "Best-fit", BASELINES["best_fit"])]
    if not a.no_agent:
        try:
            from backfill.store import get_store
            best = get_store().find("runs", {"ok": True, "kind": "candidate"}, sort=[("mean_unalloc", 1)], limit=1)
            if best:
                b = best[0]
                policies.insert(0, ("agent", f"Agent, gen {b['generation']} (harness v{b['harness_version']})",
                                    load_policy(b["code"])))
        except SystemExit as e:
            print(f"Skipping the agent's policy: {e}")
    print(f"Recording {a.workload}, seed {a.seed}:")
    index = []
    for pid, label, fn in policies:
        rec = record(pid, label, fn, a.workload, a.seed)
        (OUT / f"{pid}.json").write_text(json.dumps(rec, separators=(",", ":")))
        index.append({k: rec[k] for k in ("id", "label", "unalloc", "failed_pods", "workload", "seed")})
    (OUT / "index.json").write_text(json.dumps({"groups": GROUPS, "levels": LEVELS, "policies": index}))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
