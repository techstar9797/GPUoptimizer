"""Route an unseen workload to what the harness already learned (Atlas Vector Search).

The workload's 12-number fingerprint is matched against the training workloads'
fingerprints with $vectorSearch. The best policy found for the nearest workload is
then scored on the new one and compared with FGD.

Usage: python scripts/route.py gpushare80
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    from backfill import observe
    from backfill.config import SETTINGS
    from backfill.fingerprint import FEATURES, fingerprint
    from backfill.policies.baselines import fgd
    from backfill.sandbox import evaluate
    from backfill.sim import simulate
    from backfill.store import get_store
    from backfill.trace import list_workloads, load_nodes, load_pods

    if a.workload not in list_workloads():
        sys.exit(f"Unknown workload '{a.workload}'. Choose from: {', '.join(list_workloads())}")
    store = get_store()
    fp = fingerprint(load_pods(a.workload))
    gen = observe.Generation(event="workload_routing", harness_version=0, harness_id="router",
                             generation=0, workloads=[a.workload], model=SETTINGS.openai_model)

    with gen.tool("atlas_vector_search.workloads", input={"workload": a.workload, "k": 3}) as t:
        hits = store.vector_search("workloads", fp, k=3, filter={"split": "train"})
        t["output"] = [{"workload": h["name"], "similarity": round(h["similarity"], 4)} for h in hits]
    if not hits:
        sys.exit("No training workloads in the store. Run scripts/setup_atlas.py first.")
    print(f"Fingerprint of {a.workload}:")
    for name, v in zip(FEATURES, fp):
        print(f"  {name:22s} {v:.3f}")
    print("Nearest training workloads:")
    for h in hits:
        print(f"  {h['name']:12s} similarity {h['similarity']:.4f}")
    near = hits[0]["name"]

    runs = [r for r in store.find("runs", {"ok": True, "kind": "candidate"}) if near in (r.get("per_workload") or {})]
    if not runs:
        sys.exit(f"No working policies have been scored on {near} yet. Run the harness first.")
    best = min(runs, key=lambda r: r["per_workload"][near])
    print(f"Best policy on {near}: generation {best['generation']} (harness v{best['harness_version']}), "
          f"{best['per_workload'][near]:.0f} unallocated there.")

    with gen.tool("simulator.evaluate", input={"run": best["_id"], "workload": a.workload}) as t:
        ev = evaluate(best["code"], [a.workload], seed=a.seed)
        t["output"] = {"ok": ev.ok, "per_workload": ev.per_workload}
    with gen.tool("simulator.baseline", input={"policy": "fgd", "workload": a.workload}) as t:
        ref = simulate(fgd, load_nodes(), load_pods(a.workload), seed=a.seed, name="fgd")
        t["output"] = {"unalloc_gpus": ref.unalloc_gpus}
    if not ev.ok:
        gen.signal("policy_crashed", False)
        gen.finish(f"Routed policy failed on {a.workload}: {ev.error}")
        observe.shutdown()
        sys.exit(f"The routed policy failed on {a.workload}: {ev.error}")
    ours = ev.per_workload[a.workload]
    print(f"On {a.workload} (seed {a.seed}): routed policy {ours:.0f} unallocated GPUs, FGD {ref.unalloc_gpus:.0f}.")
    gen.signal("beat_fgd" if ours < ref.unalloc_gpus else "behind_fgd", ours < ref.unalloc_gpus,
               ours=round(ours, 1), fgd=round(ref.unalloc_gpus, 1), routed_from=near)
    store.update("workloads", a.workload, {"routed_from": near, "routed_similarity": hits[0]["similarity"],
                                          "routed_run": best["_id"], "routed_unalloc": ours,
                                          "fgd_unalloc": ref.unalloc_gpus, "routed_at": time.time()})
    gen.finish(f"{a.workload} routed from {near}: {ours:.0f} vs FGD {ref.unalloc_gpus:.0f}")
    observe.shutdown()


if __name__ == "__main__":
    main()
