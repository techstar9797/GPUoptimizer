"""Run every baseline policy over workloads and seeds; print a table and save JSON.

Usage: python scripts/run_baselines.py [--workloads default gpushare40 ...] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill.policies.baselines import BASELINES  # noqa: E402
from backfill.sim import simulate  # noqa: E402
from backfill.trace import load_nodes, load_pods  # noqa: E402


def _run(args):
    policy, workload, seed = args
    r = simulate(BASELINES[policy], load_nodes(), load_pods(workload), seed=seed, name=policy)
    return {**r.row(), "curve": r.curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=["default", "gpushare40", "multigpu20", "gpuspec33"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--policies", nargs="+", default=list(BASELINES))
    ap.add_argument("--out", default="results/baselines.json")
    a = ap.parse_args()
    jobs = [(p, w, s) for w in a.workloads for p in a.policies for s in range(a.seeds)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rows = list(ex.map(_run, jobs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    table = df.pivot_table(index="policy", columns="workload", values="unalloc_gpus", aggfunc="mean").round(1)
    table["mean"] = table.mean(axis=1).round(1)
    table = table.sort_values("mean")
    print("Unallocated GPUs when arrived demand = 100% of capacity (lower is better), mean over seeds")
    print(table.to_string())


if __name__ == "__main__":
    main()
