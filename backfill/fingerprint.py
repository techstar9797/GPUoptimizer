"""Workload fingerprints: 12 numbers describing a workload's task mix.

Stored in Atlas with a vector index so an unseen workload can be routed to the
closest workload the harness has already learned on.
"""
from __future__ import annotations

import numpy as np

from .trace import Pods

FEATURES = [
    "share_cpu_only", "share_fractional", "share_whole_1gpu", "share_multi_gpu",
    "share_8gpu", "share_pinned_model", "frac_size_mean", "frac_size_p25",
    "frac_size_p75", "frac_small_lt_250", "cpu_per_gpu_task", "gpu_demand_per_task",
]


def fingerprint(p: Pods) -> list[float]:
    n = len(p)
    g = p.num_gpu
    frac = (g == 1) & (p.gpu_milli < 1000)
    fm = p.gpu_milli[frac] / 1000 if frac.any() else np.array([0.0])
    gpu_tasks = g > 0
    v = np.array([
        (g == 0).mean(),
        frac.mean(),
        ((g == 1) & (p.gpu_milli >= 1000)).mean(),
        (g > 1).mean(),
        (g == 8).mean(),
        np.mean([s is not None for s in p.spec]),
        fm.mean(),
        np.percentile(fm, 25),
        np.percentile(fm, 75),
        (fm < 0.25).mean(),
        (p.cpu[gpu_tasks].mean() / 64000) if gpu_tasks.any() else 0.0,
        p.gpu_demand().mean() / 8000,
    ])
    assert len(v) == len(FEATURES)
    return [round(float(x), 6) for x in v]
