"""Baseline scheduling policies, after the comparisons in the FGD paper (ATC '23).

Each takes a Ctx and returns one score per candidate node; the highest wins.
"""
from __future__ import annotations

import numpy as np

from ..sim import Ctx


def _gpu_free_total(g: np.ndarray) -> np.ndarray:
    return np.where(g < 0, 0, g).sum(axis=1)


def random_fit(ctx: Ctx) -> np.ndarray:
    return ctx.rng.random(len(ctx.cpu_free))


def best_fit(ctx: Ctx) -> np.ndarray:
    """Pick the node with the least normalized resources left after placement."""
    gcap = np.maximum(ctx.n_gpu * 1000, 1)
    left = (ctx.cpu_after / ctx.cpu_cap
            + ctx.mem_after / ctx.mem_cap
            + _gpu_free_total(ctx.gpu_after) / gcap)
    return -left


def dot_product(ctx: Ctx) -> np.ndarray:
    """Tetris-style alignment between the task's demand and the node's free resources."""
    p = ctx.pod
    gcap = np.maximum(ctx.n_gpu * 1000, 1)
    return (p.cpu / ctx.cpu_cap * ctx.cpu_free / ctx.cpu_cap
            + p.mem / ctx.mem_cap * ctx.mem_free / ctx.mem_cap
            + p.gpu_demand / gcap * _gpu_free_total(ctx.gpu_free) / gcap)


def gpu_packing(ctx: Ctx) -> np.ndarray:
    """Prefer already-shared GPUs, then nodes with GPUs in use, then idle nodes."""
    g = ctx.gpu_free
    partial_fits = ((g > 0) & (g < 1000) & (g >= ctx.pod.gpu_milli)).any(axis=1)
    in_use = ((g >= 0) & (g < 1000)).any(axis=1)
    tie = -_gpu_free_total(ctx.gpu_after) / 1e6
    if ctx.pod.is_sharing:
        return 2.0 * partial_fits + 1.0 * in_use + tie
    return 1.0 * in_use + tie


def fgd(ctx: Ctx) -> np.ndarray:
    """Fragmentation Gradient Descent: minimize the growth of expected fragmentation."""
    before = ctx.frag(ctx.gpu_free, ctx.cpu_free, ctx.mem_free, ctx.model)
    after = ctx.frag(ctx.gpu_after, ctx.cpu_after, ctx.mem_after, ctx.model)
    return -(after - before)


BASELINES = {
    "random": random_fit,
    "dot_product": dot_product,
    "gpu_packing": gpu_packing,
    "best_fit": best_fit,
    "fgd": fgd,
}
