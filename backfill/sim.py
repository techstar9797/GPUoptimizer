"""Monte Carlo GPU-sharing cluster simulator (methodology after FGD, ATC '23).

Tasks are sampled from a workload and arrive one by one until the GPU demand that
has arrived reaches a target fraction of cluster capacity. Nothing departs. A
scheduling policy picks one node per task from the nodes that can host it. The
headline metric is how much GPU capacity is still unallocated when arrived
demand reaches 100% of capacity: lower means less fragmentation.

A policy is a function ``policy(ctx) -> np.ndarray`` returning one score per
candidate node (higher is better). See ``Ctx`` for what it can read.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .trace import GPU_MILLI, MAX_GPUS, Nodes, Pods, TaskTypes, task_types

ABSENT = -1  # gpu_free value for GPU slots a node does not have


@dataclass
class Pod:
    cpu: int
    mem: int
    num_gpu: int
    gpu_milli: int
    spec: frozenset | None

    @property
    def gpu_demand(self) -> int:
        return self.gpu_milli if self.num_gpu == 1 else self.num_gpu * GPU_MILLI

    @property
    def is_sharing(self) -> bool:
        return self.num_gpu == 1 and self.gpu_milli < GPU_MILLI


@dataclass
class Ctx:
    """Everything a policy sees. Arrays are indexed by candidate (C feasible nodes)."""
    pod: Pod
    cpu_free: np.ndarray      # [C]
    mem_free: np.ndarray      # [C]
    gpu_free: np.ndarray      # [C, 8], ABSENT (-1) for missing slots
    cpu_cap: np.ndarray       # [C]
    mem_cap: np.ndarray       # [C]
    n_gpu: np.ndarray         # [C]
    model: np.ndarray         # [C] GPU model names
    # State of each candidate if the pod were placed there:
    cpu_after: np.ndarray
    mem_after: np.ndarray
    gpu_after: np.ndarray
    frag: Callable            # frag(gpu_free, cpu_free, mem_free, model) -> [C]
    types: TaskTypes          # workload task-shape distribution
    progress: float           # arrived GPU demand / capacity, 0..1
    rng: np.random.Generator


class FragModel:
    """Expected GPU fragmentation of a node w.r.t. a task-shape distribution."""

    def __init__(self, types: TaskTypes, models: np.ndarray):
        self.t = types
        self.models = list(np.unique(models))
        idx = {m: i for i, m in enumerate(self.models)}
        k = len(types.prob)
        self.spec_ok = np.ones((k, len(self.models)), dtype=bool)
        for j, s in enumerate(types.spec):
            if s is not None:
                for m, i in idx.items():
                    self.spec_ok[j, i] = m in s
        self.model_idx = idx
        self.thresh = np.where(types.num_gpu == 1, types.gpu_milli, GPU_MILLI)  # [K]

    def __call__(self, gpu_free, cpu_free, mem_free, model) -> np.ndarray:
        f = np.where(gpu_free < 0, 0, gpu_free)                      # [C,8]
        total = f.sum(axis=1)                                        # [C]
        t = self.t
        mi = np.array([self.model_idx.get(m, 0) for m in model])
        res_ok = (cpu_free[:, None] >= t.cpu[None]) & (mem_free[:, None] >= t.mem[None])
        spec_ok = self.spec_ok[:, mi].T                              # [C,K]
        maxf = f.max(axis=1)
        nfull = (f == GPU_MILLI).sum(axis=1)
        gpu_fit = np.where(
            t.num_gpu[None] == 0, True,
            np.where(t.num_gpu[None] == 1, maxf[:, None] >= t.gpu_milli[None],
                     nfull[:, None] >= t.num_gpu[None]))
        can = res_ok & spec_ok & gpu_fit
        # GPUs with less free than the task needs per GPU are fragments
        small = (f[:, None, :] < self.thresh[None, :, None]) * f[:, None, :]
        part = small.sum(axis=2)                                     # [C,K]
        per_type = np.where(can & (t.num_gpu[None] > 0), part, total[:, None])
        return per_type @ t.prob


@dataclass
class Result:
    policy: str
    workload: str
    seed: int
    capacity_gpus: float
    alloc_gpus: float          # allocated at the stop point
    unalloc_gpus: float        # capacity - allocated
    failed_pods: int
    failed_gpus: float         # GPU demand of tasks that could not be placed
    arrived_pods: int
    curve: list = field(default_factory=list)  # [(arrived_frac, alloc_frac)]
    seconds: float = 0.0
    error: str | None = None

    @property
    def alloc_rate(self) -> float:
        return self.alloc_gpus / self.capacity_gpus

    def row(self) -> dict:
        return {
            "policy": self.policy, "workload": self.workload, "seed": self.seed,
            "alloc_rate": round(self.alloc_rate, 4),
            "unalloc_gpus": round(self.unalloc_gpus, 1),
            "failed_gpus": round(self.failed_gpus, 1),
            "failed_pods": self.failed_pods, "secs": round(self.seconds, 2),
            **({"error": self.error} if self.error else {}),
        }


class Cluster:
    def __init__(self, nodes: Nodes):
        self.nodes = nodes
        self.cpu_free = nodes.cpu.copy()
        self.mem_free = nodes.mem.copy()
        self.gpu_free = np.where(nodes.gpu_cap > 0, nodes.gpu_cap, ABSENT).astype(np.int64)
        self.model = nodes.model
        self.models = np.unique(nodes.model)

    def feasible(self, pod: Pod) -> np.ndarray:
        ok = (self.cpu_free >= pod.cpu) & (self.mem_free >= pod.mem)
        if pod.num_gpu > 0:
            if pod.spec is not None:
                ok &= np.isin(self.model, list(pod.spec))
            if pod.num_gpu == 1:
                ok &= self.gpu_free.max(axis=1) >= pod.gpu_milli
            else:
                ok &= (self.gpu_free == GPU_MILLI).sum(axis=1) >= pod.num_gpu
        return np.flatnonzero(ok)

    @staticmethod
    def place_gpus(gpu_free: np.ndarray, pod: Pod) -> np.ndarray:
        """Return gpu_free rows after placing pod (best-fit GPU within each node)."""
        g = gpu_free.copy()
        if pod.num_gpu == 0:
            return g
        rows = np.arange(len(g))
        if pod.num_gpu == 1:
            fits = np.where(g >= pod.gpu_milli, g, np.iinfo(np.int64).max)
            slot = fits.argmin(axis=1)
            g[rows, slot] -= pod.gpu_milli
        else:
            full = g == GPU_MILLI
            take = np.cumsum(full, axis=1) <= pod.num_gpu
            g[full & take] = 0
        return g

    def alloc_milli(self) -> int:
        cap = self.nodes.gpu_cap
        return int((cap - np.where(self.gpu_free < 0, 0, self.gpu_free)).sum())


Policy = Callable[[Ctx], np.ndarray]


def simulate(policy: Policy, nodes: Nodes, pods: Pods, *, seed: int = 0, stop_frac: float = 1.0,
             name: str = "policy", types: TaskTypes | None = None, top_k_types: int = 60,
             dedupe: bool = True, time_limit: float | None = None) -> Result:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    cl = Cluster(nodes)
    types = types or task_types(pods, top_k=top_k_types)
    frag = FragModel(types, nodes.model)
    cap = nodes.total_gpu_milli
    demand = pods.gpu_demand()
    arrived_milli = failed_milli = failed = n = 0
    curve, next_mark = [], 0.05
    prng = np.random.default_rng(seed + 1_000_003)
    err = None
    while arrived_milli < stop_frac * cap:
        i = int(rng.integers(len(pods)))
        pod = Pod(int(pods.cpu[i]), int(pods.mem[i]), int(pods.num_gpu[i]),
                  int(pods.gpu_milli[i]), pods.spec[i])
        arrived_milli += int(demand[i]); n += 1
        cand = cl.feasible(pod)
        if len(cand) == 0:
            failed += 1; failed_milli += pod.gpu_demand
        else:
            # Nodes in identical states are interchangeable; score each state once.
            if dedupe and len(cand) > 1:
                key = np.column_stack([cl.cpu_free[cand], cl.mem_free[cand],
                                       np.searchsorted(cl.models, cl.model[cand]),
                                       np.sort(cl.gpu_free[cand], axis=1)])
                _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
                rep = cand[first]
            else:
                rep, inv = cand, None
            gpu_after = cl.place_gpus(cl.gpu_free[rep], pod)
            ctx = Ctx(pod=pod, cpu_free=cl.cpu_free[rep], mem_free=cl.mem_free[rep],
                      gpu_free=cl.gpu_free[rep], cpu_cap=nodes.cpu[rep], mem_cap=nodes.mem[rep],
                      n_gpu=nodes.n_gpu[rep], model=cl.model[rep],
                      cpu_after=cl.cpu_free[rep] - pod.cpu, mem_after=cl.mem_free[rep] - pod.mem,
                      gpu_after=gpu_after, frag=frag, types=types,
                      progress=arrived_milli / cap, rng=prng)
            try:
                s = np.asarray(policy(ctx), dtype=float).reshape(-1)
                if s.shape[0] != len(rep) or not np.all(np.isfinite(s)):
                    raise ValueError(f"policy returned {s.shape} scores for {len(rep)} candidates, or non-finite")
            except Exception as e:  # a broken policy scores as a failed run
                err = f"{type(e).__name__}: {e}"
                break
            j = int(np.argmax(s))
            if inv is not None:
                node = int(cand[np.flatnonzero(inv.reshape(-1) == j)[0]])
            else:
                node = int(rep[j])
            cl.cpu_free[node] -= pod.cpu
            cl.mem_free[node] -= pod.mem
            cl.gpu_free[node] = gpu_after[j]
        while arrived_milli / cap >= next_mark:
            curve.append((round(next_mark, 2), cl.alloc_milli() / cap)); next_mark += 0.05
        if time_limit and time.time() - t0 > time_limit:
            err = f"time limit {time_limit}s exceeded"
            break
    alloc = cl.alloc_milli()
    return Result(policy=name, workload=pods.name, seed=seed, capacity_gpus=cap / GPU_MILLI,
                  alloc_gpus=alloc / GPU_MILLI, unalloc_gpus=(cap - alloc) / GPU_MILLI,
                  failed_pods=failed, failed_gpus=failed_milli / GPU_MILLI, arrived_pods=n,
                  curve=curve, seconds=time.time() - t0, error=err)
