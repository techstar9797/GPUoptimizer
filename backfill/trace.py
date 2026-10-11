"""Load the Alibaba cluster-trace-gpu-v2023 node and pod lists.

Data: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023
Paper: Weng et al., "Beware of Fragmentation", USENIX ATC '23.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_GPUS = 8
GPU_MILLI = 1000


@dataclass
class Nodes:
    names: np.ndarray        # [N]
    cpu: np.ndarray          # [N] milli-CPU capacity
    mem: np.ndarray          # [N] MiB capacity
    n_gpu: np.ndarray        # [N] GPUs per node
    model: np.ndarray        # [N] GPU model string
    gpu_cap: np.ndarray      # [N, 8] milli per GPU slot (0 = slot absent)

    @property
    def total_gpu_milli(self) -> int:
        return int(self.gpu_cap.sum())


@dataclass
class Pods:
    """A workload: an ordered pool of tasks to sample arrivals from."""
    name: str
    cpu: np.ndarray          # [P]
    mem: np.ndarray          # [P]
    num_gpu: np.ndarray      # [P] 0..8
    gpu_milli: np.ndarray    # [P] per-GPU milli (only <1000 when num_gpu == 1)
    spec: list               # [P] None or frozenset of allowed models

    def __len__(self) -> int:
        return len(self.cpu)

    def gpu_demand(self) -> np.ndarray:
        """Total GPU milli each pod requests."""
        return np.where(self.num_gpu == 1, self.gpu_milli, self.num_gpu * GPU_MILLI)


def load_nodes(path: Path | None = None) -> Nodes:
    df = pd.read_csv(path or DATA_DIR / "openb_node_list_gpu_node.csv")
    n = len(df)
    n_gpu = df["gpu"].to_numpy(int)
    cap = np.zeros((n, MAX_GPUS), dtype=np.int64)
    for i, g in enumerate(n_gpu):
        cap[i, :g] = GPU_MILLI
    return Nodes(
        names=df["sn"].to_numpy(),
        cpu=df["cpu_milli"].to_numpy(np.int64),
        mem=df["memory_mib"].to_numpy(np.int64),
        n_gpu=n_gpu,
        model=df["model"].fillna("").to_numpy(str),
        gpu_cap=cap,
    )


def load_pods(workload: str = "default", path: Path | None = None) -> Pods:
    df = pd.read_csv(path or DATA_DIR / f"openb_pod_list_{workload}.csv")
    spec = [
        None if (not isinstance(s, str) or not s) else frozenset(s.split("|"))
        for s in (df["gpu_spec"] if "gpu_spec" in df else [None] * len(df))
    ]
    num_gpu = df["num_gpu"].to_numpy(np.int64)
    gm = df["gpu_milli"].fillna(0).to_numpy(np.int64)
    gm = np.where(num_gpu == 0, 0, np.where(num_gpu > 1, GPU_MILLI, gm))
    return Pods(
        name=workload,
        cpu=df["cpu_milli"].to_numpy(np.int64),
        mem=df["memory_mib"].to_numpy(np.int64),
        num_gpu=num_gpu,
        gpu_milli=gm,
        spec=spec,
    )


def list_workloads() -> list[str]:
    return sorted(p.stem.replace("openb_pod_list_", "") for p in DATA_DIR.glob("openb_pod_list_*.csv"))


@dataclass
class TaskTypes:
    """Distribution of distinct task shapes in a workload (the 'M' in the FGD paper)."""
    cpu: np.ndarray
    mem: np.ndarray
    num_gpu: np.ndarray
    gpu_milli: np.ndarray
    spec: list
    prob: np.ndarray


def task_types(pods: Pods, top_k: int | None = None) -> TaskTypes:
    keys: dict = {}
    for i in range(len(pods)):
        k = (int(pods.cpu[i]), int(pods.mem[i]), int(pods.num_gpu[i]), int(pods.gpu_milli[i]), pods.spec[i])
        keys[k] = keys.get(k, 0) + 1
    items = sorted(keys.items(), key=lambda kv: -kv[1])
    if top_k:
        items = items[:top_k]
    tot = sum(c for _, c in items)
    return TaskTypes(
        cpu=np.array([k[0] for k, _ in items], np.int64),
        mem=np.array([k[1] for k, _ in items], np.int64),
        num_gpu=np.array([k[2] for k, _ in items], np.int64),
        gpu_milli=np.array([k[3] for k, _ in items], np.int64),
        spec=[k[4] for k, _ in items],
        prob=np.array([c / tot for _, c in items]),
    )
