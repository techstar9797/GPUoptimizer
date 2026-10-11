"""Load agent-written policy code safely and score it in worker processes.

The agent writes a function ``score(ctx) -> np.ndarray``. Before anything runs we
parse the code and reject imports, dunder access and dangerous builtins. Each
evaluation then runs in a separate process with a hard timeout, so an infinite
loop or a crash cannot take the harness down.
"""
from __future__ import annotations

import ast
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field

import numpy as np

ALLOWED_IMPORTS = {"numpy", "math"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit",
    "memoryview",
}
SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in ["abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "float", "int",
              "bool", "round", "sorted", "list", "tuple", "dict", "set", "any", "all", "map",
              "filter", "isinstance", "reversed", "pow", "divmod", "ValueError", "Exception"]
}


class PolicyRejected(Exception):
    """The code broke a sandbox rule before it ran."""


def check_source(src: str) -> None:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise PolicyRejected(f"syntax error on line {e.lineno}: {e.msg}") from e
    has_score = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) \
                else [(node.module or "").split(".")[0]]
            bad = [n for n in names if n not in ALLOWED_IMPORTS]
            if bad:
                raise PolicyRejected(f"import of {', '.join(bad)} is not allowed (numpy and math only)")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise PolicyRejected(f"access to private attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise PolicyRejected(f"use of '{node.id}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise PolicyRejected("global and nonlocal statements are not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == "score":
            has_score = True
    if not has_score:
        raise PolicyRejected("code must define a function named score(ctx)")


def load_policy(src: str):
    check_source(src)
    env = {"__builtins__": SAFE_BUILTINS, "np": np, "numpy": np, "math": math}

    def _import(name, *a, **k):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(name)
        return __import__(name, *a, **k)
    env["__builtins__"] = {**SAFE_BUILTINS, "__import__": _import}
    exec(compile(src, "<policy>", "exec"), env)  # noqa: S102 - checked above, restricted globals
    return env["score"]


@dataclass
class Evaluation:
    ok: bool
    mean_unalloc: float | None
    per_workload: dict = field(default_factory=dict)   # workload -> unallocated GPUs
    curves: dict = field(default_factory=dict)         # workload -> [(x, alloc_frac)]
    error: str | None = None
    seconds: float = 0.0


def _worker(src: str, workload: str, seed: int, sim_limit: float):
    from .sim import simulate
    from .trace import load_nodes, load_pods
    try:
        policy = load_policy(src)
    except Exception as e:
        return workload, None, None, f"{type(e).__name__}: {e}"
    r = simulate(policy, load_nodes(), load_pods(workload), seed=seed, name="candidate",
                 time_limit=sim_limit)
    return workload, r.unalloc_gpus, r.curve, r.error


def evaluate(src: str, workloads: list[str], seed: int = 0, timeout: float = 240.0,
             processes: int | None = None) -> Evaluation:
    """Score a policy on each workload in parallel. Lower mean_unalloc is better."""
    t0 = time.time()
    try:
        check_source(src)
    except PolicyRejected as e:
        return Evaluation(ok=False, mean_unalloc=None, error=f"Rejected: {e}")
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=processes or min(len(workloads), mp.cpu_count()))
    try:
        jobs = [pool.apply_async(_worker, (src, w, seed, timeout * 0.9)) for w in workloads]
        per, curves, errors = {}, {}, []
        deadline = t0 + timeout
        for j in jobs:
            w, unalloc, curve, err = j.get(timeout=max(1.0, deadline - time.time()))
            if err:
                errors.append(f"{w}: {err}")
            else:
                per[w], curves[w] = unalloc, curve
    except mp.TimeoutError:
        pool.terminate()
        return Evaluation(ok=False, mean_unalloc=None, error=f"Timed out after {timeout:.0f}s",
                          seconds=time.time() - t0)
    finally:
        pool.terminate()
    if errors:
        return Evaluation(ok=False, mean_unalloc=None, per_workload=per, error="; ".join(errors),
                          seconds=time.time() - t0)
    return Evaluation(ok=True, mean_unalloc=float(np.mean(list(per.values()))), per_workload=per,
                      curves=curves, seconds=time.time() - t0)
