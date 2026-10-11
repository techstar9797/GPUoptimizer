"""The harness: the agent's environment, as a versioned document the agent can rewrite.

A harness holds everything that shapes how the policy-writing agent works:
rules it follows, what context it is shown, which tools it may use, and the
guardrails its output must pass. The outer loop (meta-agent) edits these fields;
each edit is saved to Atlas as a new version pointing at its parent.
"""
from __future__ import annotations

import copy
import json
import time

from .store import get_store, new_id

CTX_API = """\
Write a Python function `score(ctx)` that returns a numpy array of scores, one per
candidate node. The scheduler places the task on the highest-scoring node.
Only `numpy` (as np) and `math` are available. No other imports, no I/O.

ctx.pod             the task: .cpu (milli-CPU), .mem (MiB), .num_gpu (0..8),
                    .gpu_milli (per-GPU share, 1..1000; <1000 only when num_gpu == 1),
                    .gpu_demand (total GPU milli), .is_sharing (fractional GPU task),
                    .spec (None or frozenset of allowed GPU models)
Per candidate node (arrays of length C; every candidate can already fit the task):
ctx.cpu_free, ctx.mem_free           free CPU / memory now          [C]
ctx.cpu_cap, ctx.mem_cap             capacity                       [C]
ctx.gpu_free                         free milli per GPU slot, -1 = slot absent   [C, 8]
ctx.n_gpu                            GPUs on the node               [C]
ctx.model                            GPU model name (str)           [C]
ctx.cpu_after, ctx.mem_after, ctx.gpu_after   state if the task is placed there  [C], [C], [C, 8]
ctx.frag(gpu_free, cpu_free, mem_free, model) expected fragmentation (GPU milli) of
                    each node against this workload's task-shape mix  -> [C]
ctx.types           workload task-shape mix: .cpu .mem .num_gpu .gpu_milli .prob (arrays)
ctx.progress        arrived GPU demand / cluster capacity, 0..1 (how full the cluster is)
ctx.rng             numpy Generator for tie-breaking

The cluster has 1,213 nodes and 6,212 GPUs (8-, 4-, 2- and 1-GPU nodes; 7 GPU models).
Tasks arrive until demand reaches 100% of capacity; nothing departs. The score is the
number of GPUs left unallocated at the end: lower is better. Each call must be fast:
it runs about 8,000 times per simulation.
"""

DEFAULT_SPEC = {
    "rules": [
        "Return a finite float for every candidate.",
        "Prefer vectorized numpy over Python loops over candidates.",
    ],
    "context_policy": {
        "top_policies": 2,          # best earlier policies shown with their code
        "recent_failures": 2,       # recent rejected or crashed attempts shown
        "recall_lessons": 3,        # lessons pulled from Atlas Vector Search
        "show_per_workload": True,  # break scores down by workload
    },
    "tools": {
        "duplicate_check": True,    # Atlas vector search over earlier policy code
        "lesson_recall": True,      # Atlas vector search over lessons
        "frag_helper": True,        # ctx.frag is described to the agent
    },
    "guardrails": {
        "max_code_lines": 120,
        "duplicate_similarity": 0.985,
        "duplicate_overlap": 0.97,
        "sim_timeout_s": 240,
    },
    "temperature": 0.8,
}


def create_root(store=None, spec: dict | None = None, note: str = "Initial harness") -> dict:
    store = store or get_store()
    doc = {"_id": new_id("h"), "version": 1, "parent_id": None, "created_at": time.time(),
           "spec": copy.deepcopy(spec or DEFAULT_SPEC), "rationale": note, "author": "human",
           "score": None, "heldout_score": None, "status": "active"}
    store.insert("harness_versions", doc)
    return doc


def child(parent: dict, spec: dict, rationale: str, store=None, author: str = "meta-agent") -> dict:
    store = store or get_store()
    newest = store.find("harness_versions", sort=[("version", -1)], limit=1)
    version = (newest[0]["version"] if newest else parent["version"]) + 1
    doc = {"_id": new_id("h"), "version": version, "parent_id": parent["_id"],
           "created_at": time.time(), "spec": spec, "rationale": rationale, "author": author,
           "score": None, "heldout_score": None, "status": "candidate"}
    store.insert("harness_versions", doc)
    return doc


def latest_active(store=None) -> dict | None:
    store = store or get_store()
    rows = store.find("harness_versions", {"status": "active"}, sort=[("version", -1)], limit=1)
    return rows[0] if rows else None


def spec_text(spec: dict) -> str:
    return json.dumps(spec, indent=2, sort_keys=True)
