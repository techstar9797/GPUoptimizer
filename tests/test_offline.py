"""Offline end-to-end check of one generation, with the LLM stubbed out.

The stub returns a fixed policy so the pipeline (sandbox, duplicate check,
simulator, lessons, ledger) can be tested without API keys. It is not the agent.
Run: python -m pytest tests/ -q   (or python tests/test_offline.py)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for k in ("MONGODB_URI", "OPENAI_API_KEY", "VOYAGE_API_KEY", "RAINDROP_WRITE_KEY"):
    os.environ[k] = ""

from backfill import llm  # noqa: E402
from backfill import agent  # noqa: E402
from backfill.harness import create_root  # noqa: E402
from backfill.sandbox import PolicyRejected, check_source  # noqa: E402
from backfill.store import LocalStore  # noqa: E402

POLICY = '''Pack shared GPUs first.
```python
import numpy as np
def score(ctx):
    g = np.where(ctx.gpu_free < 0, 0, ctx.gpu_free)
    left = g.sum(axis=1) / np.maximum(ctx.n_gpu * 1000, 1)
    return -left
```'''


def fake_complete(system, user, **kw):
    text = POLICY if "Write an improved policy" in user else \
        '{"lessons": [{"family": "general", "text": "Packing by GPU leftover alone ignores CPU limits."}]}'
    return llm.Reply(text=text, prompt_tokens=100, completion_tokens=50, model="stub")


def test_sandbox_rejects():
    for bad in ["import os\ndef score(ctx): return 0", "def score(ctx): return open('x')",
                "def score(ctx): return ctx.__class__", "x = 1"]:
        try:
            check_source(bad)
            raise AssertionError(f"accepted: {bad}")
        except PolicyRejected:
            pass


def test_generation_and_duplicate():
    llm.complete = fake_complete
    d = Path(tempfile.mkdtemp())
    try:
        store = LocalStore(d)
        h = create_root(store)
        r1 = agent.run_generation(h, 1, workloads=["default"], store=store)
        assert r1["ok"], r1["error"]
        assert r1["mean_unalloc"] > 0
        assert store.find("lessons"), "critic lessons were not stored"
        r2 = agent.run_generation(h, 2, workloads=["default"], store=store)
        assert r2.get("duplicate_of") == r1["_id"], "duplicate check did not fire"
        assert "duplicate_policy" in r2["signals"]
        assert len(store.find("ledger")) == 2
    finally:
        shutil.rmtree(d)



def test_meta_epoch():
    """One outer-loop epoch: meta-agent edits the harness, child is judged on held-out workloads."""
    import itertools
    from backfill import meta
    from backfill.config import TRAIN_WORKLOADS
    counter = itertools.count()
    variants = ["-left", "-left + 0.3 * ctx.cpu_after / ctx.cpu_cap", "-left - 0.5 * ctx.mem_after / ctx.mem_cap"]

    def stub(system, user, **kw):
        if "Diagnose what is holding" in user:
            text = ('{"diagnosis": "Few valid policies.", "rationale": "Show more failures and recall lessons.", '
                    '"spec": {"context_policy": {"recent_failures": 4, "recall_lessons": 3, "top_policies": 99}, '
                    '"tools": {"lesson_recall": true}, "rules": ["Use ctx.frag."], "temperature": 0.5}}')
        elif "Write an improved policy" in user:
            v = variants[next(counter) % len(variants)]
            text = POLICY.replace("return -left", f"return {v}")
        else:
            text = '{"lessons": [{"family": "gpushare", "text": "Leftover-based packing strands small GPU slices."}]}'
        return llm.Reply(text=text, prompt_tokens=10, completion_tokens=10, model="stub")

    llm.complete = stub
    meta.TRAIN_WORKLOADS = ["default"]
    meta.HELDOUT_WORKLOADS = ["gpushare80"]
    agent.TRAIN_WORKLOADS = ["default"]
    d = Path(tempfile.mkdtemp())
    try:
        store = LocalStore(d)
        h = create_root(store)
        after = meta.run_epoch(h, 1, gens_per_harness=1, store=store)
        versions = store.find("harness_versions", sort=[("version", 1)])
        assert len(versions) == 2
        kid = versions[1]
        assert kid["parent_id"] == h["_id"]
        assert kid["spec"]["context_policy"]["top_policies"] == 5, "bounds were not enforced"
        assert kid["spec"]["tools"]["lesson_recall"] is True
        assert kid["status"] in ("active", "rejected")
        assert len(store.lineage(kid["_id"])) == 2
        off = {**h["spec"], "tools": {**h["spec"]["tools"], "lesson_recall": False}}
        locked_spec, notes = meta.validate_spec(off, {"tools": {"lesson_recall": True}}, meta.MEMORY_KEYS)
        assert locked_spec["tools"]["lesson_recall"] is False and notes
        print(f"  epoch result: v{kid['version']} {kid['status']}, active v{after['version']}")
    finally:
        shutil.rmtree(d)
        meta.TRAIN_WORKLOADS = TRAIN_WORKLOADS


if __name__ == "__main__":
    test_sandbox_rejects()
    test_generation_and_duplicate()
    test_meta_epoch()
    print("offline tests passed")
