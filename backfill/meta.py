"""Outer loop: a meta-agent rewrites the harness from measured outcomes.

Each epoch:
  1. run a few inner generations under the active harness
  2. summarize how the agent behaved under it (scores, failures, duplicates,
     guardrail trips, tokens), straight from the runs stored in Atlas
  3. the meta-agent proposes an edited harness (rules, context policy, tool
     grants, guardrails) with a rationale; the edit is validated against bounds
  4. the child harness runs the same number of generations
  5. the best policy from each harness is scored on held-out workloads; the
     child is kept only if it generalizes better than its parent
Every rewrite is a new document in Atlas with parent_id, so the lineage is a tree.
"""
from __future__ import annotations

import copy
import json
import time

from . import llm
from .agent import baseline_scores, run_generation
from .config import HELDOUT_WORKLOADS, SETTINGS, TRAIN_WORKLOADS
from .harness import child, spec_text
from .observe import Generation
from .sandbox import evaluate
from .store import get_store, new_id

BOUNDS = {
    "context_policy": {"top_policies": (0, 5), "recent_failures": (0, 5), "recall_lessons": (0, 8),
                       "show_per_workload": bool},
    "tools": {"duplicate_check": bool, "lesson_recall": bool, "frag_helper": bool},
    "guardrails": {"max_code_lines": (30, 250), "duplicate_similarity": (0.9, 0.999),
                   "duplicate_overlap": (0.85, 1.0),
                   "sim_timeout_s": (60, 400)},
}
MEMORY_KEYS = [("tools", "duplicate_check"), ("tools", "lesson_recall"), ("context_policy", "recall_lessons")]

META_SYSTEM = ("You improve the harness that an AI agent works inside. The agent writes GPU "
               "scheduling policies; you change its environment, not the policies.")


def validate_spec(parent: dict, proposed: dict, locked: list | None = None) -> tuple[dict, list[str]]:
    """Clamp a proposed spec to BOUNDS; keep the parent's value for anything invalid or locked."""
    spec, notes = copy.deepcopy(parent), []
    rules = proposed.get("rules")
    if isinstance(rules, list) and all(isinstance(r, str) for r in rules):
        spec["rules"] = [r.strip()[:300] for r in rules if r.strip()][:12]
    elif rules is not None:
        notes.append("ignored rules: not a list of strings")
    for section, fields in BOUNDS.items():
        for key, rule in fields.items():
            if key not in proposed.get(section, {}):
                continue
            if locked and (section, key) in locked:
                notes.append(f"{section}.{key} is locked for this run")
                continue
            v = proposed[section][key]
            if rule is bool:
                if isinstance(v, bool):
                    spec[section][key] = v
                else:
                    notes.append(f"ignored {section}.{key}: not true/false")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                lo, hi = rule
                clamped = min(max(v, lo), hi)
                kind = type(parent[section][key]) if key in parent.get(section, {}) else float
                spec[section][key] = kind(clamped)
                if clamped != v:
                    notes.append(f"clamped {section}.{key} to {clamped}")
            else:
                notes.append(f"ignored {section}.{key}: not a number")
    t = proposed.get("temperature")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        spec["temperature"] = float(min(max(t, 0.0), 1.2))
    return spec, notes


def harness_stats(store, harness_id: str) -> dict:
    runs = store.find("runs", {"harness_id": harness_id, "kind": "candidate"})
    ok = [r for r in runs if r.get("ok") and not r.get("duplicate_of")]
    sig = {}
    for r in runs:
        for s in r.get("signals", []):
            sig[s] = sig.get(s, 0) + 1
    led = store.find("ledger", {"harness_id": harness_id})
    best = min(ok, key=lambda r: r["mean_unalloc"]) if ok else None
    return {
        "generations": len(runs),
        "valid_policies": len(ok),
        "failed": sum(1 for r in runs if not r.get("ok")),
        "duplicates": sum(1 for r in runs if r.get("duplicate_of")),
        "signals": sig,
        "best_mean_unalloc": round(best["mean_unalloc"], 1) if best else None,
        "best_per_workload": {w: round(v) for w, v in best["per_workload"].items()} if best else None,
        "best_run_id": best["_id"] if best else None,
        "mean_tokens_per_generation": round(sum(l["tokens"] for l in led) / max(1, len(led))),
        "errors": [r["error"][:200] for r in runs if not r.get("ok")][:4],
    }


def heldout_score(store, run_id: str | None, workloads: list[str], seed: int, gen: Generation | None,
                  cache: dict) -> float | None:
    """Mean unallocated GPUs of a stored policy on held-out workloads (cached per run)."""
    if not run_id:
        return None
    if run_id in cache:
        return cache[run_id]
    run = store.get("runs", run_id)
    if gen:
        with gen.tool("simulator.heldout", input={"run": run_id, "workloads": workloads}) as t:
            ev = evaluate(run["code"], workloads, seed=seed)
            t["output"] = {"mean_unalloc": ev.mean_unalloc, "per_workload": ev.per_workload}
    else:
        ev = evaluate(run["code"], workloads, seed=seed)
    cache[run_id] = ev.mean_unalloc if ev.ok else None
    store.update("runs", run_id, {"heldout": {"workloads": workloads, "seed": seed,
                                              "mean_unalloc": ev.mean_unalloc,
                                              "per_workload": ev.per_workload}})
    return cache[run_id]


def propose(parent: dict, stats: dict, history: list[dict], locked: list, gen: Generation) -> tuple[dict, str, list]:
    base = baseline_scores()
    fgd = {w: round(base.get("fgd", {}).get(w, 0)) for w in TRAIN_WORKLOADS}
    hist = "\n".join(f"- v{h['version']} ({h['status']}): {h['rationale'][:240]} -> train best "
                     f"{h.get('score')}, held-out {h.get('heldout_score')}" for h in history) or "- none yet"
    lock_txt = ("\nThese settings are locked and cannot change: " +
                ", ".join(f"{a}.{b}" for a, b in locked)) if locked else ""
    prompt = f"""The agent's current harness (version {parent['version']}):
```json
{spec_text(parent['spec'])}
```
How the agent did under it over its last {stats['generations']} generations:
```json
{json.dumps(stats, indent=1)}
```
Target to beat, FGD (USENIX ATC '23), unallocated GPUs per training workload: {fgd}
Earlier harness versions and how they did:
{hist}

What each setting does:
- rules: instructions added to the agent's prompt (max 12)
- context_policy.top_policies: best earlier policies shown with code
- context_policy.recent_failures: recent failed attempts shown with their errors
- context_policy.recall_lessons + tools.lesson_recall: lessons retrieved by vector search from earlier generations
- context_policy.show_per_workload: show scores per workload, not only the mean
- tools.duplicate_check: skip simulating policies nearly identical to earlier ones
- tools.frag_helper: describe ctx.frag (expected fragmentation) to the agent
- guardrails.duplicate_similarity: vector similarity at which an earlier policy is checked as a possible duplicate
- guardrails.duplicate_overlap: share of identical code tokens needed to confirm a duplicate (1.0 = only exact copies)
- guardrails.max_code_lines, guardrails.sim_timeout_s
- temperature: sampling temperature for the agent (0 to 1.2){lock_txt}

Diagnose what is holding the agent back, from the numbers above. Then propose ONE focused change
to the harness. Answer as JSON: {{"diagnosis": "...", "rationale": "...", "spec": {{...full new spec...}}}}"""
    with gen.tool("llm.meta_propose", input={"chars": len(prompt)}) as t:
        rep = llm.complete(META_SYSTEM, prompt, temperature=0.4, max_tokens=2000)
        t["output"] = rep.text[:4000]
    gen.tokens += rep.prompt_tokens + rep.completion_tokens
    try:
        obj = json.loads(llm.extract_json(rep.text))
    except json.JSONDecodeError:
        return parent["spec"], "Meta-agent reply was not valid JSON; kept the harness unchanged.", []
    spec, notes = validate_spec(parent["spec"], obj.get("spec", {}), locked)
    rationale = (obj.get("diagnosis", "") + " " + obj.get("rationale", "")).strip()[:1500]
    return spec, rationale or "No rationale given.", notes


def run_epoch(parent: dict, epoch: int, gens_per_harness: int, seed: int = 0, locked: list | None = None,
              store=None, log=print) -> dict:
    """One outer-loop step. Returns the harness that is active afterwards."""
    store = store or get_store()
    cache: dict = {}
    next_gen = lambda: len(store.find("runs", {"kind": "candidate"})) + 1  # noqa: E731

    # Make sure the parent has enough generations of evidence.
    have = len(store.find("runs", {"harness_id": parent["_id"], "kind": "candidate"}))
    for _ in range(max(0, gens_per_harness - have)):
        run_generation(parent, next_gen(), seed=seed, store=store, log=log)
    pstats = harness_stats(store, parent["_id"])

    gen = Generation(event="harness_rewrite", harness_version=parent["version"], harness_id=parent["_id"],
                     generation=epoch, workloads=TRAIN_WORKLOADS, model=SETTINGS.openai_model)
    gen.tokens = 0
    history = store.find("harness_versions", sort=[("version", 1)])
    spec, rationale, notes = propose(parent, pstats, history, locked or [], gen)
    if spec == parent["spec"]:
        gen.signal("no_change", False)
        gen.finish("Meta-agent proposed no valid change.")
        log(f"epoch {epoch}: no valid change proposed ({'; '.join(notes) or rationale[:120]})")
        return parent
    kid = child(parent, spec, rationale, store)
    gen.edit("harness_rewrite", spec_text(parent["spec"]), spec_text(spec), rationale)
    log(f"epoch {epoch}: v{parent['version']} -> v{kid['version']}: {rationale[:200]}")
    if notes:
        log("  validator: " + "; ".join(notes))
    store.update("harness_versions", kid["_id"], {"validator_notes": notes, "raindrop_event_id": gen.event_id})

    for _ in range(gens_per_harness):
        run_generation(kid, next_gen(), seed=seed, store=store, log=log)
    kstats = harness_stats(store, kid["_id"])

    # Keep the child only if its best policy generalizes better than the parent's.
    p_ho = heldout_score(store, pstats["best_run_id"], HELDOUT_WORKLOADS, seed + 1, gen, cache)
    k_ho = heldout_score(store, kstats["best_run_id"], HELDOUT_WORKLOADS, seed + 1, gen, cache)
    store.update("harness_versions", parent["_id"], {"score": pstats["best_mean_unalloc"], "heldout_score": p_ho,
                                                      "stats": pstats})
    keep = k_ho is not None and (p_ho is None or k_ho < p_ho)
    store.update("harness_versions", kid["_id"], {"score": kstats["best_mean_unalloc"], "heldout_score": k_ho,
                                                   "stats": kstats, "status": "active" if keep else "rejected",
                                                   "decided_at": time.time()})
    if keep:
        store.update("harness_versions", parent["_id"], {"status": "retired"})
        gen.signal("harness_accepted", True, parent_heldout=p_ho, child_heldout=k_ho)
    else:
        gen.signal("harness_rejected", False, parent_heldout=p_ho, child_heldout=k_ho)
    store.insert("ledger", {"_id": new_id("led"), "generation": None, "epoch": epoch, "harness_id": kid["_id"],
                            "tokens": gen.tokens, "seconds": None, "created_at": time.time(), "kind": "meta"})
    verdict = "kept" if keep else "rejected"
    gen.finish(f"v{kid['version']} {verdict}: held-out {k_ho} vs parent {p_ho}")
    fmt = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
    log(f"  v{kid['version']} {verdict}: held-out {fmt(k_ho)} vs parent v{parent['version']} {fmt(p_ho)}")
    return store.get("harness_versions", kid["_id"]) if keep else parent
