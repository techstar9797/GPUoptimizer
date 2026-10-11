"""Inner loop: the agent writes a scheduling policy, the harness checks and scores it.

One generation:
  1. build context from the harness's context policy (best policies, failures,
     lessons recalled from Atlas Vector Search)
  2. the LLM writes score(ctx)
  3. guardrails and the sandbox check the code
  4. Atlas Vector Search looks for a near-duplicate already scored
  5. the simulator scores it on the training workloads
  6. a critic turns the result into lessons, embedded and stored in Atlas
Raindrop records each step as a tool span, and each outcome as a signal.
"""
from __future__ import annotations

import difflib
import io
import json
import time
import tokenize
from pathlib import Path

from . import llm
from .config import SETTINGS, TRAIN_WORKLOADS
from .embed import embed_one
from .harness import CTX_API
from .observe import Generation
from .sandbox import evaluate, check_source, PolicyRejected
from .store import family_of, get_store, new_id

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = ("You are an expert in cluster scheduling and bin packing. You write fast, "
          "vectorized numpy scoring functions for a GPU cluster scheduler.")


def baseline_scores(seed: int = 0) -> dict:
    """Per-policy, per-workload unallocated GPUs for the given seed, from results/baselines.json."""
    f = ROOT / "results" / "baselines.json"
    out: dict = {}
    if f.exists():
        for r in json.loads(f.read_text()):
            if r["seed"] == seed:
                out.setdefault(r["policy"], {})[r["workload"]] = r["unalloc_gpus"]
    return out


def normalize_code(src: str) -> str:
    """Code without comments, docstrings or formatting, for comparing two policies."""
    out = []
    try:
        prev = None
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            if tok.type == tokenize.STRING and prev in (None, ":"):
                continue  # docstring
            out.append(tok.string)
            prev = tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return " ".join(src.split())
    return " ".join(out)


def code_overlap(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_code(a), normalize_code(b), autojunk=False).ratio()


def _fmt_scores(per: dict, workloads: list[str]) -> str:
    return ", ".join(f"{w} {per[w]:.0f}" for w in workloads if w in per)


def build_prompt(harness: dict, workloads: list[str], store, gen: Generation) -> tuple[str, list[dict]]:
    spec = harness["spec"]
    cp, tools = spec["context_policy"], spec["tools"]
    base = baseline_scores()
    parts = [CTX_API if tools.get("frag_helper", True) else CTX_API.split("ctx.frag")[0]]

    parts.append("## Baselines on these workloads (unallocated GPUs, lower is better)")
    for p in ("fgd", "gpu_packing", "best_fit"):
        if p in base:
            vals = {w: base[p][w] for w in workloads if w in base[p]}
            mean = sum(vals.values()) / max(1, len(vals))
            parts.append(f"- {p}: mean {mean:.0f} ({_fmt_scores(vals, workloads)})")

    if spec["rules"]:
        parts.append("## Rules\n" + "\n".join(f"- {r}" for r in spec["rules"]))

    top = store.find("runs", {"ok": True, "kind": "candidate"}, sort=[("mean_unalloc", 1)], limit=1)
    if top and "fgd" in base:
        gaps = sorted(((top[0]["per_workload"].get(w, 0) - base["fgd"].get(w, 0), w) for w in workloads), reverse=True)
        if gaps and gaps[0][0] > 0:
            g, w = gaps[0]
            parts.append(f"## Biggest gap\nOn {w} the best policy so far leaves {top[0]['per_workload'][w]:.0f} GPUs idle, "
                         f"{g:.0f} more than FGD ({base['fgd'][w]:.0f}). "
                         + ("Tasks there are pinned to specific GPU models (ctx.pod.spec), so placing other tasks on scarce "
                            "models strands capacity those pinned tasks need." if w.startswith("gpuspec") else
                            "Close this gap without losing ground on the other workloads."))

    best = store.find("runs", {"ok": True, "kind": "candidate"}, sort=[("mean_unalloc", 1)],
                      limit=cp.get("top_policies", 0)) if cp.get("top_policies") else []
    for i, r in enumerate(best, 1):
        detail = f" ({_fmt_scores(r['per_workload'], workloads)})" if cp.get("show_per_workload") else ""
        parts.append(f"## Best policy so far #{i}: mean {r['mean_unalloc']:.1f}{detail}\n```python\n{r['code']}\n```")

    tried = store.find("runs", {"kind": "candidate"}, sort=[("created_at", -1)], limit=8)
    if tried:
        rows = [f"- {(r.get('idea') or 'no description').splitlines()[0][:160]} -> "
                f"{('%.0f' % r['mean_unalloc']) if r.get('ok') and r.get('mean_unalloc') is not None else 'failed'}"
                + (" (duplicate)" if r.get("duplicate_of") else "") for r in tried]
        parts.append("## Ideas already tried (newest first)\n" + "\n".join(rows))

    if cp.get("recent_failures"):
        fails = store.find("runs", {"ok": False, "kind": "candidate"}, sort=[("created_at", -1)],
                           limit=cp["recent_failures"])
        if fails:
            parts.append("## Recent failed attempts\n" + "\n".join(f"- {f['error'][:300]}" for f in fails))

    recalled: list[dict] = []
    k = cp.get("recall_lessons", 0)
    if k and tools.get("lesson_recall"):
        weakest = _weakest_workload(best[0] if best else None, base, workloads)
        query = (f"How to reduce unallocated GPUs on {weakest} workloads; "
                 f"what made earlier policies regress there.")
        with gen.tool("atlas_vector_search.lessons", input={"query": query, "k": k}) as t:
            try:
                recalled = store.vector_search("lessons", embed_one(query, "query"), k=k,
                                               filter={"family": {"$in": [family_of(weakest), "general"]}})
            except Exception:
                recalled = []
            t["output"] = [{"text": l["text"], "similarity": round(l["similarity"], 3)} for l in recalled]
        if recalled:
            parts.append("## Lessons recalled from earlier generations\n" +
                         "\n".join(f"- {l['text']}" for l in recalled))

    parts.append("Write an improved policy that leaves fewer GPUs unallocated than every policy "
                 "above. Do not resubmit an idea already tried with different constants: change how "
                 "the score is built. Explain your idea in one or two sentences, then give the code "
                 "in one ```python block.")
    return "\n\n".join(parts), recalled


def _weakest_workload(best_run, base, workloads):
    """Workload where the current best trails FGD the most."""
    if not best_run or "fgd" not in base:
        return workloads[0]
    gaps = {w: best_run["per_workload"].get(w, 0) - base["fgd"].get(w, 0) for w in workloads}
    return max(gaps, key=gaps.get)


def critique(code: str, idea: str, ev, base: dict, workloads: list[str], gen: Generation) -> list[dict]:
    """Ask the LLM what this result teaches; returns lesson dicts (not yet stored)."""
    fgd = base.get("fgd", {})
    if ev.ok:
        rows = "\n".join(f"- {w}: {ev.per_workload[w]:.0f} unallocated (FGD {fgd.get(w, float('nan')):.0f})"
                         for w in workloads)
        result = f"Scores:\n{rows}\nMean {ev.mean_unalloc:.1f} (FGD mean {sum(fgd.get(w,0) for w in workloads)/len(workloads):.1f})"
    else:
        result = f"The policy failed: {ev.error}"
    prompt = (f"A scheduling policy was tried.\nIdea: {idea}\n```python\n{code}\n```\n{result}\n\n"
              "Write 1 to 3 short, specific lessons for whoever writes the next policy: what worked "
              "or failed on which workload, and why. Each lesson names a workload family "
              "(default, gpushare, multigpu, gpuspec, or general). Answer as JSON: "
              '{"lessons": [{"family": "...", "text": "..."}]}')
    with gen.tool("llm.critic", input={"chars": len(prompt)}) as t:
        rep = llm.complete(SYSTEM, prompt, temperature=0.2, max_tokens=600)
        t["output"] = rep.text[:2000]
    gen.tokens += rep.prompt_tokens + rep.completion_tokens
    try:
        lessons = json.loads(llm.extract_json(rep.text)).get("lessons", [])
    except json.JSONDecodeError:
        lessons = []
    return [l for l in lessons if isinstance(l, dict) and l.get("text")][:3]


def run_generation(harness: dict, generation: int, workloads: list[str] | None = None,
                   seed: int = 0, store=None, log=print) -> dict:
    store = store or get_store()
    workloads = workloads or TRAIN_WORKLOADS
    spec = harness["spec"]
    guards = spec["guardrails"]
    t0 = time.time()
    gen = Generation(event="policy_generation", harness_version=harness["version"],
                     harness_id=harness["_id"], generation=generation, workloads=workloads,
                     model=SETTINGS.openai_model)
    gen.tokens = 0
    base = baseline_scores(seed)

    prompt, recalled = build_prompt(harness, workloads, store, gen)
    with gen.tool("llm.write_policy", input={"chars": len(prompt)}) as t:
        rep = llm.complete(SYSTEM, prompt, temperature=spec.get("temperature", 0.8))
        t["output"] = rep.text[:4000]
    gen.tokens += rep.prompt_tokens + rep.completion_tokens
    code = llm.extract_code(rep.text)
    idea = rep.text.split("```")[0].strip()[:600]
    gen.attach_code("policy.py", code)

    run = {"_id": new_id("run"), "kind": "candidate", "harness_id": harness["_id"],
           "harness_version": harness["version"], "generation": generation, "created_at": time.time(),
           "code": code, "idea": idea, "workloads": workloads, "seed": seed,
           "recalled_lessons": [l["_id"] for l in recalled], "raindrop_event_id": gen.event_id}

    # Guardrails and sandbox
    err = None
    if code.count("\n") + 1 > guards.get("max_code_lines", 120):
        err = f"Guardrail: {code.count(chr(10)) + 1} lines, limit {guards['max_code_lines']}"
        gen.signal("guardrail_tripped", False, guardrail="max_code_lines")
    else:
        try:
            check_source(code)
        except PolicyRejected as e:
            err = f"Rejected: {e}"
            gen.signal("sandbox_violation", False, reason=str(e)[:200])

    # Duplicate check with Atlas Vector Search
    if not err:
        try:
            run["code_embedding"] = embed_one(code)
        except Exception as e:  # network trouble: skip the duplicate check rather than lose the generation
            log(f"  embedding failed ({type(e).__name__}); skipping duplicate check")
        if spec["tools"].get("duplicate_check") and "code_embedding" in run:
            with gen.tool("atlas_vector_search.runs", input={"k": 3}) as t:
                hits = store.vector_search("runs", run["code_embedding"], k=3, filter={"kind": "candidate"})
                t["output"] = [{"run": h["_id"], "similarity": round(h["similarity"], 4)} for h in hits]
            # Vector search finds look-alikes; a token-level comparison confirms a real duplicate.
            dup = None
            for h in hits:
                if h["similarity"] < guards.get("duplicate_similarity", 0.985) or not h.get("ok"):
                    continue
                full = store.get("runs", h["_id"])
                overlap = code_overlap(code, full.get("code", "")) if full else 0.0
                if overlap >= guards.get("duplicate_overlap", 0.97):
                    dup = {**h, "overlap": overlap}
                    break
            if dup:
                run.update(ok=True, duplicate_of=dup["_id"], mean_unalloc=dup["mean_unalloc"],
                           per_workload=dup["per_workload"], curves=dup.get("curves", {}),
                           seconds=0.0, error=None)
                gen.signal("duplicate_policy", False, duplicate_of=dup["_id"],
                           similarity=round(dup["similarity"], 4), overlap=round(dup["overlap"], 3))
                log(f"  duplicate of {dup['_id']} (vector similarity {dup['similarity']:.3f}, "
                    f"code overlap {dup['overlap']:.2f}); reused its score")

    if err:
        run.update(ok=False, error=err, mean_unalloc=None, per_workload={}, seconds=0.0)
    elif "duplicate_of" not in run:
        with gen.tool("simulator.evaluate", input={"workloads": workloads, "seed": seed}) as t:
            ev = evaluate(code, workloads, seed=seed, timeout=guards.get("sim_timeout_s", 240))
            t["output"] = {"ok": ev.ok, "mean_unalloc": ev.mean_unalloc, "per_workload": ev.per_workload,
                           "error": ev.error}
        run.update(ok=ev.ok, error=ev.error, mean_unalloc=ev.mean_unalloc, per_workload=ev.per_workload,
                   curves=ev.curves, seconds=ev.seconds)
        if not ev.ok:
            gen.signal("sandbox_violation" if (ev.error or "").startswith("Rejected") else "policy_crashed",
                       False, error=(ev.error or "")[:200])

    # Outcome signals against the best earlier run and the published baseline
    prev = store.find("runs", {"ok": True, "kind": "candidate"}, sort=[("mean_unalloc", 1)], limit=1)
    fgd_mean = (sum(base["fgd"][w] for w in workloads) / len(workloads)) if "fgd" in base else None
    if run["ok"] and "duplicate_of" not in run:
        if prev:
            better = run["mean_unalloc"] < prev[0]["mean_unalloc"]
            gen.signal("beat_best" if better else "regressed", better,
                       delta=round(run["mean_unalloc"] - prev[0]["mean_unalloc"], 1))
        if fgd_mean is not None and run["mean_unalloc"] < fgd_mean:
            gen.signal("beat_fgd", True, margin=round(fgd_mean - run["mean_unalloc"], 1))

    # Critic writes lessons; stored with embeddings for later recall
    lessons = []
    if "duplicate_of" not in run:
        class _E:  # light view of the evaluation for the critic
            ok, per_workload, mean_unalloc, error = run["ok"], run.get("per_workload", {}), \
                run.get("mean_unalloc"), run.get("error")
        try:
            lessons = critique(code, idea, _E, base, workloads, gen)
        except Exception as e:  # lessons are best-effort
            log(f"  critic failed: {e}")
    for l in lessons:
        doc = {"_id": new_id("les"), "text": l["text"], "family": l.get("family", "general"),
               "run_id": run["_id"], "harness_id": harness["_id"], "generation": generation,
               "outcome": run.get("mean_unalloc"), "created_at": time.time()}
        try:
            doc["embedding"] = embed_one(l["text"])
        except Exception as e:
            log(f"  lesson stored without embedding ({type(e).__name__})")
        store.insert("lessons", doc)
    run["lessons"] = [l["text"] for l in lessons]
    run["signals"] = gen.signals
    run["tools"] = gen.tools
    store.insert("runs", run)

    store.insert("ledger", {"_id": new_id("led"), "generation": generation, "harness_id": harness["_id"],
                            "tokens": gen.tokens, "seconds": round(time.time() - t0, 1),
                            "created_at": time.time()})
    summary = (f"gen {generation} h{harness['version']}: " +
               (f"{run['mean_unalloc']:.1f} unallocated" if run["ok"] else f"failed: {run['error'][:120]}"))
    gen.set(outcome=run.get("mean_unalloc"), ok=run["ok"], signals=",".join(gen.signals))
    gen.finish(summary)
    log("  " + summary + (f"  [{', '.join(gen.signals)}]" if gen.signals else ""))
    return run
