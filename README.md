# Backfill

A self-improving agent harness that teaches itself to schedule a real GPU cluster.

GPU clusters strand capacity. When tasks ask for fractions of a GPU, the leftovers on each card become too small for the next task, and whole GPUs sit idle while jobs wait. Backfill's agent writes scheduling policies, replays them against a production trace, and rewrites its own harness (rules, context policy, tools, guardrails) based on hard metrics. Its target is the published state of the art: Fragmentation Gradient Descent (FGD), USENIX ATC '23.

## What was built, and when

Everything in this repository was written on October 11, 2026, starting at 10:30 AM PT. The git history is the record.

| Part | Status |
|---|---|
| Trace loader for Alibaba `cluster-trace-gpu-v2023` | done |
| Monte Carlo GPU-sharing simulator, with state deduplication for speed | done |
| Baseline policies: random, dot-product, GPU packing, best-fit, FGD | done |
| Console: live cluster map of all 6,212 GPUs, failure curve, leaderboard | done |
| Inner loop: OpenAI writes policies; sandboxed, deduplicated, simulated, critiqued | done |
| Outer loop: a meta-agent rewrites the harness; children kept only if they generalize | done |
| MongoDB Atlas: harness lineage, runs, lessons, workload fingerprints, ledger | done |
| Atlas Vector Search: duplicate check, lesson recall, workload routing | done |
| Raindrop: interactions, tool spans, outcome and edit signals per harness version | done |
| Console: Harness view, live over an Atlas change stream | done |

## How it works

```
outer loop (backfill/meta.py)             meta-agent reads stats from Atlas, rewrites the harness
  └─ harness vN  (Atlas: harness_versions, parent_id → vN-1)
       └─ inner loop (backfill/agent.py)  per generation:
            build context ── Atlas Vector Search recalls lessons
            OpenAI writes score(ctx)
            guardrails + sandbox (backfill/sandbox.py)
            Atlas Vector Search: near-duplicate of an earlier policy? reuse its score
            simulator scores it on 4 training workloads (backfill/sim.py)
            critic writes lessons → Voyage embeddings → Atlas
  child harness kept only if its best policy beats the parent's on 4 held-out workloads
Raindrop observes every step (backfill/observe.py)
```

## Where MongoDB Atlas is used

| Feature | What it does here | Code |
|---|---|---|
| Documents with a changing shape | Each harness version is a document; the meta-agent can add rules, tools and guardrails without a migration | `backfill/harness.py`, `backfill/meta.py` |
| `$graphLookup` | Walks a harness version's ancestry for the lineage view | `AtlasStore.lineage` in `backfill/store.py` |
| Vector Search on `runs.code_embedding` | Finds a near-identical earlier policy and reuses its score instead of simulating again | `run_generation` in `backfill/agent.py` |
| Vector Search on `lessons.embedding`, filtered by workload family | Recalls what earlier generations learned about the workload the agent is weakest on | `build_prompt` in `backfill/agent.py` |
| Vector Search on `workloads.fingerprint` | Routes an unseen workload to the closest known one | `scripts/route.py` |
| Aggregation pipeline | Best result per harness version | `AtlasStore.leaderboard` |
| Change streams | Pushes new runs, lessons and harness versions to the console as they land | `events` in `backfill/server.py` |

Setup creates the collections and the three vector indexes: `python scripts/setup_atlas.py`.

## Where Raindrop is used

| What the agent does | What Raindrop records | Code |
|---|---|---|
| A generation (prompt, policy, score, critique) | One interaction, tagged with `harness_version`, `harness_id`, `generation` | `Generation` in `backfill/observe.py` |
| Writes policy code | The code as a `code` attachment | `run_generation` |
| Calls a tool: LLM, simulator, Atlas Vector Search | A tool span with input, output, duration and errors | `Generation.tool` |
| Gets a measured result | Signals: `beat_best`, `regressed`, `beat_fgd`, `duplicate_policy`, `sandbox_violation`, `policy_crashed`, `guardrail_tripped` | `run_generation` |
| Rewrites its harness | An `edit` signal with the spec before and after, and the rationale; then `harness_accepted` or `harness_rejected` | `run_epoch` in `backfill/meta.py` |

Because every event carries the harness version, Raindrop shows how the agent's behavior changes as the harness rewrites itself.

## Demo: live or recorded

- **Live:** `python -m backfill.server` while `scripts/run_harness.py` runs. The Harness view updates over an Atlas change stream: headline result, progress against FGD, run log, harness lineage, and the memory on/off comparison.
- **Recorded:** `python scripts/snapshot.py` saves the run from Atlas to `results/snapshot.json`. `python -m backfill.server --replay` serves it with no network, and **Replay the run** plays it back step by step.

## Data

`data/` holds the CSVs from Alibaba's [cluster-trace-gpu-v2023](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023), published by Alibaba Group with the paper:

> Qizhen Weng, Lingyun Yang, Yinghao Yu, Wei Wang, Xiaochuan Tang, Guodong Yang, Liping Zhang. *Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent.* USENIX ATC '23.

- **Cluster:** 1,213 GPU nodes and 6,212 GPUs across 7 GPU models.
- **Workloads:** 25 task pools of about 8,000 tasks each:
  - **default:** the production mix.
  - **gpushare\*:** heavier on fractional-GPU tasks.
  - **multigpu\*:** heavier on multi-GPU tasks.
  - **gpuspec\*:** tasks pinned to specific GPU models.
  - **cpu\*:** varying CPU-only share.

## How the simulator scores a policy

The method follows the FGD paper. Tasks are sampled from a workload and arrive one at a time until the GPU demand that has arrived equals 100% of cluster capacity. Nothing departs. For each task, the simulator finds the nodes that can host it (CPU, memory, GPU model, GPU fit), and the policy scores them. The highest score wins. Within a node, a fractional task goes on the tightest GPU that fits.

**Headline metric:** unallocated GPUs at the end of the run. Lower is better, and every point is real hardware sitting idle.

A policy is a function `policy(ctx) -> scores`. `ctx` exposes, for each candidate node:
- free and total CPU, memory and per-GPU capacity
- the GPU model
- the node's state if the task were placed there
- a `frag()` helper that computes expected fragmentation against the workload's task-shape distribution

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # fill in the keys
python scripts/setup_atlas.py                     # collections, vector indexes, harness v1
python scripts/run_harness.py --epochs 10 --gens 3            # the long run
python scripts/run_harness.py --epochs 10 --gens 3 --no-memory --db backfill_ablation
python tests/test_offline.py                      # offline checks, no keys needed
python -m backfill.server                         # console at http://localhost:8000
python scripts/run_baselines.py                      # 4 workloads x 3 seeds
python scripts/run_baselines.py --workloads default --seeds 1
```
