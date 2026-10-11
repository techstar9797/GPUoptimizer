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
| Inner loop: agent writes policies, sandboxed and scored | next |
| Outer loop: harness rewrites itself, versioned in MongoDB Atlas | planned |
| Long-horizon memory with Atlas Vector Search | planned |

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
pip install -r requirements.txt
python -m backfill.server                         # console at http://localhost:8000
python scripts/run_baselines.py                      # 4 workloads x 3 seeds
python scripts/run_baselines.py --workloads default --seeds 1
```
