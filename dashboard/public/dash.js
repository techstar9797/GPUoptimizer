// Backfill dashboard. Polls /api/data (Atlas, live); falls back to data.json (recorded).
(function () {
  const $ = (id) => document.getElementById(id);
  const fmt = (n, d = 0) => (n == null || !isFinite(n) ? "–" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const POLL_MS = 5000;
  const NAMES = { fgd: "FGD (ATC '23)", gpu_packing: "GPU packing", best_fit: "Best-fit", dot_product: "Dot-product", random: "Random" };
  const SHORT = { fgd: "FGD", gpu_packing: "GPU packing", best_fit: "Best-fit", dot_product: "Dot-product", random: "Random" };
  const GOOD = new Set(["beat_best", "beat_fgd", "harness_accepted"]);

  // What kind of operator each workload profile resembles. Illustrative, not provider data.
  const PROFILES = {
    default: ["Production mix", "General-purpose GPU cloud"],
    gpushare40: ["Fractional GPUs, 40% of tasks", "Inference and notebooks, e.g. Together AI, Lambda"],
    gpushare60: ["Fractional GPUs, 60% of tasks", "Serverless inference, e.g. Together AI"],
    gpushare80: ["Fractional GPUs, 80% of tasks", "Small-model inference at high density"],
    multigpu20: ["Multi-GPU jobs, 20% of tasks", "Mixed training and inference, e.g. Crusoe"],
    multigpu50: ["Multi-GPU jobs, 50% of tasks", "Large-scale training, e.g. CoreWeave"],
    gpuspec33: ["Pinned to GPU model, 33%", "Mixed GPU generations in one fleet"],
    gpuspec20: ["Pinned to GPU model, 20%", "Mixed GPU generations in one fleet"],
  };

  let baselineRows = [];
  let lastOk = 0;

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { const j = await r.json(); if (j.error) msg = j.error; } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  }

  async function load() {
    let data, live = true, apiError = null;
    try {
      data = await getJSON("/api/data");
      if (data.source === "recorded" || data.source === "replay") live = false;
    } catch (e) {
      apiError = e.message;
      try { data = await getJSON("data.json"); live = false; } catch (e2) {
        $("status").className = "status";
        $("status").textContent = `Live data unavailable: ${apiError}`;
        return;
      }
    }
    lastOk = Date.now();
    const st = $("status");
    st.className = "status " + (live ? "live" : "recorded");
    st.dataset.label = live ? "Live from MongoDB Atlas" : `Recorded run${data.recorded_at || data.generated_at ? ", " + new Date((data.recorded_at || data.generated_at) * 1000).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : ""}`;
    st.textContent = st.dataset.label;
    st.title = apiError ? `Live data unavailable: ${apiError}` : "";
    render(data);
  }

  setInterval(() => {
    const st = $("status");
    if (st.classList.contains("live") && lastOk) st.textContent = `${st.dataset.label}, updated ${Math.round((Date.now() - lastOk) / 1000)}s ago`;
  }, 1000);

  // Baseline score for a policy on a workload at a seed (train uses seed 0, held-out seed 1).
  function base(policy, workload, seed) {
    const r = baselineRows.find((x) => x.policy === policy && x.workload === workload && x.seed === seed);
    return r ? r.unalloc_gpus : null;
  }
  function trainMean(policy, train) {
    const v = train.map((w) => base(policy, w, 0));
    return v.every((x) => x != null) ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }

  function render(d) {
    const M = d.main;
    const runs = M.runs;
    const ok = runs.filter((r) => r.ok && r.mean_unalloc != null && !r.duplicate_of);
    const best = ok.length ? ok.reduce((a, r) => (r.mean_unalloc < a.mean_unalloc ? r : a)) : null;
    const fgd = trainMean("fgd", d.train);
    const pack = trainMean("gpu_packing", d.train);

    // KPIs
    $("k-best").textContent = best ? fmt(best.mean_unalloc) : "–";
    if (best && fgd != null) {
      const diff = fgd - best.mean_unalloc;
      $("k-best-sub").innerHTML = `FGD (USENIX ATC '23) leaves <b>${fmt(fgd)}</b> <span class="delta ${diff > 0 ? "good" : "bad"}">${diff > 0 ? `${fmt(diff)} fewer` : `${fmt(-diff)} more`}</span>`;
    } else $("k-best-sub").textContent = "Waiting for the first scored policy";
    const kept = M.versions.filter((v) => v.parent_id && v.status !== "rejected" && v.status !== "candidate").length;
    const rejected = M.versions.filter((v) => v.status === "rejected").length;
    $("k-versions").textContent = fmt(M.versions.length);
    $("k-versions-sub").innerHTML = `<b>${kept}</b> kept, <b>${rejected}</b> rejected by the held-out test`;
    const dups = runs.filter((r) => r.duplicate_of).length;
    const blocked = runs.filter((r) => !r.ok && /Rejected|Guardrail/.test(r.error || "")).length;
    const crashed = runs.filter((r) => !r.ok).length - blocked;
    $("k-policies").textContent = fmt(runs.length);
    $("k-policies-sub").innerHTML = `<b>${ok.length}</b> simulated, <b>${dups}</b> repeats skipped, <b>${blocked + crashed}</b> blocked or crashed`;
    const recalls = runs.reduce((a, r) => a + (r.recalled_lessons || 0), 0);
    $("k-memory").textContent = fmt(recalls);
    $("k-memory-sub").innerHTML = `from <b>${fmt(M.lessons)}</b> lessons in Atlas Vector Search`;

    // Pipeline
    const rewrites = M.versions.filter((v) => v.parent_id).length;
    const steps = [
      ["Write a policy", `OpenAI ${esc(d.model)}`, runs.length, "policies written"],
      ["Check it's safe", "AST sandbox", blocked, "blocked"],
      ["Skip repeats", "Atlas Vector Search", dups, "repeats caught"],
      ["Simulate", "6,212-GPU replay", ok.length, "policies scored"],
      ["Learn", "Voyage AI to Atlas", M.lessons, "lessons stored"],
      ["Rewrite the harness", "Meta-agent", rewrites, "rewrites"],
      ["Keep if it generalizes", "Held-out workloads", kept, "versions kept"],
    ];
    $("steps").innerHTML = steps.map(([t, tech, n, what]) =>
      `<li><span class="s-title">${t}</span><span class="s-tech">${tech}</span><span class="s-count">${fmt(n)}</span><span class="s-what">${what}</span></li>`).join("");
    const rd = runs.filter((r) => r.raindrop_event_id).length;
    const sigs = runs.reduce((a, r) => a + (r.signals || []).length, 0);
    $("raindrop").innerHTML = rd
      ? `<b>Raindrop</b> traced ${fmt(rd)} generations (one interaction each) with ${fmt(sigs)} outcome signals, plus ${fmt(rewrites)} harness rewrites as edit signals. OpenAI tokens across both: ${fmt(M.tokens)}.`
      : `${fmt(M.tokens)} OpenAI tokens used.`;

    progress(runs, fgd, pack);
    wins(d, runs);
    ablation(d, fgd);
    log(M);
    usage(d);
    workloads(d);
  }

  // ---------- Raindrop usage ----------
  const SIGNAL_LABELS = [
    ["beat_best", "beat_best: new best policy"],
    ["beat_fgd", "beat_fgd: better than FGD"],
    ["regressed", "regressed: worse than best"],
    ["duplicate_policy", "duplicate_policy: repeat skipped"],
    ["sandbox_violation", "sandbox_violation: unsafe code blocked"],
    ["policy_crashed", "policy_crashed: failed in simulation"],
    ["guardrail_tripped", "guardrail_tripped: broke a harness limit"],
  ];
  function usageOf(M) {
    const runs = M.runs, u = M.usage || {};
    const count = (name) => runs.reduce((a, r) => a + (r.signals || []).filter((s) => s === name).length, 0);
    const child = M.versions.filter((v) => v.parent_id);
    const gens = runs.filter((r) => r.raindrop_event_id).length;
    const rewrites = u.rewrites != null ? u.rewrites : child.length;
    const spans = runs.some((r) => r.tool_spans != null) ? runs.reduce((a, r) => a + (r.tool_spans || 0), 0) : null;
    const sig = Object.fromEntries(SIGNAL_LABELS.map(([k]) => [k, count(k)]));
    const outcome = runs.reduce((a, r) => a + (r.signals || []).length, 0);
    const accepted = child.filter((v) => v.status === "active" || v.status === "retired").length;
    const rejected = child.filter((v) => v.status === "rejected").length;
    return {
      gens, rewrites, interactions: gens + rewrites, spans, sig, outcome, accepted, rejected,
      tg: u.tokens_generations, tr: u.tokens_rewrites, tt: M.tokens,
      perGen: u.tokens_generations != null && runs.length ? u.tokens_generations / runs.length : null,
    };
  }
  function usage(d) {
    const cols = [["Memory on", usageOf(d.main)]];
    if (d.ablation) cols.push(["Memory off", usageOf(d.ablation)]);
    const both = cols.length > 1;
    const v = (f) => cols.map(([, u]) => `<td class="num">${fmt(f(u))}</td>`).join("") +
      (both ? `<td class="num">${fmt(cols.every(([, u]) => f(u) != null) ? cols.reduce((a, [, u]) => a + f(u), 0) : null)}</td>` : "");
    const row = (label, f, cls = "") => `<tr class="${cls}"><td class="${cls === "sub" ? "sub" : ""}">${label}</td>${v(f)}</tr>`;
    const sub = (label, f) => `<tr><td class="sub">${label}</td>${v(f)}</tr>`;
    const group = (label) => `<tr class="group"><td colspan="${cols.length + 1 + (both ? 1 : 0)}">${label}</td></tr>`;
    $("usage").innerHTML = `<thead><tr><th></th>${cols.map(([n]) => `<th class="num">${n}</th>`).join("")}${both ? '<th class="num">Total</th>' : ""}</tr></thead><tbody>
      ${group("Interactions")}
      ${sub("Generations (policy written, checked, simulated)", (u) => u.gens)}
      ${sub("Harness rewrites (meta-agent)", (u) => u.rewrites)}
      ${row("All interactions", (u) => u.interactions, "total")}
      ${row("Tool spans (LLM, Atlas search, simulator, critic)", (u) => u.spans)}
      ${group("Outcome signals on generations")}
      ${SIGNAL_LABELS.map(([k, l]) => sub(l, (u) => u.sig[k])).join("")}
      ${row("All outcome signals", (u) => u.outcome, "total")}
      ${group("Signals on harness rewrites")}
      ${sub("harness_rewrite: edit, spec before and after", (u) => u.rewrites)}
      ${sub("harness_accepted: won on held-out", (u) => u.accepted)}
      ${sub("harness_rejected: lost on held-out", (u) => u.rejected)}
      ${group("OpenAI tokens")}
      ${sub("Writing and critiquing policies", (u) => u.tg)}
      ${sub("Rewriting the harness", (u) => u.tr)}
      ${row("All tokens", (u) => u.tt, "total")}
      ${sub("Average per generation", (u) => u.perGen)}
    </tbody>`;
  }

  // ---------- Training vs held-out ----------
  function workloads(d) {
    const best = (w) => {
      const rows = baselineRows.filter((r) => r.workload === w);
      if (!rows.length) return null;
      const by = {};
      rows.forEach((r) => { (by[r.policy] = by[r.policy] || []).push(r.unalloc_gpus); });
      const [p, xs] = Object.entries(by).map(([p, xs]) => [p, xs.reduce((a, b) => a + b, 0) / xs.length]).sort((a, b) => a[1] - b[1])[0];
      return `${esc(SHORT[p] || p)} <span class="muted">${fmt(xs)}</span>`;
    };
    const line = (w, kind) => {
      const pr = PROFILES[w] || [w, ""];
      return `<tr><td><span class="tag ${kind === "Training" ? "train" : "held"}">${kind}</span></td><td><b>${esc(w)}</b><br><span class="muted">${esc(pr[0])}</span></td><td class="num">${best(w) || "–"}</td></tr>`;
    };
    $("workloads").innerHTML = `<thead><tr><th>Split</th><th>Workload</th><th class="num">Best baseline, GPUs idle</th></tr></thead><tbody>
      ${(d.train || []).map((w) => line(w, "Training")).join("")}
      ${(d.heldout || []).map((w) => line(w, "Held-out")).join("")}
    </tbody>`;
  }

  // ---------- Progress chart ----------
  function progress(runs, fgd, pack) {
    const el = $("progress");
    const pts = runs.filter((r) => r.generation != null);
    if (!pts.length) { el.innerHTML = '<p class="foot">No generations yet.</p>'; return; }
    const W = 980, H = 300, m = { t: 14, r: 150, b: 32, l: 52 };
    const n = pts[pts.length - 1].generation;
    let b = Infinity;
    const stepPts = pts.map((r) => {
      if (r.ok && r.mean_unalloc != null) b = Math.min(b, r.mean_unalloc);
      return { g: r.generation, v: isFinite(b) ? b : null };
    });
    const vals = [...pts.filter((r) => r.ok).map((r) => r.mean_unalloc), fgd, pack].filter((v) => v != null);
    const lo = Math.min(...vals) * 0.94, hi = Math.max(...vals) * 1.03;
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const sx = (g) => m.l + ((g - 0.5) / Math.max(1, n)) * iw;
    const sy = (v) => m.t + ih - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo || 1)) * ih;
    const bands = [];
    for (const r of pts) {
      const last = bands[bands.length - 1];
      if (!last || last.v !== r.harness_version) bands.push({ v: r.harness_version, a: r.generation, z: r.generation });
      else last.z = r.generation;
    }
    const path = stepPts.filter((p) => p.v != null).map((p, i) => (i ? `H${sx(p.g).toFixed(1)}V${sy(p.v).toFixed(1)}` : `M${sx(p.g).toFixed(1)},${sy(p.v).toFixed(1)}`)).join("");
    const ticks = [lo, (lo + hi) / 2, hi];
    const labels = [];
    if (fgd != null) labels.push({ y: sy(fgd), t: `FGD (ATC '23) ${fmt(fgd)}`, c: "lbl" });
    if (pack != null) labels.push({ y: sy(pack), t: `GPU packing ${fmt(pack)}`, c: "lbl" });
    if (isFinite(b)) labels.push({ y: sy(b), t: `Agent ${fmt(b)}`, c: "lbl-strong" });
    labels.sort((a, c) => a.y - c.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 15);
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="GPUs left idle by the best policy so far, by generation">
      ${bands.map((bd, i) => `${i % 2 ? "" : `<rect class="band" x="${sx(bd.a - 0.5)}" y="${m.t}" width="${sx(bd.z + 0.5) - sx(bd.a - 0.5)}" height="${ih}"/>`}
        <text class="band-lbl" x="${sx(bd.a - 0.5) + 4}" y="${m.t + 12}">v${bd.v}</text>`).join("")}
      ${ticks.map((t) => `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${sy(t)}" y2="${sy(t)}"/><text class="ax" x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join("")}
      <text class="ax" x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">Generation (shaded bands: harness version)</text>
      ${fgd != null ? `<line class="ref" x1="${m.l}" x2="${m.l + iw}" y1="${sy(fgd)}" y2="${sy(fgd)}"/>` : ""}
      ${pack != null ? `<line class="ref-2" x1="${m.l}" x2="${m.l + iw}" y1="${sy(pack)}" y2="${sy(pack)}"/>` : ""}
      ${pts.map((r) => r.ok && r.mean_unalloc != null
        ? `<circle class="pt" cx="${sx(r.generation)}" cy="${sy(r.mean_unalloc)}" r="4.5" data-id="${r._id}"/>`
        : `<circle class="pt-fail" cx="${sx(r.generation)}" cy="${m.t + 8}" r="4.5" data-id="${r._id}"/>`).join("")}
      <path class="best" d="${path}"/>
      ${labels.map((l) => `<text class="${l.c}" x="${m.l + iw + 10}" y="${l.y + 4}">${l.t}</text>`).join("")}
    </svg>`;
    el.querySelectorAll("circle").forEach((c) => {
      const r = pts.find((x) => x._id === c.dataset.id);
      c.addEventListener("mousemove", (ev) => tip(ev, `<b>Gen ${r.generation}, harness v${r.harness_version}</b><br>` +
        (r.ok ? `${fmt(r.mean_unalloc, 1)} GPUs idle${r.duplicate_of ? " (repeat)" : ""}` : esc(r.error || "failed")) +
        (r.idea ? `<br>${esc(r.idea)}` : "")));
      c.addEventListener("mouseleave", untip);
    });
  }

  // ---------- Who wins where ----------
  function wins(d, runs) {
    const rows = [];
    const okRuns = runs.filter((r) => r.ok);
    for (const [w, split] of [...d.train.map((w) => [w, "train"]), ...d.heldout.map((w) => [w, "held-out"])]) {
      const seed = split === "train" ? 0 : 1;
      const cands = Object.keys(NAMES).map((p) => ({ p, v: base(p, w, seed) })).filter((c) => c.v != null);
      let agent = null;
      for (const r of okRuns) {
        const v = split === "train" ? r.per_workload && r.per_workload[w] : r.heldout && r.heldout.per_workload && r.heldout.per_workload[w];
        if (v != null && (agent == null || v < agent)) agent = v;
      }
      if (!cands.length && agent == null) continue;
      const bestBase = cands.length ? cands.reduce((a, c) => (c.v < a.v ? c : a)) : null;
      const agentWins = agent != null && bestBase != null && agent < bestBase.v;
      rows.push({ w, split, bestBase, agent, agentWins });
    }
    $("wins").innerHTML = `<thead><tr><th>Workload profile</th><th>Best published policy</th><th class="num">Agent</th><th>Winner</th></tr></thead>
      <tbody>${rows.map((r) => {
        const [name, ex] = PROFILES[r.w] || [r.w, ""];
        return `<tr><td><span class="wl">${esc(name)}</span>${r.split === "held-out" ? '<span class="tag">held-out</span>' : ""}<span class="ex">${esc(ex)}</span></td>
          <td>${r.bestBase ? `${NAMES[r.bestBase.p]} <span class="ex">${fmt(r.bestBase.v)}</span>` : "–"}</td>
          <td class="num">${fmt(r.agent)}</td>
          <td class="win ${r.agentWins ? "agent" : ""}">${r.agent == null || !r.bestBase ? '<span class="ex">not scored yet</span>' : r.agentWins ? `Agent <span class="ex">${fmt(r.bestBase.v - r.agent)} fewer</span>` : SHORT[r.bestBase.p]}</td></tr>`;
      }).join("")}</tbody>`;
  }

  // ---------- Memory on vs off ----------
  function ablation(d, fgd) {
    const el = $("ablation");
    const curve = (db) => {
      let b = null;
      return db.runs.map((r) => {
        if (r.ok && r.mean_unalloc != null && !r.duplicate_of) b = b == null ? r.mean_unalloc : Math.min(b, r.mean_unalloc);
        return { g: r.generation, v: b };
      }).filter((p) => p.g != null);
    };
    if (!d.ablation) {
      $("ablation-sub").textContent = "Waiting for the memory-off run";
      el.innerHTML = '<p class="foot">Starts when the memory-off run writes its first generation.</p>';
      return;
    }
    const on = curve(d.main), off = curve(d.ablation);
    const firstBeat = (c) => { const p = c.find((x) => x.v != null && fgd != null && x.v < fgd); return p ? p.g : null; };
    const fo = firstBeat(on), ff = firstBeat(off);
    $("ablation-sub").textContent = fgd == null ? "Best policy so far" :
      `Beat FGD at gen ${fo ?? "–"} with memory, ${ff ?? "–"} without`;
    const W = 560, H = 230, m = { t: 12, r: 92, b: 28, l: 46 };
    const n = Math.max(on.length ? on[on.length - 1].g : 1, off.length ? off[off.length - 1].g : 1);
    const vals = [...on, ...off].map((p) => p.v).filter((v) => v != null).concat(fgd != null ? [fgd] : []);
    if (!vals.length) { el.innerHTML = ""; return; }
    const lo = Math.min(...vals) * 0.94, hi = Math.max(...vals) * 1.03;
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const sx = (g) => m.l + ((g - 0.5) / n) * iw;
    const sy = (v) => m.t + ih - ((v - lo) / (hi - lo || 1)) * ih;
    const step = (c) => c.filter((p) => p.v != null).map((p, i) => (i ? `H${sx(p.g).toFixed(1)}V${sy(p.v).toFixed(1)}` : `M${sx(p.g).toFixed(1)},${sy(p.v).toFixed(1)}`)).join("");
    const end = (c) => { const p = c.filter((x) => x.v != null).pop(); return p ? p.v : null; };
    const labels = [{ y: end(on), t: `On ${fmt(end(on))}`, c: "lbl-strong" }, { y: end(off), t: `Off ${fmt(end(off))}`, c: "lbl" }]
      .concat(fgd != null ? [{ y: fgd, t: `FGD ${fmt(fgd)}`, c: "lbl" }] : []).filter((l) => l.y != null)
      .map((l) => ({ ...l, y: sy(l.y) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 15);
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Best policy so far with and without Atlas memory">
      ${[lo, hi].map((t) => `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${sy(t)}" y2="${sy(t)}"/><text class="ax" x="${m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join("")}
      <text class="ax" x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">Generation</text>
      ${fgd != null ? `<line class="ref" x1="${m.l}" x2="${m.l + iw}" y1="${sy(fgd)}" y2="${sy(fgd)}"/>` : ""}
      <path class="off-line" d="${step(off)}"/>
      <path class="best" d="${step(on)}"/>
      ${labels.map((l) => `<text class="${l.c}" x="${m.l + iw + 8}" y="${l.y + 4}">${l.t}</text>`).join("")}
    </svg>`;
  }

  // ---------- Run log ----------
  function log(M) {
    const ver = Object.fromEntries(M.versions.map((v) => [v._id, v]));
    const lines = [];
    for (const v of M.versions) {
      if (v.parent_id) lines.push({ t: v.created_at, c: "meta", s: `v${(ver[v.parent_id] || {}).version} → v${v.version}  ${v.rationale}` });
      if (v.decided_at && v.parent_id) {
        const keep = v.status !== "rejected";
        lines.push({ t: v.decided_at, c: keep ? "good" : "bad", s: `v${v.version} ${keep ? "kept" : "rejected"}  held-out ${fmt(v.heldout_score, 1)} vs v${(ver[v.parent_id] || {}).version} ${fmt((ver[v.parent_id] || {}).heldout_score, 1)}` });
      }
    }
    for (const r of M.runs) {
      const sig = (r.signals || []).filter((s) => s !== "regressed");
      lines.push({ t: r.created_at, c: (r.signals || []).some((s) => GOOD.has(s)) ? "good" : r.ok ? "" : "bad",
        s: `gen ${r.generation} v${r.harness_version}  ${r.ok ? `${fmt(r.mean_unalloc, 1)} idle` : "failed"}${r.duplicate_of ? "  repeat skipped" : ""}${sig.length ? "  [" + sig.join(", ") + "]" : ""}${r.recalled_lessons ? `  recalled ${r.recalled_lessons}` : ""}` });
    }
    lines.sort((a, b) => a.t - b.t);
    const show = lines.slice(-40);
    const clock = (t) => new Date(t * 1000).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    const el = $("log");
    el.innerHTML = show.map((l) => `<li class="${l.c}"><span class="t">${clock(l.t)}</span> ${esc(l.s)}</li>`).join("");
    el.scrollTop = el.scrollHeight;
  }

  // ---------- Tooltip ----------
  function tip(ev, html) {
    const t = $("tip"); t.innerHTML = html; t.hidden = false;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + t.offsetWidth > innerWidth - 8) x = ev.clientX - t.offsetWidth - 14;
    if (y + t.offsetHeight > innerHeight - 8) y = ev.clientY - t.offsetHeight - 14;
    t.style.left = x + "px"; t.style.top = y + "px";
  }
  function untip() { $("tip").hidden = true; }

  getJSON("baselines.json").then((b) => { baselineRows = b; }).catch(() => {}).finally(() => {
    load();
    setInterval(load, POLL_MS);
  });
})();
