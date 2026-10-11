// Harness view: progress chart, lineage of harness versions, and per-version detail.
// Data comes from /api/harness; /api/events pushes updates (an Atlas change stream when on Atlas).

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) => (n == null ? "–" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const H = { full: null, data: null, selected: null, via: "Waiting for updates…", cutoff: null, timer: 0 };
const GOOD = new Set(["beat_best", "beat_fgd", "harness_accepted"]);
const SIGNAL_LABEL = {
  beat_best: "Beat best so far", beat_fgd: "Beat FGD", regressed: "Regressed",
  duplicate_policy: "Duplicate caught", sandbox_violation: "Sandbox blocked",
  policy_crashed: "Crashed", guardrail_tripped: "Guardrail tripped",
};

async function load() {
  try {
    H.full = await fetch("/api/harness").then((r) => r.json());
    H.data = view(H.full);
  } catch (e) {
    $("sponsors").textContent = `Could not load harness data: ${e.message}`;
    return;
  }
  render();
}

// During a replay, show only what existed at the replay clock.
function view(full) {
  if (H.cutoff == null) return full;
  const t = H.cutoff;
  const versions = full.versions.filter((v) => v.created_at <= t).map((v) =>
    (v.decided_at && v.decided_at > t) ? { ...v, status: "candidate", heldout_score: null } : v);
  const runs = full.runs.filter((r) => r.created_at <= t);
  const maxGen = Math.max(0, ...runs.map((r) => r.generation || 0));
  const lessons = Object.fromEntries(Object.entries(full.lessons).filter(([, l]) => (l.generation ?? 0) <= maxGen));
  const tokens = full.runs.length ? Math.round(full.tokens * runs.length / full.runs.length) : 0;
  return { ...full, versions, runs, lessons, tokens };
}

function timeline(full) {
  const ts = [...full.runs.map((r) => r.created_at), ...full.versions.map((v) => v.created_at),
    ...full.versions.map((v) => v.decided_at).filter(Boolean)].sort((a, b) => a - b);
  return [...new Set(ts)];
}

function startReplay() {
  if (!H.full) return;
  const steps = timeline(H.full);
  if (!steps.length) return;
  clearInterval(H.timer);
  let i = 0;
  $("replay-btn").setAttribute("aria-pressed", "true");
  $("replay-btn").textContent = "Stop replay";
  const tick = () => {
    H.cutoff = steps[i];
    H.data = view(H.full);
    $("replay-status").textContent = `Replaying: step ${i + 1} of ${steps.length}`;
    render();
    if (++i >= steps.length) stopReplay(true);
  };
  tick();
  H.timer = setInterval(tick, 1600);
}

function stopReplay(finished = false) {
  clearInterval(H.timer);
  H.cutoff = null;
  if (H.full) H.data = H.full;
  $("replay-btn").setAttribute("aria-pressed", "false");
  $("replay-btn").textContent = "Replay the run";
  $("replay-status").textContent = finished ? "Replay finished." : "";
  render();
}

function render() {
  const d = H.data;
  renderSponsors(d);
  renderHeadline(d);
  const has = d.versions.length > 0;
  $("h-empty").hidden = has;
  $("h-body").hidden = !has;
  if (!has) return;
  if (!H.selected || !d.versions.find((v) => v._id === H.selected)) {
    H.selected = (d.versions.find((v) => v.status === "active") || d.versions[d.versions.length - 1])._id;
  }
  renderProgress(d);
  if (H.cutoff == null) renderAblation(); else $("ablation-wrap").hidden = true;
  renderRunLog(d);
  renderLineage(d);
  renderDetail(d);
}

function renderSponsors(d) {
  const recalls = d.runs.reduce((a, r) => a + (r.recalled_lessons?.length || 0), 0);
  const dups = d.runs.filter((r) => r.duplicate_of).length;
  const replay = d.backend === "replay";
  const atlas = d.backend === "atlas" || replay;
  const rdOn = d.raindrop || (replay && d.runs.some((r) => r.raindrop_event_id));
  const sigs = d.runs.reduce((a, r) => a + (r.signals?.length || 0), 0);
  $("sponsors").innerHTML = `
    <div class="sponsor ${atlas ? "on" : ""}">
      <b>${replay ? "Recorded from MongoDB Atlas" : atlas ? `MongoDB Atlas, database ${esc(d.database)}` : "Local store (Atlas not connected)"}</b>
      <span>${fmt(d.versions.length)} harness versions, ${fmt(d.runs.length)} runs, ${fmt(Object.keys(d.lessons).length)} lessons</span>
      <span>Vector search: ${fmt(recalls)} lessons recalled, ${fmt(dups)} duplicate policies caught</span>
    </div>
    <div class="sponsor ${rdOn ? "on" : ""}">
      <b>${replay ? (rdOn ? "Recorded in Raindrop" : "Raindrop was not connected") : rdOn ? "Raindrop is recording" : "Raindrop not connected"}</b>
      ${rdOn
        ? `<span>${fmt(d.runs.length)} generation interactions, ${fmt(sigs)} outcome signals</span>
           <span>Every event tagged with its harness version</span>`
        : `<span>Add RAINDROP_WRITE_KEY to .env to record each generation</span>`}
    </div>
    <div class="sponsor on">
      <b>${esc(d.model)}</b>
      <span>${fmt(d.tokens)} tokens used so far</span>
      <span id="live-via">${esc(H.via)}</span>
    </div>`;
}

// ---------- Headline ----------
function renderHeadline(d) {
  const el = $("h-headline");
  const ok = d.runs.filter((r) => r.ok && r.mean_unalloc != null);
  const fgd = d.reference?.fgd;
  if (!ok.length || fgd == null) { el.hidden = true; return; }
  const best = ok.reduce((a, r) => (r.mean_unalloc < a.mean_unalloc ? r : a));
  const diff = fgd - best.mean_unalloc;
  el.hidden = false;
  el.innerHTML = `
    <span class="hl-value">${fmt(best.mean_unalloc)}</span>
    <span class="hl-label">GPUs left unallocated by the agent's best policy, mean of ${d.train?.length || 4} production workloads</span>
    <span class="hl-compare">${diff > 0
      ? `${fmt(diff)} fewer than FGD (USENIX ATC '23) at ${fmt(fgd)}. Found in generation ${best.generation}, harness v${best.harness_version}.`
      : `FGD (USENIX ATC '23) is at ${fmt(fgd)}, ${fmt(-diff)} ahead. Best so far is from generation ${best.generation}.`}</span>`;
}

// ---------- Run log ----------
function renderRunLog(d) {
  const ver = Object.fromEntries(d.versions.map((v) => [v._id, v]));
  const lines = [];
  for (const v of d.versions) {
    if (v.parent_id) lines.push({ t: v.created_at, cls: "meta",
      text: `meta-agent: v${ver[v.parent_id]?.version ?? "?"} -> v${v.version}: ${v.rationale}` });
    else lines.push({ t: v.created_at, cls: "meta", text: `harness v${v.version} created: ${v.rationale}` });
    if (v.decided_at && (v.status === "active" || v.status === "rejected" || v.status === "retired")) {
      const kept = v.status !== "rejected";
      const parent = ver[v.parent_id];
      lines.push({ t: v.decided_at, cls: kept ? "good" : "bad",
        text: `  v${v.version} ${kept ? "kept" : "rejected"}: held-out ${fmt(v.heldout_score, 1)} vs parent v${parent?.version ?? "?"} ${fmt(parent?.heldout_score, 1)}` });
    }
  }
  for (const r of d.runs) {
    const sig = r.signals?.length ? `  [${r.signals.join(", ")}]` : "";
    const body = r.ok ? `${fmt(r.mean_unalloc, 1)} unallocated` : `failed: ${(r.error || "").slice(0, 140)}`;
    const recall = r.recalled_lessons?.length ? `  (recalled ${r.recalled_lessons.length} lessons from Atlas)` : "";
    lines.push({ t: r.created_at, cls: r.signals?.some((s) => GOOD.has(s)) ? "good" : r.ok ? "" : "bad",
      text: `  gen ${r.generation} h${r.harness_version}: ${body}${sig}${recall}` });
  }
  lines.sort((a, b) => a.t - b.t);
  const clock = (t) => new Date(t * 1000).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" });
  const el = $("runlog");
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  el.innerHTML = lines.map((l) => `<li class="${l.cls}"><span class="t">${clock(l.t)}</span> ${esc(l.text)}</li>`).join("");
  if (atBottom || H.cutoff != null) el.scrollTop = el.scrollHeight;
}

// ---------- Progress chart ----------
function renderProgress(d) {
  const W = 1100, Ht = 260, m = { t: 14, r: 120, b: 34, l: 52 };
  const runs = d.runs.filter((r) => r.generation != null).sort((a, b) => a.generation - b.generation);
  if (!runs.length) { $("progress").innerHTML = '<p class="panel-note">No generations yet.</p>'; return; }
  const n = runs[runs.length - 1].generation;
  let best = Infinity;
  const steps = runs.map((r) => {
    if (r.ok && r.mean_unalloc != null) best = Math.min(best, r.mean_unalloc);
    return { g: r.generation, best: Number.isFinite(best) ? best : null };
  });
  const okVals = runs.filter((r) => r.ok).map((r) => r.mean_unalloc);
  const refs = Object.entries(d.reference || {});
  const vals = [...okVals, ...refs.map(([, v]) => v)];
  const yMin = Math.max(0, Math.min(...vals) * 0.9), yMax = Math.max(...vals) * 1.05;
  const iw = W - m.l - m.r, ih = Ht - m.t - m.b;
  const sx = (g) => m.l + ((g - 0.5) / Math.max(1, n)) * iw;
  const sy = (v) => m.t + ih - ((Math.min(v, yMax) - yMin) / (yMax - yMin || 1)) * ih;

  // Harness version bands
  const bands = [];
  for (const r of runs) {
    const last = bands[bands.length - 1];
    if (!last || last.v !== r.harness_version) bands.push({ v: r.harness_version, g0: r.generation, g1: r.generation });
    else last.g1 = r.generation;
  }
  const refLabel = { fgd: "FGD (ATC '23)", gpu_packing: "GPU packing", best_fit: "Best-fit" };
  const path = steps.filter((s) => s.best != null)
    .map((s, i) => `${i ? "H" + sx(s.g).toFixed(1) + "V" : "M" + sx(s.g).toFixed(1) + ","}${sy(s.best).toFixed(1)}`).join("");
  const yt = [yMin, (yMin + yMax) / 2, yMax];
  $("progress").innerHTML = `
  <svg viewBox="0 0 ${W} ${Ht}" role="img" aria-label="Best unallocated GPUs by generation">
    ${bands.map((b, i) => `<rect class="band" x="${sx(b.g0 - 0.5)}" y="${m.t}" width="${sx(b.g1 + 0.5) - sx(b.g0 - 0.5)}" height="${ih}" opacity="${i % 2 ? 0 : 1}"/>
      <text class="band-lbl" x="${sx(b.g0 - 0.5) + 4}" y="${m.t + 12}">v${b.v}</text>`).join("")}
    <g class="axis">
      ${yt.map((t) => `<text x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join("")}
      <text x="${m.l + iw / 2}" y="${Ht - 4}" text-anchor="middle">Generation</text>
      ${runs.filter((r, i) => n <= 20 || i % Math.ceil(n / 20) === 0).map((r) => `<text x="${sx(r.generation)}" y="${Ht - 18}" text-anchor="middle">${r.generation}</text>`).join("")}
    </g>
    ${refs.map(([k, v]) => `<line class="ref" x1="${m.l}" x2="${m.l + iw}" y1="${sy(v)}" y2="${sy(v)}"/>
      <text class="lbl" x="${m.l + iw + 8}" y="${sy(v) + 4}">${refLabel[k] || k} ${fmt(v)}</text>`).join("")}
    ${runs.map((r) => r.ok
      ? `<circle class="pt" cx="${sx(r.generation)}" cy="${sy(r.mean_unalloc)}" r="4" data-run="${r._id}"/>`
      : `<circle class="pt-fail" cx="${sx(r.generation)}" cy="${m.t + 6}" r="4" data-run="${r._id}"/>`).join("")}
    <path class="best" d="${path}"/>
    ${Number.isFinite(best) ? `<text class="lbl-strong" x="${m.l + iw + 8}" y="${sy(best) + 4}">Best ${fmt(best, 1)}</text>` : ""}
  </svg>`;
  $("progress").querySelectorAll("circle").forEach((c) => {
    c.addEventListener("mousemove", (ev) => {
      const r = runs.find((x) => x._id === c.dataset.run);
      tip(ev, `<b>Generation ${r.generation}, harness v${r.harness_version}</b><br>` +
        (r.ok ? `${fmt(r.mean_unalloc, 1)} unallocated GPUs` : esc((r.error || "").slice(0, 160))) +
        (r.signals?.length ? `<br>${r.signals.map((s) => SIGNAL_LABEL[s] || s).join(", ")}` : ""));
    });
    c.addEventListener("mouseleave", untip);
    c.addEventListener("click", () => { const r = runs.find((x) => x._id === c.dataset.run); H.selected = r.harness_id; render(); });
  });
}

// ---------- Ablation: memory on vs off ----------
async function renderAblation() {
  let a;
  try { a = await fetch("/api/ablation").then((r) => r.json()); } catch { return; }
  const wrap = $("ablation-wrap");
  if (!a.memory_off || !a.memory_on?.points?.length) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const on = a.memory_on, off = a.memory_off;
  const when = (c) => (c.beat_fgd_at ? `beat FGD at generation ${c.beat_fgd_at}` : "has not beaten FGD yet");
  $("ablation-note").textContent =
    `Memory on: best ${fmt(on.best, 1)} after ${on.generations} generations, ${when(on)}, ${on.duplicates} duplicate${on.duplicates === 1 ? "" : "s"} skipped. ` +
    `Memory off: best ${fmt(off.best, 1)} after ${off.generations} generations, ${when(off)}.`;
  const W = 1100, Ht = 220, m = { t: 12, r: 120, b: 30, l: 52 };
  const n = Math.max(on.generations, off.generations, 1);
  const vals = [...on.points, ...off.points].map((p) => p.best).filter((v) => v != null);
  if (a.fgd != null) vals.push(a.fgd);
  if (!vals.length) { $("ablation").innerHTML = ""; return; }
  const yMin = Math.min(...vals) * 0.92, yMax = Math.max(...vals) * 1.04;
  const iw = W - m.l - m.r, ih = Ht - m.t - m.b;
  const sx = (g) => m.l + ((g - 0.5) / n) * iw;
  const sy = (v) => m.t + ih - ((v - yMin) / (yMax - yMin || 1)) * ih;
  const step = (pts) => pts.filter((p) => p.best != null)
    .map((p, i) => (i ? `H${sx(p.g).toFixed(1)}V${sy(p.best).toFixed(1)}` : `M${sx(p.g).toFixed(1)},${sy(p.best).toFixed(1)}`)).join("");
  const endLbl = (c, cls, name) => c.best == null ? "" :
    `<text class="${cls === "on" ? "lbl-strong" : "lbl"}" x="${m.l + iw + 8}" y="${sy(c.best) + 4}">${name} ${fmt(c.best, 1)}</text>`;
  $("ablation").innerHTML = `
  <svg viewBox="0 0 ${W} ${Ht}" role="img" aria-label="Best unallocated GPUs by generation, with and without Atlas memory">
    <g class="axis">
      ${[yMin, (yMin + yMax) / 2, yMax].map((t) => `<text x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join("")}
      <text x="${m.l + iw / 2}" y="${Ht - 2}" text-anchor="middle">Generation</text>
    </g>
    ${a.fgd != null ? `<line class="ref" x1="${m.l}" x2="${m.l + iw}" y1="${sy(a.fgd)}" y2="${sy(a.fgd)}"/>
      <text class="lbl" x="${m.l + iw + 8}" y="${sy(a.fgd) + 4}">FGD ${fmt(a.fgd)}</text>` : ""}
    <path class="off" d="${step(off.points)}"/>
    <path class="on" d="${step(on.points)}"/>
    ${endLbl(on, "on", "On")}${endLbl(off, "off", "Off")}
  </svg>`;
}

// ---------- Lineage ----------
function renderLineage(d) {
  const statusText = { active: "Active", rejected: "Rejected", retired: "Kept, superseded", candidate: "Running" };
  $("lineage").innerHTML = d.versions.slice().reverse().map((v) => `
    <li class="${v.status}">
      <button aria-current="${v._id === H.selected}" data-id="${v._id}">
        <span class="row1"><span>Version ${v.version}${v.parent_id ? ` <small>from v${verOf(d, v.parent_id)}</small>` : ""}</span>
          <span class="status ${v.status}">${statusText[v.status] || v.status}</span></span>
        <span class="nums">Train ${fmt(v.score, 1)}, held-out ${fmt(v.heldout_score, 1)}</span>
        <span class="why">${esc(v.rationale)}</span>
      </button>
    </li>`).join("");
  $("lineage").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { H.selected = b.dataset.id; render(); }));
}
const verOf = (d, id) => d.versions.find((v) => v._id === id)?.version ?? "?";

// ---------- Detail ----------
function flatten(obj, pre = "") {
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    const key = pre ? `${pre}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) Object.assign(out, flatten(v, key));
    else out[key] = v;
  }
  return out;
}

function specDiff(parent, spec) {
  const a = flatten(parent), b = flatten(spec);
  const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((k) => k !== "rules" && JSON.stringify(a[k]) !== JSON.stringify(b[k]));
  const items = keys.map((k) => `<li><code>${esc(k)}</code>: <del>${esc(JSON.stringify(a[k]))}</del> to <b>${esc(JSON.stringify(b[k]))}</b></li>`);
  const ra = new Set(parent?.rules || []), rb = new Set(spec?.rules || []);
  for (const r of rb) if (!ra.has(r)) items.push(`<li>Added rule: <b>${esc(r)}</b></li>`);
  for (const r of ra) if (!rb.has(r)) items.push(`<li>Removed rule: <del>${esc(r)}</del></li>`);
  return items.join("") || "<li>No changes</li>";
}

function renderDetail(d) {
  const v = d.versions.find((x) => x._id === H.selected);
  if (!v) return;
  const parent = d.versions.find((x) => x._id === v.parent_id);
  const runs = d.runs.filter((r) => r.harness_id === v._id);
  const train = d.train || [];
  $("detail").innerHTML = `
    <h2>Version ${v.version}</h2>
    <p class="rationale">${esc(v.rationale)}</p>
    ${v.author ? `<p class="panel-note">Written by the ${esc(v.author)}${v.raindrop_event_id ? `. Raindrop event ${esc(v.raindrop_event_id)}` : ""}</p>` : ""}
    ${parent ? `<h3>What changed from version ${parent.version}</h3><ul class="diff">${specDiff(parent.spec, v.spec)}</ul>` :
      `<h3>Starting harness</h3><ul class="diff">${(v.spec.rules || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`}
    ${v.validator_notes?.length ? `<p class="panel-note">Validator: ${esc(v.validator_notes.join("; "))}</p>` : ""}
    <h3>Generations under this version</h3>
    ${runs.length ? `<div class="table-scroll"><table class="gens">
      <thead><tr><th>Gen</th><th>Idea and outcome</th>${train.map((w) => `<th class="num">${esc(w)}</th>`).join("")}<th class="num">Mean</th></tr></thead>
      <tbody>${runs.map((r) => genRow(d, r, train)).join("")}</tbody></table></div>` : '<p class="panel-note">No generations yet.</p>'}`;
}

function genRow(d, r, train) {
  const chips = (r.signals || []).map((s) =>
    `<span class="chip ${GOOD.has(s) ? "good" : ["regressed", "sandbox_violation", "policy_crashed", "guardrail_tripped"].includes(s) ? "bad" : ""}">${SIGNAL_LABEL[s] || s}</span>`).join("");
  const recalled = (r.recalled_lessons || []).map((id) => d.lessons[id]?.text).filter(Boolean);
  const used = [];
  if (recalled.length) used.push(`<b>Atlas vector search</b> recalled ${recalled.length} lesson${recalled.length > 1 ? "s" : ""}: ${recalled.map((t) => `“${esc(t)}”`).join(" ")}`);
  if (r.duplicate_of) used.push(`<b>Atlas vector search</b> matched an earlier policy, so its score was reused`);
  if (r.raindrop_event_id) used.push(`<b>Raindrop</b> event ${esc(r.raindrop_event_id)}`);
  const cells = train.map((w) => `<td class="num">${r.per_workload?.[w] != null ? fmt(r.per_workload[w]) : ""}</td>`).join("");
  return `<tr>
    <td class="num">${r.generation}</td>
    <td>${esc(r.idea || "").slice(0, 280) || "<i>No description</i>"}
      ${r.ok ? "" : `<div class="used-by">${esc((r.error || "").slice(0, 240))}</div>`}
      <div class="chips">${chips}</div>
      ${used.map((u) => `<div class="used-by">${u}</div>`).join("")}
      ${r.lessons?.length ? `<div class="used-by">Lessons written: ${r.lessons.map((t) => esc(t)).join(" ")}</div>` : ""}
      ${r.code ? `<details class="code"><summary>Policy code</summary><pre><code>${esc(r.code)}</code></pre></details>` : ""}
    </td>${cells}<td class="num"><b>${r.ok ? fmt(r.mean_unalloc, 1) : "–"}</b></td></tr>`;
}

// ---------- Tooltip (shares #tip with the cluster view) ----------
function tip(ev, html) {
  const t = $("tip"); t.innerHTML = html; t.hidden = false;
  const w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
  if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
  t.style.left = `${x}px`; t.style.top = `${y}px`;
}
const untip = () => { $("tip").hidden = true; };

// ---------- Live updates ----------
function listen() {
  const es = new EventSource("/api/events");
  let t = 0;
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    const via = $("live-via");
    if (ev.type === "ready") {
      H.via = ev.via === "atlas-change-stream" ? "Live via Atlas change stream"
        : ev.via === "recording" ? "Recorded run, replayed offline" : "Live via polling";
      if (via) via.textContent = H.via;
    }
    if (ev.type === "changed" && H.cutoff == null) { clearTimeout(t); t = setTimeout(load, 400); }
  };
  es.onerror = () => { H.via = "Live updates paused, reconnecting…"; const via = $("live-via"); if (via) via.textContent = H.via; };
}

$("replay-btn").addEventListener("click", () => (H.cutoff == null ? startReplay() : stopReplay()));
load().then(() => {
  if (H.full?.backend === "replay" && H.full.recorded_at) {
    $("replay-status").textContent = `Recorded run from ${new Date(H.full.recorded_at * 1000).toLocaleString("en-US")}`;
  }
  listen();
});
