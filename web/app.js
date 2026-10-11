// Backfill console: live cluster map, failure curve and leaderboard.

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) => Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  meta: null,
  rows: [],
  labels: {},
  gpuFree: null,   // flat per-GPU free milli, in map order
  live: [],        // [{x: arrived fraction, y: GPUs turned away}]
  running: null,   // EventSource
};

// ---------- Names people understand ----------
function workloadLabel(w) {
  if (w === "default") return "Production mix";
  const m = w.match(/^([a-z]+?)(\d+)$/);
  if (!m) return w;
  const [, kind, num] = m;
  const pct = kind === "cpu" ? Number(num) / 10 : Number(num);
  const names = {
    gpushare: `Fractional-GPU heavy (${pct}%)`,
    multigpu: `Multi-GPU heavy (${pct}%)`,
    gpuspec: `Pinned to GPU model (${pct}%)`,
    cpu: `CPU-only tasks, variant ${num}`,
  };
  return names[kind] || w;
}
const policyLabel = (p) => state.labels[p] || state.meta?.policies.find((x) => x.id === p)?.label || p;

// ---------- Cluster map ----------
const MAP = { cw: 6, ch: 10, gap: 1, nodeGap: 4, headerH: 22, groupGap: 14, rects: [] };

function layoutMap(width) {
  const rects = [];
  const headers = [];
  let y = 0;
  for (const g of state.meta.groups) {
    const gpus = g.nodes.reduce((a, b) => a + b, 0);
    headers.push({ y, text: `${g.model}`, sub: `${fmt(g.nodes.length)} nodes, ${fmt(gpus)} GPUs` });
    y += MAP.headerH;
    let x = 0, rowH = 0;
    for (const n of g.nodes) {
      const cols = Math.min(4, n), rows = Math.ceil(n / 4);
      const w = cols * (MAP.cw + MAP.gap) - MAP.gap;
      const h = rows * (MAP.ch + MAP.gap) - MAP.gap;
      if (x > 0 && x + w > width) { x = 0; y += rowH + MAP.nodeGap; rowH = 0; }
      for (let s = 0; s < n; s++) {
        rects.push({
          x: x + (s % 4) * (MAP.cw + MAP.gap),
          y: y + Math.floor(s / 4) * (MAP.ch + MAP.gap),
          model: g.model, slot: s,
        });
      }
      x += w + MAP.nodeGap;
      rowH = Math.max(rowH, h);
    }
    y += rowH + MAP.groupGap;
  }
  return { rects, headers, height: y - MAP.groupGap };
}

function drawMap() {
  if (!state.meta) return;
  const canvas = $("map");
  const width = canvas.clientWidth;
  const { rects, headers, height } = layoutMap(width);
  MAP.rects = rects;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const cFree = css("--slot-free"), cUsed = css("--used"), cFrag = css("--frag");
  ctx.textBaseline = "top";
  for (const h of headers) {
    ctx.fillStyle = css("--ink");
    ctx.font = '600 13px "IBM Plex Sans", sans-serif';
    ctx.fillText(h.text, 0, h.y + 2);
    const w = ctx.measureText(h.text).width;
    ctx.fillStyle = css("--ink-3");
    ctx.font = '400 12px "IBM Plex Sans", sans-serif';
    ctx.fillText(h.sub, w + 8, h.y + 3);
  }
  const free = state.gpuFree;
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i];
    const f = free ? free[i] : 1000;
    if (f >= 1000) {
      ctx.fillStyle = cFree; ctx.fillRect(r.x, r.y, MAP.cw, MAP.ch);
    } else if (f <= 0) {
      ctx.fillStyle = cUsed; ctx.fillRect(r.x, r.y, MAP.cw, MAP.ch);
    } else {
      const usedH = Math.max(1, Math.round(MAP.ch * (1 - f / 1000)));
      ctx.fillStyle = cFrag; ctx.fillRect(r.x, r.y, MAP.cw, MAP.ch - usedH);
      ctx.fillStyle = cUsed; ctx.fillRect(r.x, r.y + MAP.ch - usedH, MAP.cw, usedH);
    }
  }
}

function mapHover(ev) {
  const canvas = $("map");
  const b = canvas.getBoundingClientRect();
  const x = ev.clientX - b.left, y = ev.clientY - b.top;
  const i = MAP.rects.findIndex((r) => x >= r.x - 1 && x <= r.x + MAP.cw + 1 && y >= r.y - 1 && y <= r.y + MAP.ch + 1);
  if (i < 0) return hideTip();
  const r = MAP.rects[i];
  const f = state.gpuFree ? state.gpuFree[i] : 1000;
  const status = f >= 1000 ? "Idle" : f <= 0 ? "Fully allocated" : `${fmt(100 - f / 10)}% allocated, ${fmt(f / 10)}% left over`;
  showTip(ev, `<b>${r.model} GPU</b><br>${status}`);
}

// ---------- Tooltip ----------
function showTip(ev, html) {
  const t = $("tip");
  t.innerHTML = html; t.hidden = false;
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let left = ev.clientX + pad, top = ev.clientY + pad;
  if (left + w > window.innerWidth - 8) left = ev.clientX - w - pad;
  if (top + h > window.innerHeight - 8) top = ev.clientY - h - pad;
  t.style.left = `${left}px`; t.style.top = `${top}px`;
}
const hideTip = () => { $("tip").hidden = true; };

// ---------- Baseline curves ----------
function baselineCurves(workload) {
  const cap = state.meta.gpus;
  const by = {};
  for (const r of state.rows.filter((r) => r.workload === workload)) {
    (by[r.policy] ||= []).push(r);
  }
  return Object.entries(by).map(([policy, runs]) => {
    const pts = new Map();
    for (const run of runs) {
      for (const [x, a] of run.curve) {
        const k = x.toFixed(2);
        const failed = Math.max(0, (x - a) * cap);
        pts.set(k, (pts.get(k) || 0) + failed / runs.length);
      }
    }
    const end = runs.reduce((s, r) => s + r.unalloc_gpus, 0) / runs.length;
    pts.set("1.00", end);
    const points = [...pts].map(([k, y]) => ({ x: Number(k), y })).sort((a, b) => a.x - b.x);
    return { policy, points, end };
  });
}

// ---------- Failure chart ----------
const CH = { w: 348, h: 260, m: { t: 10, r: 92, b: 30, l: 40 }, x0: 0.8, x1: 1.0 };

function drawChart() {
  const el = $("chart");
  if (!state.meta) return;
  const workload = $("workload").value;
  const bases = baselineCurves(workload);
  const live = state.live;
  const allY = [...bases.flatMap((b) => b.points.filter((p) => p.x >= CH.x0).map((p) => p.y)), ...live.map((p) => p.y), 10];
  const yMax = niceMax(Math.max(...allY));
  const iw = CH.w - CH.m.l - CH.m.r, ih = CH.h - CH.m.t - CH.m.b;
  const sx = (x) => CH.m.l + ((x - CH.x0) / (CH.x1 - CH.x0)) * iw;
  const sy = (y) => CH.m.t + ih - (y / yMax) * ih;
  const path = (pts) => pts.filter((p) => p.x >= CH.x0 - 1e-9).map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");

  const yTicks = [0, yMax / 2, yMax];
  const xTicks = [0.8, 0.85, 0.9, 0.95, 1.0];
  const current = $("policy").value;

  // Direct labels at the right end, nudged apart so they never collide
  const labels = bases.map((b) => ({ text: policyLabel(b.policy), y: sy(b.end), hot: b.policy === current }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
  const overflow = labels.length ? labels[labels.length - 1].y - (CH.m.t + ih) : 0;
  if (overflow > 0) labels.forEach((l) => (l.y -= overflow));

  el.innerHTML = `
  <svg viewBox="0 0 ${CH.w} ${CH.h}" role="img" aria-label="GPU demand turned away versus arrived demand, per policy">
    <g class="grid">${yTicks.map((t) => `<line x1="${CH.m.l}" x2="${CH.m.l + iw}" y1="${sy(t)}" y2="${sy(t)}"/>`).join("")}</g>
    <g class="axis">
      ${yTicks.map((t) => `<text x="${CH.m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t)}</text>`).join("")}
      ${xTicks.map((t) => `<text x="${sx(t)}" y="${CH.h - 12}" text-anchor="middle">${fmt(t * 100)}%</text>`).join("")}
      <text x="${CH.m.l + iw / 2}" y="${CH.h}" text-anchor="middle">Arrived demand, % of cluster GPUs</text>
    </g>
    ${bases.map((b) => `<path class="base${b.policy === current ? " hot" : ""}" d="${path(b.points)}"/>`).join("")}
    ${live.length > 1 ? `<path class="live" d="${path(live)}"/>` : ""}
    ${labels.map((l) => `<text class="${l.hot ? "lbl-strong" : "lbl"}" x="${CH.m.l + iw + 8}" y="${l.y + 4}">${l.text}</text>`).join("")}
    <line class="cross" id="cross" y1="${CH.m.t}" y2="${CH.m.t + ih}" x1="-10" x2="-10"/>
    <rect id="hit" x="${CH.m.l}" y="${CH.m.t}" width="${iw}" height="${ih}" fill="transparent"/>
  </svg>`;

  const svg = el.querySelector("svg");
  const hit = el.querySelector("#hit");
  hit.addEventListener("mousemove", (ev) => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const x = Math.round((CH.x0 + ((p.x - CH.m.l) / iw) * (CH.x1 - CH.x0)) * 20) / 20;
    const cross = el.querySelector("#cross");
    cross.setAttribute("x1", sx(x)); cross.setAttribute("x2", sx(x));
    const rows = bases.map((b) => [policyLabel(b.policy), interp(b.points, x)])
      .sort((a, b) => a[1] - b[1]);
    if (live.length && live[live.length - 1].x >= x) rows.unshift(["This run", interp(live, x)]);
    showTip(ev, `<b>At ${fmt(x * 100)}% of capacity</b>` +
      rows.map(([n, v]) => `<div class="row"><span>${n}</span><span>${fmt(v)} GPUs</span></div>`).join(""));
  });
  hit.addEventListener("mouseleave", () => {
    hideTip();
    const cross = el.querySelector("#cross");
    cross.setAttribute("x1", -10); cross.setAttribute("x2", -10);
  });
}

function interp(pts, x) {
  if (!pts.length) return 0;
  if (x <= pts[0].x) return pts[0].y;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].x >= x) {
      const a = pts[i - 1], b = pts[i];
      return a.y + ((x - a.x) / (b.x - a.x || 1)) * (b.y - a.y);
    }
  }
  return pts[pts.length - 1].y;
}
function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((m) => m * p).find((c) => c >= v) || v;
}

// ---------- Leaderboard ----------
function drawBoard() {
  const t = $("board");
  if (!state.rows.length) {
    t.innerHTML = `<tbody><tr><td>No baseline results yet. Run <code>python scripts/run_baselines.py</code> to fill this table.</td></tr></tbody>`;
    return;
  }
  const workloads = [...new Set(state.rows.map((r) => r.workload))];
  const policies = [...new Set(state.rows.map((r) => r.policy))];
  const mean = (p, w) => {
    const rs = state.rows.filter((r) => r.policy === p && r.workload === w);
    return rs.length ? rs.reduce((s, r) => s + r.unalloc_gpus, 0) / rs.length : null;
  };
  const grid = policies.map((p) => {
    const vals = workloads.map((w) => mean(p, w));
    const ok = vals.filter((v) => v != null);
    return { p, vals, avg: ok.reduce((a, b) => a + b, 0) / ok.length };
  }).sort((a, b) => a.avg - b.avg);
  const best = workloads.map((_, j) => Math.min(...grid.map((g) => g.vals[j] ?? Infinity)));
  const bestAvg = Math.min(...grid.map((g) => g.avg));
  const sel = $("workload").value;
  t.innerHTML = `
    <thead><tr><th>Policy</th>${workloads.map((w) => `<th class="${w === sel ? "sel" : ""}">${workloadLabel(w)}</th>`).join("")}<th>Mean</th></tr></thead>
    <tbody>${grid.map((g) => `<tr><td>${policyLabel(g.p)}</td>${g.vals.map((v, j) =>
      `<td class="${v === best[j] ? "best" : ""} ${workloads[j] === sel ? "sel" : ""}">${v == null ? "" : fmt(v)}</td>`).join("")}
      <td class="${g.avg === bestAvg ? "best" : ""}">${fmt(g.avg)}</td></tr>`).join("")}</tbody>`;
}

// ---------- Run ----------
function setReadout(s, final = false) {
  const cap = state.meta.gpus;
  $("m-failed").textContent = fmt(final ? s.unalloc_gpus : s.failed_gpus);
  $("m-progress").textContent = `${fmt(Math.min(1, s.progress ?? 1) * 100)}%`;
  if (s.alloc != null) $("m-alloc").textContent = `${fmt(s.alloc)} of ${fmt(cap)}`;
  $("m-rejected").textContent = fmt(s.failed_pods);
  if (state.gpuFree) {
    const left = state.gpuFree.reduce((a, f) => a + (f > 0 && f < 1000 ? f : 0), 0) / 1000;
    $("m-frag").textContent = `${fmt(left, 1)} GPUs`;
  }
}

function compareLine(policy, workload, value) {
  const bases = baselineCurves(workload).filter((b) => b.policy !== policy);
  if (!bases.length) return "";
  const bestOther = bases.reduce((a, b) => (b.end < a.end ? b : a));
  const d = value - bestOther.end;
  const who = policyLabel(bestOther.policy);
  if (Math.abs(d) < 0.5) return `Level with ${who}, the best other policy`;
  return d < 0
    ? `${fmt(-d)} fewer than ${who}, the best other policy`
    : `${fmt(d)} more than ${who}, the best other policy`;
}

function run(ev) {
  ev?.preventDefault();
  if (state.running) state.running.close();
  const policy = $("policy").value, workload = $("workload").value, seed = $("seed").value || 0;
  state.live = []; state.gpuFree = null;
  $("m-compare").textContent = "";
  $("run").disabled = true;
  $("status").textContent = `Running ${policyLabel(policy)} on ${workloadLabel(workload).toLowerCase()}…`;
  drawMap(); drawChart(); drawBoard();
  let raf = 0;
  const es = new EventSource(`/api/run?${new URLSearchParams({ policy, workload, seed })}`);
  state.running = es;
  es.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.type === "frame") {
      state.gpuFree = d.gpu_free;
      state.live.push({ x: d.progress, y: d.failed_gpus });
      setReadout(d);
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => { drawMap(); drawChart(); });
    } else if (d.type === "done") {
      es.close(); state.running = null;
      state.live.push({ x: 1.0, y: d.unalloc_gpus });
      setReadout({ ...d, progress: 1, alloc: d.capacity_gpus - d.unalloc_gpus }, true);
      $("m-compare").textContent = compareLine(policy, workload, d.unalloc_gpus);
      $("status").textContent = `Finished in ${fmt(d.secs, 1)} s.`;
      $("run").disabled = false;
      drawMap(); drawChart();
    } else if (d.type === "error") {
      es.close(); state.running = null;
      $("status").textContent = `The run stopped: ${d.message}`;
      $("run").disabled = false;
    }
  };
  es.onerror = () => {
    if (!state.running) return;
    es.close(); state.running = null;
    $("status").textContent = "Lost the connection to the simulator. Check that the server is still running, then run again.";
    $("run").disabled = false;
  };
}

// ---------- Boot ----------
async function boot() {
  const [meta, base] = await Promise.all([
    fetch("/api/meta").then((r) => r.json()),
    fetch("/api/baselines").then((r) => r.json()),
  ]);
  state.meta = meta;
  state.rows = base.rows || [];
  state.labels = base.labels || {};
  $("facts").textContent = `${fmt(meta.nodes)} nodes and ${fmt(meta.gpus)} GPUs across ${meta.models} GPU models, from Alibaba's 2023 production trace`;

  const order = ["default", "gpushare40", "multigpu20", "gpuspec33"];
  const ws = [...meta.workloads].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99) || a.localeCompare(b));
  $("workload").innerHTML = ws.map((w) => `<option value="${w}">${workloadLabel(w)}</option>`).join("");
  $("policy").innerHTML = meta.policies.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
  $("policy").value = "fgd";

  $("controls").addEventListener("submit", run);
  $("workload").addEventListener("change", () => { drawChart(); drawBoard(); });
  $("policy").addEventListener("change", drawChart);
  $("map").addEventListener("mousemove", mapHover);
  $("map").addEventListener("mouseleave", hideTip);
  let rt = 0;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { drawMap(); }, 120); });
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${b.dataset.view}`));
    if (b.dataset.view === "arena") drawMap();
  }));
  await document.fonts?.ready;
  drawMap(); drawChart(); drawBoard();
}

boot().catch((e) => { $("status").textContent = `Could not load the cluster: ${e.message}`; });
