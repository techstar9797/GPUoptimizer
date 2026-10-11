// Cluster tab: plays back recorded fill-ups of all 6,212 GPUs, one recording per policy.
(function () {
  const $ = (id) => document.getElementById(id);
  const fmt = (n, d = 0) => (n == null ? "–" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }));
  const CELL = { w: 6, h: 10, gap: 1, node: 4, head: 20, group: 12 };
  let index = null, current = null, frame = 0, timer = 0, rects = [];
  const cache = {};

  // Tabs
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $("view-overview").hidden = b.dataset.view !== "overview";
    $("view-cluster").hidden = b.dataset.view !== "cluster";
    if (b.dataset.view === "cluster") init();
  }));

  async function init() {
    if (index) { draw(); return; }
    try {
      index = await fetch("cluster/index.json", { cache: "no-store" }).then((r) => r.json());
    } catch (e) {
      $("cl-foot").textContent = "No cluster recordings yet. Run python scripts/export_cluster.py.";
      return;
    }
    const sorted = [...index.policies].sort((a, b) => a.unalloc - b.unalloc);
    $("cl-pick").innerHTML = index.policies.map((p) =>
      `<button role="radio" aria-checked="false" data-id="${p.id}">${p.label}<span>${fmt(p.unalloc)} GPUs idle</span></button>`).join("");
    $("cl-pick").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => select(b.dataset.id, true)));
    const max = Math.max(...index.policies.map((p) => p.unalloc));
    $("cl-bars").innerHTML = sorted.map((p) =>
      `<li data-id="${p.id}"><div class="row1"><span>${p.label}</span><span>${fmt(p.unalloc)}</span></div><div class="track"><div class="fill" style="width:${(p.unalloc / max * 100).toFixed(1)}%"></div></div></li>`).join("");
    const w = index.policies[0];
    $("cl-foot").textContent = `Recorded with the simulator on Alibaba's 2023 trace, ${w.workload === "default" ? "production" : w.workload} workload, seed ${w.seed}. Tasks arrive until requested GPUs equal 100% of the cluster; nothing departs.`;
    $("cl-play").addEventListener("click", () => (timer ? stop() : play()));
    $("cl-scrub").addEventListener("input", (e) => { stop(); frame = +e.target.value; draw(); });
    window.addEventListener("resize", () => { rects = []; draw(); });
    await select((index.policies.find((p) => p.id === "agent") || index.policies[0]).id, false);
  }

  async function select(id, autoplay) {
    stop();
    $("cl-pick").querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.id === id)));
    $("cl-bars").querySelectorAll("li").forEach((li) => li.classList.toggle("sel", li.dataset.id === id));
    current = cache[id] || (cache[id] = await fetch(`cluster/${id}.json`).then((r) => r.json()));
    $("cl-scrub").max = current.frames.length - 1;
    frame = autoplay ? 0 : current.frames.length - 1;
    draw();
    if (autoplay) play();
  }

  function play() {
    if (!current) return;
    if (frame >= current.frames.length - 1) frame = 0;
    $("cl-play").textContent = "Pause";
    timer = setInterval(() => {
      frame++;
      draw();
      if (frame >= current.frames.length - 1) stop();
    }, 110);
  }
  function stop() { clearInterval(timer); timer = 0; $("cl-play").textContent = "Play"; }

  function layout(width) {
    const out = [], heads = [];
    let y = 0;
    for (const g of index.groups) {
      heads.push({ y, model: g.model, nodes: g.nodes.length, gpus: g.nodes.reduce((a, b) => a + b, 0) });
      y += CELL.head;
      let x = 0, rowH = 0;
      for (const n of g.nodes) {
        const cols = Math.min(4, n), rows = Math.ceil(n / 4);
        const w = cols * (CELL.w + CELL.gap) - CELL.gap, h = rows * (CELL.h + CELL.gap) - CELL.gap;
        if (x > 0 && x + w > width) { x = 0; y += rowH + CELL.node; rowH = 0; }
        for (let s = 0; s < n; s++) out.push({ x: x + (s % 4) * (CELL.w + CELL.gap), y: y + Math.floor(s / 4) * (CELL.h + CELL.gap) });
        x += w + CELL.node; rowH = Math.max(rowH, h);
      }
      y += rowH + CELL.group;
    }
    return { out, heads, height: y - CELL.group };
  }

  function draw() {
    if (!current) return;
    const f = current.frames[Math.min(frame, current.frames.length - 1)];
    const canvas = $("cl-canvas");
    const width = canvas.clientWidth || canvas.parentElement.clientWidth;
    const L = layout(width);
    rects = L.out;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.height = L.height + "px";
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(L.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, L.height);
    ctx.textBaseline = "top";
    for (const h of L.heads) {
      ctx.fillStyle = "#eef2f6"; ctx.font = '600 13px "IBM Plex Sans", sans-serif';
      ctx.fillText(h.model, 0, h.y + 2);
      const tw = ctx.measureText(h.model).width;
      ctx.fillStyle = "#6f7b88"; ctx.font = '400 12px "IBM Plex Sans", sans-serif';
      ctx.fillText(`${fmt(h.nodes)} nodes, ${fmt(h.gpus)} GPUs`, tw + 8, h.y + 3);
    }
    const lv = index.levels, g = f.g;
    let left = 0;
    for (let i = 0; i < rects.length; i++) {
      const r = rects[i], q = g.charCodeAt(i) - 65;
      if (q >= lv) { ctx.fillStyle = "#2a3440"; ctx.fillRect(r.x, r.y, CELL.w, CELL.h); }
      else if (q <= 0) { ctx.fillStyle = "#5aa9e6"; ctx.fillRect(r.x, r.y, CELL.w, CELL.h); }
      else {
        left += q / lv;
        const usedH = Math.max(1, Math.round(CELL.h * (1 - q / lv)));
        ctx.fillStyle = "#f2b13d"; ctx.fillRect(r.x, r.y, CELL.w, CELL.h - usedH);
        ctx.fillStyle = "#5aa9e6"; ctx.fillRect(r.x, r.y + CELL.h - usedH, CELL.w, usedH);
      }
    }
    $("cl-failed").textContent = fmt(f.failed);
    $("cl-alloc").textContent = fmt(f.alloc);
    $("cl-left").textContent = fmt(left);
    $("cl-scrub").value = frame;
    $("cl-progress").textContent = `${fmt(Math.min(1, f.p) * 100)}% of capacity requested`;
  }
})();
