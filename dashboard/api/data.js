// Vercel serverless function: the dashboard's data, read live from MongoDB Atlas.
// Same shape as backfill/demo_data.py. Set MONGODB_URI (and optionally MONGODB_DB) in Vercel.
const { MongoClient } = require("mongodb");

let clientPromise = null;
const DB = process.env.MONGODB_DB || "backfill";
const ABLATION_DB = process.env.BACKFILL_ABLATION_DB || "backfill_ablation";
const TRAIN = ["default", "gpushare40", "multigpu20", "gpuspec33"];
const HELDOUT = ["gpushare80", "multigpu50", "gpuspec20", "gpushare60"];

function client() {
  if (!clientPromise) {
    clientPromise = new MongoClient(process.env.MONGODB_URI, { appName: "backfill-dashboard", serverSelectionTimeoutMS: 6000 }).connect();
  }
  return clientPromise;
}

const firstLine = (s, n) => (s || "").split("\n")[0].slice(0, n);

async function readDb(db) {
  const [runs, versions, lessons, ledger] = await Promise.all([
    db.collection("runs").find({ kind: "candidate" }, { projection: { code: 0, code_embedding: 0, curves: 0 } }).toArray(),
    db.collection("harness_versions").find({}, { projection: { spec: 0, stats: 0 } }).toArray(),
    db.collection("lessons").countDocuments(),
    db.collection("ledger").find({}, { projection: { tokens: 1, kind: 1 } }).toArray(),
  ]);
  return {
    runs: runs.map((r) => ({
      _id: r._id, generation: r.generation, harness_version: r.harness_version, harness_id: r.harness_id,
      created_at: r.created_at, ok: r.ok, mean_unalloc: r.mean_unalloc, per_workload: r.per_workload,
      heldout: r.heldout || null, signals: r.signals || [], recalled_lessons: (r.recalled_lessons || []).length,
      duplicate_of: r.duplicate_of || null, idea: firstLine(r.idea, 160), error: r.error ? r.error.slice(0, 160) : null,
      raindrop_event_id: r.raindrop_event_id || null, tool_spans: (r.tools || []).length,
    })).sort((a, b) => (a.created_at || 0) - (b.created_at || 0)),
    versions: versions.map((v) => ({
      _id: v._id, version: v.version, parent_id: v.parent_id, status: v.status, rationale: (v.rationale || "").slice(0, 400),
      score: v.score, heldout_score: v.heldout_score, created_at: v.created_at, decided_at: v.decided_at || null, author: v.author,
    })).sort((a, b) => a.version - b.version),
    lessons,
    tokens: ledger.reduce((s, l) => s + (l.tokens || 0), 0),
    usage: {
      tokens_generations: ledger.filter((l) => l.kind !== "meta").reduce((s, l) => s + (l.tokens || 0), 0),
      tokens_rewrites: ledger.filter((l) => l.kind === "meta").reduce((s, l) => s + (l.tokens || 0), 0),
      rewrites: ledger.filter((l) => l.kind === "meta").length,
    },
  };
}

module.exports = async (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  if (!process.env.MONGODB_URI) {
    res.status(503).json({ error: "MONGODB_URI is not set on this deployment" });
    return;
  }
  try {
    const c = await client();
    const main = await readDb(c.db(DB));
    let ablation = null;
    try {
      const ab = await readDb(c.db(ABLATION_DB));
      ablation = ab.runs.length ? ab : null;
    } catch (_) { ablation = null; }
    res.status(200).json({
      source: "atlas", generated_at: Date.now() / 1000, recorded_at: null,
      model: process.env.OPENAI_MODEL || "gpt-4.1", train: TRAIN, heldout: HELDOUT, main, ablation,
    });
  } catch (e) {
    clientPromise = null;
    res.status(502).json({ error: `Atlas read failed: ${e.message}` });
  }
};
