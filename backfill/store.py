"""Harness memory and state in MongoDB Atlas.

Collections
  harness_versions  one document per harness version, with parent_id for lineage
  runs              every policy evaluation (code, scores, curves, errors, embedding)
  lessons           what the critic learned, with an embedding for recall
  workloads         a numeric fingerprint per workload, for routing
  ledger            tokens and seconds spent per generation

Atlas Vector Search indexes
  runs_code_vec     runs.code_embedding        duplicate-policy detection
  lessons_vec       lessons.embedding          lesson recall, filtered by workload family
  workloads_vec     workloads.fingerprint      route a new workload to the closest known one

Without MONGODB_URI the same interface runs on local JSON files with brute-force
cosine search, so the code can be developed offline. The demo uses Atlas.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from .config import EMBED_DIM, FINGERPRINT_DIM, SETTINGS

COLLECTIONS = ["harness_versions", "runs", "lessons", "workloads", "ledger"]
VECTOR_INDEXES = {
    "runs": ("runs_code_vec", "code_embedding", EMBED_DIM, ["kind"]),
    "lessons": ("lessons_vec", "embedding", EMBED_DIM, ["family"]),
    "workloads": ("workloads_vec", "fingerprint", FINGERPRINT_DIM, ["split"]),
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def family_of(workload: str) -> str:
    for f in ("gpushare", "multigpu", "gpuspec", "cpu"):
        if workload.startswith(f):
            return f
    return "default"


class AtlasStore:
    backend = "atlas"

    def __init__(self, uri: str, db: str):
        from pymongo import MongoClient
        self.client = MongoClient(uri, appname="backfill-harness", serverSelectionTimeoutMS=15000)
        self.db = self.client[db]
        from pymongo.errors import OperationFailure, ServerSelectionTimeoutError
        try:
            self.client.admin.command("ping")
        except OperationFailure as e:
            if e.code == 18:
                raise SystemExit("Atlas rejected the username or password in MONGODB_URI. "
                                 "Run: python scripts/check_env.py") from None
            raise
        except ServerSelectionTimeoutError:
            raise SystemExit("Could not reach the Atlas cluster. Add your IP under Security > Network Access, "
                             "then run: python scripts/check_env.py") from None

    # ---------- setup ----------
    def setup(self, wait: bool = True) -> list[str]:
        from pymongo.operations import SearchIndexModel
        log = []
        existing = set(self.db.list_collection_names())
        for c in COLLECTIONS:
            if c not in existing:
                self.db.create_collection(c)
                log.append(f"created collection {c}")
        self.db.harness_versions.create_index("created_at")
        self.db.runs.create_index([("generation", 1), ("harness_id", 1)])
        for coll, (name, path, dims, filters) in VECTOR_INDEXES.items():
            have = {i["name"] for i in self.db[coll].list_search_indexes()}
            if name in have:
                continue
            fields = [{"type": "vector", "path": path, "numDimensions": dims, "similarity": "cosine"}]
            fields += [{"type": "filter", "path": f} for f in filters]
            self.db[coll].create_search_index(
                SearchIndexModel(definition={"fields": fields}, name=name, type="vectorSearch"))
            log.append(f"created vector index {name} on {coll}.{path} ({dims} dims)")
        if wait:
            for coll, (name, *_rest) in VECTOR_INDEXES.items():
                for _ in range(120):
                    idx = list(self.db[coll].list_search_indexes(name))
                    if idx and idx[0].get("queryable"):
                        break
                    time.sleep(2)
                else:
                    log.append(f"warning: index {name} is still building")
        return log

    # ---------- writes ----------
    def insert(self, coll: str, doc: dict) -> dict:
        self.db[coll].insert_one(dict(doc))
        return doc

    def update(self, coll: str, _id: str, fields: dict) -> None:
        self.db[coll].update_one({"_id": _id}, {"$set": fields})

    def upsert(self, coll: str, _id: str, doc: dict) -> None:
        self.db[coll].replace_one({"_id": _id}, {"_id": _id, **doc}, upsert=True)

    # ---------- reads ----------
    def get(self, coll: str, _id: str) -> dict | None:
        return self.db[coll].find_one({"_id": _id})

    def find(self, coll: str, query: dict | None = None, sort: list | None = None, limit: int = 0,
             projection: dict | None = None) -> list[dict]:
        cur = self.db[coll].find(query or {}, projection)
        if sort:
            cur = cur.sort(sort)
        if limit:
            cur = cur.limit(limit)
        return list(cur)

    def vector_search(self, coll: str, vector: list[float], k: int = 5, filter: dict | None = None,
                      num_candidates: int = 100) -> list[dict]:
        name, path, _, _ = VECTOR_INDEXES[coll]
        stage = {"index": name, "path": path, "queryVector": list(vector),
                 "numCandidates": max(num_candidates, k * 10), "limit": k}
        if filter:
            stage["filter"] = filter
        return list(self.db[coll].aggregate([
            {"$vectorSearch": stage},
            {"$project": {path: 0, "score": {"$meta": "vectorSearchScore"}}},
            {"$addFields": {"similarity": "$score"}},
        ]))

    def lineage(self, harness_id: str) -> list[dict]:
        """Ancestors of a harness version, newest first, via $graphLookup."""
        out = list(self.db.harness_versions.aggregate([
            {"$match": {"_id": harness_id}},
            {"$graphLookup": {"from": "harness_versions", "startWith": "$parent_id",
                              "connectFromField": "parent_id", "connectToField": "_id",
                              "as": "ancestors", "depthField": "depth"}},
        ]))
        if not out:
            return []
        anc = sorted(out[0]["ancestors"], key=lambda d: d["depth"])
        out[0].pop("ancestors")
        return [out[0], *anc]

    def leaderboard(self) -> list[dict]:
        return list(self.db.runs.aggregate([
            {"$match": {"ok": True, "kind": "candidate"}},
            {"$sort": {"mean_unalloc": 1}},
            {"$group": {"_id": "$harness_id", "best": {"$first": "$mean_unalloc"},
                        "runs": {"$sum": 1}, "best_run": {"$first": "$_id"}}},
            {"$sort": {"best": 1}},
        ]))

    def clear(self) -> None:
        for c in COLLECTIONS:
            self.db[c].delete_many({})


class LocalStore(AtlasStore):
    """Offline stand-in with the same interface: JSON files, brute-force cosine search."""
    backend = "local"

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.data = {c: self._load(c) for c in COLLECTIONS}

    def _load(self, c):
        f = self.root / f"{c}.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def _save(self, c):
        (self.root / f"{c}.json").write_text(json.dumps(self.data[c], default=str))

    def setup(self, wait: bool = True):
        return ["local store: nothing to set up"]

    def insert(self, coll, doc):
        self.data[coll][doc["_id"]] = dict(doc)
        self._save(coll)
        return doc

    def update(self, coll, _id, fields):
        self.data[coll][_id].update(fields)
        self._save(coll)

    def upsert(self, coll, _id, doc):
        self.data[coll][_id] = {"_id": _id, **doc}
        self._save(coll)

    def get(self, coll, _id):
        return self.data[coll].get(_id)

    def find(self, coll, query=None, sort=None, limit=0, projection=None):
        rows = [d for d in self.data[coll].values() if _match(d, query or {})]
        for key, direction in reversed(sort or []):
            rows.sort(key=lambda d: (d.get(key) is None, d.get(key)), reverse=direction < 0)
        return rows[:limit] if limit else rows

    def vector_search(self, coll, vector, k=5, filter=None, num_candidates=100):
        _, path, _, _ = VECTOR_INDEXES[coll]
        q = np.asarray(vector, float)
        q /= np.linalg.norm(q) or 1
        scored = []
        for d in self.data[coll].values():
            if path not in d or not _match(d, filter or {}):
                continue
            v = np.asarray(d[path], float)
            sim = float(q @ v / (np.linalg.norm(v) or 1))
            scored.append(({k2: v2 for k2, v2 in d.items() if k2 != path}, (sim + 1) / 2))
        scored.sort(key=lambda t: -t[1])
        return [{**d, "similarity": s, "score": s} for d, s in scored[:k]]

    def lineage(self, harness_id):
        out, cur = [], self.get("harness_versions", harness_id)
        while cur:
            out.append(cur)
            cur = self.get("harness_versions", cur.get("parent_id")) if cur.get("parent_id") else None
        return out

    def leaderboard(self):
        best = {}
        for r in self.data["runs"].values():
            if r.get("ok") and r.get("kind") == "candidate":
                b = best.setdefault(r["harness_id"], {"_id": r["harness_id"], "best": 1e18, "runs": 0})
                b["runs"] += 1
                if r["mean_unalloc"] < b["best"]:
                    b["best"], b["best_run"] = r["mean_unalloc"], r["_id"]
        return sorted(best.values(), key=lambda b: b["best"])

    def clear(self):
        for c in COLLECTIONS:
            self.data[c] = {}
            self._save(c)


class ReplayStore(LocalStore):
    """Read-only store over a recorded snapshot (results/snapshot.json), for demos without network."""
    backend = "replay"

    def __init__(self, snapshot: dict, db: str):
        self.root = None
        dbs = snapshot.get("databases", {})
        src = dbs.get(db, {})
        self.data = {c: {d["_id"]: d for d in src.get(c, [])} for c in COLLECTIONS}
        self.recorded_at = snapshot.get("created_at")

    def _load(self, c):
        return self.data[c]

    def _save(self, c):
        pass


def _match(d: dict, q: dict) -> bool:
    for k, v in q.items():
        if isinstance(v, dict) and "$in" in v:
            if d.get(k) not in v["$in"]:
                return False
        elif isinstance(v, dict) and "$eq" in v:
            if d.get(k) != v["$eq"]:
                return False
        elif d.get(k) != v:
            return False
    return True


_STORES: dict = {}


def get_store(db: str | None = None):
    """Atlas when MONGODB_URI is set, otherwise a local JSON store under .localstore/.
    Pass db to open another database, such as the no-memory ablation run."""
    import os
    name = db or os.environ.get("MONGODB_DB") or SETTINGS.mongodb_db
    if name not in _STORES:
        replay = os.environ.get("BACKFILL_REPLAY")
        if replay:
            import json as _json
            _STORES[name] = ReplayStore(_json.loads(Path(replay).read_text()), name)
        elif SETTINGS.mongodb_uri:
            _STORES[name] = AtlasStore(SETTINGS.mongodb_uri, name)
        else:
            root = Path(__file__).resolve().parent.parent / ".localstore"
            _STORES[name] = LocalStore(root if name == SETTINGS.mongodb_db else root / name)
    return _STORES[name]
