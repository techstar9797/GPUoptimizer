"""Text embeddings from Voyage AI (MongoDB), used by Atlas Vector Search."""
from __future__ import annotations

import hashlib
import re

import httpx
import numpy as np

from .config import EMBED_DIM, SETTINGS

# Keys created in MongoDB Atlas (AI Model APIs) use ai.mongodb.com; keys from the
# Voyage dashboard use api.voyageai.com. We try the configured one first, then the other.
ENDPOINTS = ["https://ai.mongodb.com/v1/embeddings", "https://api.voyageai.com/v1/embeddings"]
_working: str | None = None


def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed texts with Voyage. Without a key, fall back to a local hashed bag of words
    so the pipeline still runs offline (search quality is much lower)."""
    if not texts:
        return []
    if not SETTINGS.voyage_api_key:
        return [_offline(t) for t in texts]
    global _working
    order = [_working] if _working else ([SETTINGS.voyage_url] if SETTINGS.voyage_url else []) + \
        [e for e in ENDPOINTS if e != SETTINGS.voyage_url]
    last = None
    for url in order:
        r = httpx.post(url, headers={"Authorization": f"Bearer {SETTINGS.voyage_api_key}"},
                       json={"input": texts, "model": SETTINGS.voyage_model, "input_type": input_type},
                       timeout=60)
        if r.status_code == 200:
            _working = url
            break
        last = r
        if r.status_code not in (401, 403):
            break
    else:
        r = None
    if r is None or r.status_code != 200:
        raise RuntimeError(f"Voyage embeddings failed with HTTP {last.status_code}: {last.text[:300]}")
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def embed_one(text: str, input_type: str = "document") -> list[float]:
    return embed([text], input_type)[0]


def _offline(text: str) -> list[float]:
    v = np.zeros(EMBED_DIM)
    for tok in re.findall(r"[a-z_][a-z0-9_]+|\d+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % EMBED_DIM] += 1.0 if (h >> 20) & 1 else -1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()
