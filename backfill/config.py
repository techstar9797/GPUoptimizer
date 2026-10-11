"""Settings, read from the environment and an optional .env file in the repo root."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # dotenv is optional
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4.1"))
    mongodb_uri: str = field(default_factory=lambda: _env("MONGODB_URI"))
    mongodb_db: str = field(default_factory=lambda: _env("MONGODB_DB", "backfill"))
    voyage_api_key: str = field(default_factory=lambda: _env("VOYAGE_API_KEY"))
    voyage_model: str = field(default_factory=lambda: _env("VOYAGE_MODEL", "voyage-3.5"))
    raindrop_write_key: str = field(default_factory=lambda: _env("RAINDROP_WRITE_KEY"))
    run_user: str = field(default_factory=lambda: _env("BACKFILL_USER", "backfill-harness"))

    def missing(self) -> list[str]:
        out = []
        if not self.openai_api_key:
            out.append("OPENAI_API_KEY")
        if not self.mongodb_uri:
            out.append("MONGODB_URI")
        if not self.voyage_api_key:
            out.append("VOYAGE_API_KEY")
        if not self.raindrop_write_key:
            out.append("RAINDROP_WRITE_KEY")
        return out


SETTINGS = Settings()

# Train on these; hold the rest out for the generalization test.
TRAIN_WORKLOADS = ["default", "gpushare40", "multigpu20", "gpuspec33"]
HELDOUT_WORKLOADS = ["gpushare80", "multigpu50", "gpuspec20", "gpushare60"]
EMBED_DIM = 1024          # voyage-3.5 default output dimension
FINGERPRINT_DIM = 12      # workload fingerprint length
