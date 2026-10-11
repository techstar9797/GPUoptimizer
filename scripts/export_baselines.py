"""Write dashboard/public/baselines.json (baseline scores per workload and seed) for the dashboard."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backfill.demo_data import baselines  # noqa: E402

out = Path(__file__).resolve().parent.parent / "dashboard" / "public" / "baselines.json"
rows = baselines()
out.write_text(json.dumps(rows))
print(f"Wrote {len(rows)} baseline rows to {out}")
