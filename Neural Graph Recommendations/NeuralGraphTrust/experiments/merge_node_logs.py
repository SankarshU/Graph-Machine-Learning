"""Merge results/logs/*.jsonl from every node archive in results/node_logs/*.tgz into results/logs/
(ml100k__runs.jsonl, ml100k__graphtrust.jsonl). Rows are concatenated in node order; the table
script keeps the last row per (model, setting, seed)."""
import io, json, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
out = {"ml100k__runs.jsonl": [], "ml100k__graphtrust.jsonl": []}
for tgz in sorted((ROOT / "results" / "node_logs").glob("*.tgz")):
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            name = Path(m.name).name
            if name in out:
                rows = [l for l in t.extractfile(m).read().decode().splitlines() if l.strip()]
                out[name] += rows
                print(f"{tgz.name}: {name} {len(rows)} rows")
for name, rows in out.items():
    (ROOT / "results" / "logs" / name).write_text("\n".join(rows) + "\n")
    print("merged", name, len(rows))
