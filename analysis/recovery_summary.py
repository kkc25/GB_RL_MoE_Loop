from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
    validation = [row for row in rows if row.get("phase") == "validation"]
    by_update = {}
    for row in validation:
        by_update.setdefault(row["update"], {})[row["split"]] = row
    selection_split = "id_validation" if any("id_validation" in row for row in by_update.values()) else "train"
    best_update, best = min(by_update.items(), key=lambda item: item[1][selection_split]["ce_full"])
    final = json.loads((root / "stage_result.json").read_text()) if (root / "stage_result.json").exists() else {}
    return {
        "run_dir": str(root),
        "best_validation_update": best_update,
        "best_validation": best,
        "final": final.get("metrics", {}),
        "ledger": final.get("ledger", {}),
        "stage_pass": final.get("stage_pass"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {path: summarize(path) for path in args.run_dirs}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
