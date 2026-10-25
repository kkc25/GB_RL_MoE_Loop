from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {}
    for run_dir in args.run_dirs:
        root = Path(run_dir)
        if not (root / "run_summary.json").exists():
            report[run_dir] = {"status": "running_or_missing"}
            continue
        summary = json.loads((root / "run_summary.json").read_text())
        ledger = summary["ledger"]
        report[run_dir] = {
            "parameters": summary["environment"]["parameters"],
            "input_tokens": ledger["input_tokens"],
            "gpu_hours": ledger["allocated_gpu_hours"],
            "peak_memory_bytes": ledger["peak_memory_bytes"],
            "aggregate_tokens_per_second": ledger["input_tokens"] / max(ledger["elapsed_seconds"], 1e-9),
            "validation": summary["validation"],
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
