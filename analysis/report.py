from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_report(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    metrics = read_jsonl(root / "metrics.jsonl")
    math_checks = json.loads((root / "math_checks.json").read_text()) if (root / "math_checks.json").exists() else {}
    ledger = json.loads((root / "cost_ledger.json").read_text()) if (root / "cost_ledger.json").exists() else {}
    training = [row for row in metrics if row.get("phase") in {"rloo", "exact", "halt", "imitation", "pretrain"}]
    validation = [row for row in metrics if row.get("phase") == "validation"]
    algorithms = {}
    for row in validation:
        algorithms.setdefault(row.get("algorithm", "unknown"), []).append(row)
    latest = {name: rows[-1] for name, rows in algorithms.items()}
    rloo = [row for row in training if row.get("phase") == "rloo"]
    window = min(20, max(1, len(rloo) // 5))
    rloo_trend = None
    if rloo:
        early = rloo[:window]
        late = rloo[-window:]
        rloo_trend = {
            "early_nll": sum(row["nll"] for row in early) / len(early),
            "late_nll": sum(row["nll"] for row in late) / len(late),
            "early_cost": sum(row["cost"] for row in early) / len(early),
            "late_cost": sum(row["cost"] for row in late) / len(late),
        }
        rloo_trend["nll_change"] = rloo_trend["late_nll"] - rloo_trend["early_nll"]
    math_pass = not math_checks or (
        math_checks["max_abs_pg_vs_fd"] < 1e-7
        and math_checks["max_abs_exact_halt_vs_fd"] < 1e-7
        and math_checks["max_abs_rloo2_vs_exact_pg"] < 1e-7
    )
    return {
        "run_dir": str(root),
        "math_gate_pass": math_pass,
        "training_records": len(training),
        "latest_validation": latest,
        "rloo_trend": rloo_trend,
        "ledger": ledger,
        "p0_learning_gate_pass": bool(rloo_trend and rloo_trend["nll_change"] < 0),
        "large_scale_gate": "not_assessed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = build_report(args.run_dir)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
