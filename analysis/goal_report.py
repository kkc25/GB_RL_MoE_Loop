from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.registry import TITLES


def build_report(path: str | Path, repo_root: str | Path | None = None) -> dict:
    registry_path = Path(path)
    root = Path(repo_root) if repo_root is not None else registry_path.parent.parent if registry_path.parent.name == "runs" else Path.cwd()
    registry = json.loads(registry_path.read_text())
    experiments = {}
    missing_evidence = []
    for experiment, title in TITLES.items():
        record = registry["experiments"][experiment]
        evidence = []
        for item in record["evidence"]:
            exists = (root / item).exists()
            evidence.append({"path": item, "exists": exists})
            if not exists:
                missing_evidence.append({"experiment": experiment, "path": item})
        experiments[experiment] = {
            "title": title,
            "status": record["status"],
            "components": record["components"],
            "metrics": record["metrics"],
            "gpu_hours": record.get("gpu_hours"),
            "gpu_hours_unknown": record.get("gpu_hours") is None,
            "stop_reason": record["stop_reason"],
            "next_trigger": record["next_trigger"],
            "evidence": evidence,
        }
    return {
        "registry": str(registry_path),
        "experiments": experiments,
        "missing_evidence": missing_evidence,
        "complete_experiment_records": len(experiments),
        "all_have_components": all(bool(record["components"]) for record in registry["experiments"].values()),
        "terminal_statuses": {
            status: sum(record["status"] == status for record in registry["experiments"].values())
            for status in ("finished", "stopped_by_evidence", "blocked_by_precondition", "failed")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="runs/registry.json")
    parser.add_argument("--repo-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(args.registry, args.repo_root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["terminal_statuses"], indent=2))
    print(f"missing_evidence={len(report['missing_evidence'])}")


if __name__ == "__main__":
    main()
