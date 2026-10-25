from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


TITLES = {
    "E00": "feasibility_and_baselines",
    "E01": "oracle_and_learnability",
    "E02": "counterfactual_rollout_distillation",
    "E03": "value_calibrated_halting",
    "E04": "memory_and_transition_graph",
    "E05": "prices_and_congestion_games",
    "E06": "causal_transport_assignment",
    "E07": "budget_scheduling",
    "E08": "stability_and_synthetic_mechanisms",
    "E09": "router_only_policy_gradient",
    "E10": "ordered_complementarity",
    "E11": "systems_and_scale_validation",
    "E12": "expert_self_bidding",
}

PRECONDITIONS = {
    "E00": "none",
    "E01": "qualified_frozen_backbone",
    "E02": "E01_deployable_learnability_pass",
    "E03": "usable_multi_depth_backbone",
    "E04": "E02_same_hash_labels",
    "E05": "measured_load_bottleneck_and_E02_values",
    "E06": "causal_service_window_and_scores",
    "E07": "E03_calibrated_values_and_service_load",
    "E08": "none_for_synthetic",
    "E09": "E01_positive_E02_student_and_credit_limit_evidence",
    "E10": "E01_two_round_counterfactual_labels",
    "E11": "last_layer_mechanism_pass",
    "E12": "E01_labels_and_equivalence_screen",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_registry() -> dict:
    return {
        "version": 1,
        "updated_at": now(),
        "experiments": {
            experiment: {
                "run_id": f"{experiment}_{title}",
                "title": title,
                "status": "planned",
                "precondition": PRECONDITIONS[experiment],
                "components": {},
                "evidence": [],
                "metrics": {},
                "gpu_hours": 0.0,
                "stop_reason": None,
                "next_trigger": PRECONDITIONS[experiment],
            }
            for experiment, title in TITLES.items()
        },
    }


class Registry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = initial_registry()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = now()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n")
        os.replace(temporary, self.path)

    def update(self, experiment: str, component: str, status: str, evidence: list[str] | None = None, metrics: dict | None = None, gpu_hours: float = 0.0, stop_reason: str | None = None, next_trigger: str | None = None) -> None:
        record = self.data["experiments"][experiment]
        record["status"] = status
        record["components"][component] = {"status": status, "evidence": evidence or [], "metrics": metrics or {}, "gpu_hours": gpu_hours, "updated_at": now()}
        record["evidence"] = sorted(set(record["evidence"] + (evidence or [])))
        record["metrics"].update(metrics or {})
        record["gpu_hours"] = sum(item["gpu_hours"] for item in record["components"].values())
        record["stop_reason"] = stop_reason
        record["next_trigger"] = next_trigger
        self.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="runs/registry.json")
    parser.add_argument("--experiment", choices=list(TITLES))
    parser.add_argument("--component")
    parser.add_argument("--status", choices=("planned", "running", "finished", "stopped_by_evidence", "blocked_by_precondition", "failed"))
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--metrics")
    parser.add_argument("--gpu-hours", type=float, default=0.0)
    parser.add_argument("--stop-reason")
    parser.add_argument("--next-trigger")
    args = parser.parse_args()
    registry = Registry(args.path)
    if args.experiment:
        if not args.component or not args.status:
            parser.error("updates require --component and --status")
        registry.update(args.experiment, args.component, args.status, args.evidence, json.loads(args.metrics) if args.metrics else None, args.gpu_hours, args.stop_reason, args.next_trigger)
    else:
        registry.save()
    print(json.dumps(registry.data, indent=2))


if __name__ == "__main__":
    main()
