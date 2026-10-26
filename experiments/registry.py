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


def aggregate_status(components: dict) -> str:
    """Experiment-level status derived from all components, never the last one.

    "finished" requires every component finished; otherwise the most severe
    non-finished status wins, so a blocked/stopped/failed component is not
    hidden by a finished sibling regardless of update order.
    """
    statuses = {component.get("status") for component in components.values()}
    statuses.discard(None)
    if not statuses:
        return "planned"
    if statuses == {"finished"}:
        return "finished"
    for status in ("failed", "blocked_by_precondition", "stopped_by_evidence", "running", "planned"):
        if status in statuses:
            return status
    return "finished"


def aggregate_gpu_hours(components: dict) -> float | None:
    """Sum component GPU hours; unknown (None) if any component lacks them."""
    hours = [component.get("gpu_hours") for component in components.values()]
    if any(value is None for value in hours):
        return None
    return sum(hours)


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

    def update(self, experiment: str, component: str, status: str, evidence: list[str] | None = None, metrics: dict | None = None, gpu_hours: float | None = None, stop_reason: str | None = None, next_trigger: str | None = None) -> None:
        record = self.data["experiments"][experiment]
        record["components"][component] = {"status": status, "evidence": evidence or [], "metrics": metrics or {}, "gpu_hours": gpu_hours, "stop_reason": stop_reason, "next_trigger": next_trigger, "updated_at": now()}
        record["status"] = aggregate_status(record["components"])
        record["evidence"] = sorted(set(record["evidence"] + (evidence or [])))
        record["metrics"].update(metrics or {})
        record["gpu_hours"] = aggregate_gpu_hours(record["components"])
        matching = [item for item in record["components"].values() if item["status"] == record["status"]]
        for field in ("stop_reason", "next_trigger"):
            values = [item for item in matching if item.get(field) is not None]
            record[field] = max(values, key=lambda item: item["updated_at"])[field] if values else None
        self.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="runs/registry.json")
    parser.add_argument("--experiment", choices=list(TITLES))
    parser.add_argument("--component")
    parser.add_argument("--status", choices=("planned", "running", "finished", "stopped_by_evidence", "blocked_by_precondition", "failed"))
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--metrics")
    parser.add_argument("--gpu-hours", type=float, default=None, help="recorded GPU hours for this component; omit when unknown")
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
