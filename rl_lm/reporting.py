"""Versioned reporting helpers; legacy fields retain their original meanings."""
from __future__ import annotations

import torch

METRIC_SCHEMA_VERSION = "rl_lm_metrics_v2"


def stop_metrics(actions: torch.Tensor, stop_allowed: torch.Tensor, *, max_calls: int) -> dict:
    """Counts over recorded (not padded) decisions, not over episodes.

    Pair/single v1 have at most two calls, so every STOP-eligible decision is
    exactly at r=1. An R1-only run has no such decisions: report null, not zero.
    """
    if actions.shape != stop_allowed.shape or actions.ndim != 1:
        raise ValueError("actions and stop_allowed must be matching decision vectors")
    if max_calls not in (1, 2):
        raise ValueError("r=1 reporting requires the v1 one/two-call contract")
    stopped = actions == 0
    if bool((stopped & ~stop_allowed).any()):
        raise ValueError("STOP recorded where it was forbidden")
    decisions = actions.numel()
    eligible = int(stop_allowed.sum())
    stops = int(stopped.sum())
    per_decision = stops / decisions if decisions else None
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "stop_fraction": float(stopped.float().mean()) if decisions else None,  # legacy FP32 reduction
        "stop_fraction_per_decision": per_decision,
        "stop_fraction_conditional_r1": stops / eligible if eligible else None,
        "stop_count": stops,
        "decision_count": decisions,
        "r1_decision_count": eligible,
    }


def reference_metrics(nll: float, reference_nll: float, identity: dict) -> dict:
    return {"identity": identity, "nll": reference_nll,
            "delta_nll": nll - reference_nll, "gain_nll": reference_nll - nll}
