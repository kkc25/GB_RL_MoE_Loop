"""Read-only supplemental audit of causality entry point and production helpers.

Run with the repository root on PYTHONPATH (plus the audit torch dependency path).
No source patching or training is performed. The unsupported-config exception
is recorded as evidence instead of being treated as a corrected result.
"""

import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1] / "GB_RL_MoE_Loop"
sys.path.insert(0, str(ROOT))

import torch
from data.affine import affine_collate
from eval import causality as cz

torch.set_num_threads(1)
report = {}

try:
    cz.run(seed=0)
except Exception as error:
    report["aggregate_entrypoint"] = {
        "completed": False,
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
else:
    report["aggregate_entrypoint"] = {"completed": True}

items = cz.make_items(4, seed=0)
batch = affine_collate(items)
model, controller = cz.build_pair(target_only=True, seed=0, patch_memory_dtype=False)
with torch.no_grad():
    out = model(batch["input_ids"], labels=batch["labels"],
                controller=controller, budget=2, sample=False)
report["unpatched_production_float64"] = {
    "completed": True,
    "memory_dtype": str(controller.initial_memory(4, torch.device("cpu")).dtype),
    "finite_logits": bool(torch.isfinite(out.logits).all()),
    "instance_initial_memory_patch_present": "initial_memory" in controller.__dict__,
}

model, controller = cz.build_pair(target_only=False, seed=0, patch_memory_dtype=False)
seed = 5
with torch.no_grad():
    ordinary = model(batch["input_ids"], labels=batch["labels"],
                     controller=controller, budget=3, sample=True,
                     generator=torch.Generator().manual_seed(seed))
    order = [3, 2, 1, 0]
    reverse_batch = affine_collate([items[i] for i in order])
    reverse = model(reverse_batch["input_ids"], labels=reverse_batch["labels"],
                    controller=controller, budget=3, sample=True,
                    generator=torch.Generator().manual_seed(seed))
    results = []
    for i, item in enumerate(items):
        solo = model(item["input_ids"].unsqueeze(0), labels=item["labels"].unsqueeze(0),
                     controller=controller, budget=3, sample=True,
                     generator=torch.Generator().manual_seed(seed))
        n = item["input_ids"].numel()
        j = order.index(i)
        results.append({
            "item": i,
            "solo_batch_max_logit_difference": float((solo.logits[0, :n] - ordinary.logits[i, :n]).abs().max()),
            "solo_batch_max_call_difference": int((solo.calls[0, :n] - ordinary.calls[i, :n]).abs().max()),
            "reorder_max_logit_difference": float((reverse.logits[j, :n] - ordinary.logits[i, :n]).abs().max()),
            "reorder_max_call_difference": int((reverse.calls[j, :n] - ordinary.calls[i, :n]).abs().max()),
        })
report["sampled_batch_reorder_production"] = results
report["scope"] = "CPU, no-cache path, Top-1, small random model; not LM Top-2 training or KV-cache validation."
destination = Path(__file__).with_name("causality_supplement_checks.json")
destination.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
