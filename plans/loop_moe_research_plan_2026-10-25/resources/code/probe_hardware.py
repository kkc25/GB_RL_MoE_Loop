#!/usr/bin/env python3
"""Read-only hardware inventory. Does not launch training or allocate model tensors.
Run on the actual server: python3 probe_hardware.py > hardware_inventory.json
"""
import json
import platform
import subprocess

out = {"status": "inventory_not_performance_benchmark", "python": platform.python_version(),
       "platform": platform.platform()}
commands = {
    "gpus": ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id,power.limit", "--format=csv,noheader"],
    "topology": ["nvidia-smi", "topo", "-m"],
}
for name, cmd in commands.items():
    try:
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=20, check=False)
        out[name] = {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        out[name] = {"error": str(exc)}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["visible_gpu_count"] = torch.cuda.device_count()
        out["devices"] = []
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                props = torch.cuda.get_device_properties(idx)
                out["devices"].append({"index": idx, "name": props.name,
                    "compute_capability": list(torch.cuda.get_device_capability(idx)),
                    "total_memory_bytes": props.total_memory,
                    "bf16_supported": torch.cuda.is_bf16_supported()})
except ImportError:
    out["torch"] = "not_installed"
print(json.dumps(out, indent=2, ensure_ascii=False))
