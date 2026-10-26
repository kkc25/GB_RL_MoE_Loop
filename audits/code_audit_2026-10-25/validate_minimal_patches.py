"""Build reviewable diffs and test patched modules in memory, leaving source intact."""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "GB_RL_MoE_Loop"
sys.path.insert(0, str(REPO))


def patch_texts() -> dict[str, tuple[str, str]]:
    dynamic_path = "eval/dynamic_halting.py"
    dynamic = (REPO / dynamic_path).read_text()
    dynamic_fixed = dynamic.replace(
        '        stop_cost = predict(self.students["immediate"]["student"], flat_normalized,',
        '        flat_h = h.reshape(-1, h.shape[-1])\n        stop_cost = predict(self.students["immediate"]["student"], flat_h,',
    ).replace(
        '        continue_cost = predict(self.students["q_pi0"]["student"], flat_normalized,',
        '        continue_cost = predict(self.students["q_pi0"]["student"], flat_h,',
    )
    bidding_path = "experiments/e12_bidding.py"
    bidding = (REPO / bidding_path).read_text()
    bidding_fixed = bidding.replace('def _apply(self, rows:', 'def _forward_rows(self, rows:').replace(
        'torch.einsum("...ei,oih->...oh", rows, self.w1)',
        'torch.einsum("...oi,oih->...oh", rows, self.w1)',
    ).replace('self._apply(self._rows(', 'self._forward_rows(self._rows(')
    assert dynamic_fixed != dynamic and bidding_fixed != bidding
    return {dynamic_path: (dynamic, dynamic_fixed), bidding_path: (bidding, bidding_fixed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patched", action="store_true")
    args = parser.parse_args()
    patches = patch_texts()
    patch_dir = ROOT / "patches"
    patch_dir.mkdir(exist_ok=True)
    combined = ""
    for index, (path, (original, fixed)) in enumerate(patches.items(), start=1):
        diff = "".join(difflib.unified_diff(original.splitlines(True), fixed.splitlines(True), fromfile="a/" + path, tofile="b/" + path))
        (patch_dir / f"0{index}_{Path(path).stem}.patch").write_text(diff)
        combined += diff
    tests = (ROOT / "test_audit_contracts.py").read_text()
    combined += "".join(difflib.unified_diff([], tests.splitlines(True), fromfile="/dev/null", tofile="b/tests/test_audit_contracts.py"))
    (patch_dir / "minimal_correctness_fixes_with_tests.patch").write_text(combined)

    if args.patched:
        for path, (_, fixed) in patches.items():
            module_name = path[:-3].replace("/", ".")
            module = types.ModuleType(module_name)
            module.__file__ = str(REPO / path)
            sys.modules[module_name] = module
            exec(compile(fixed, module.__file__, "exec"), module.__dict__)
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("audit_regression_tests", ROOT / "test_audit_contracts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    mode = "patched" if args.patched else "original"
    with (ROOT / f"minimal_patch_tests_{mode}.log").open("w") as handle:
        result = unittest.TextTestRunner(stream=handle, verbosity=2).run(suite)
    summary = {"mode": mode, "torch_version": torch.__version__, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "successful": result.wasSuccessful(), "repository_changed": False}
    (ROOT / f"minimal_patch_tests_{mode}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.patched and not result.wasSuccessful():
        raise SystemExit(1)
    if not args.patched and result.wasSuccessful():
        raise SystemExit("Regression tests unexpectedly did not detect the original defects")


if __name__ == "__main__":
    main()
