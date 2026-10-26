"""One read-only probe of the legacy report helper's missing-math-file behavior.

Executes the original analysis/report.py build_report against an empty temporary
run directory. No formulas are substituted and no repository files are modified.
This does not concern the independently verified current NumPy E09 gradients.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1] / "GB_RL_MoE_Loop")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("legacy_report_missing_math_checks.json"))
    args = parser.parse_args()
    source = args.repo.resolve() / "analysis/report.py"
    spec = importlib.util.spec_from_file_location("audit_legacy_report", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audit_empty_report_", dir=args.output.parent) as temporary:
        root = Path(temporary)
        input_files = sorted(path.name for path in root.iterdir())
        result = module.build_report(root)
        assert input_files == []
        assert result["math_gate_pass"] is True
        assert result["training_records"] == 0
        assert result["p0_learning_gate_pass"] is False
        output = {
            "scope": "Legacy analysis/report.py helper only; no evidence this affects current NumPy E09, whose gradient checks were independently reproduced.",
            "source": "analysis/report.py:17,38-45",
            "repository_commit": "f2bef422f7315f30618f00c690bf82810d1e85ac",
            "input_directory_files": input_files,
            "math_checks_json_present": (root / "math_checks.json").exists(),
            "original_build_report_output": result,
            "observed_bug": "A completely empty run directory yields math_gate_pass=true because not math_checks is true.",
            "required_semantics": "Missing checks must be unknown/not_assessed (or fail closed); only present valid passing checks can establish a passed mathematical gate.",
            "temporary_input_cleanup": "The empty temporary run directory is removed after this probe.",
        }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
