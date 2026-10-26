"""Validate and package this audit, without modifying the reviewed repository.

This validates delivery integrity, not new scientific/GPU results. Defaults
expect the repository as a sibling named GB_RL_MoE_Loop at the pinned commit.
Existing reproduction JSON/logs are preserved. No tests or training are run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlparse
import zipfile

COMMIT = "f2bef422f7315f30618f00c690bf82810d1e85ac"
ROOT = Path(__file__).resolve().parent
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "pytest_tmp"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def delivery_files():
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and not EXCLUDED_PARTS.intersection(p.relative_to(ROOT).parts)
                  and p.suffix != ".pyc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT.parent / "GB_RL_MoE_Loop")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "GB_RL_MoE_Loop_code_audit_2026-10-25.zip")
    args = parser.parse_args()
    repo = args.repo.resolve()
    snapshot = json.loads((ROOT / "repository_snapshot.json").read_text())
    assert snapshot["commit"] == COMMIT
    blobs = [x for x in snapshot["tree"]["tree"] if x["type"] == "blob"]
    verified = []
    for item in blobs:
        path = repo / item["path"]
        payload = path.read_bytes()
        digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        assert digest == item["sha"], f"Repository bytes changed: {item['path']}"
        verified.append({"path": item["path"], "bytes": len(payload), "git_blob_sha": digest})
    assert len(verified) == 390
    python_paths = [x["path"] for x in verified if x["path"].endswith(".py")]
    production = [p for p in python_paths if not p.startswith(("plans/", "tests/"))]
    tests = [p for p in python_paths if p.startswith("tests/")]
    plans = [p for p in python_paths if p.startswith("plans/")]
    configs = [x["path"] for x in verified if x["path"].startswith("configs/")]
    write_json(ROOT / "source_inventory.json", {
        "repository": "kkc25/GB_RL_MoE_Loop", "commit": COMMIT,
        "tracked_files": len(verified), "tracked_bytes": sum(x["bytes"] for x in verified),
        "python_production_and_initializers": production, "python_test_files": tests,
        "python_design_materials": plans, "config_files": configs,
        "counts": {"production_and_initializers": len(production), "tests": len(tests),
                   "plan_python": len(plans), "config_files": len(configs)},
        "all_verified_files": verified,
        "scope": "File acquisition and blob integrity are complete. Review coverage and limits are explained in 04_reproduction_and_scope.md; no trained checkpoint or GPU quality/latency rerun is implied."
    })
    original = json.loads((ROOT / "minimal_patch_tests_original.json").read_text())
    patched = json.loads((ROOT / "minimal_patch_tests_patched.json").read_text())
    assert (original["tests_run"], original["failures"], original["errors"]) == (3, 2, 1)
    assert patched["tests_run"] == 3 and patched["successful"] is True
    full_suite = (ROOT / "existing_tests.log").read_text()
    assert "424 passed in 83.15s" in full_suite
    write_json(ROOT / "verification_summary.json", {
        "source_commit": COMMIT,
        "repository_integrity": {"files_verified": len(verified), "changed_tracked_files": 0},
        "original_suite": {"passed": 424, "duration_seconds": 83.15, "source": "existing_tests.log"},
        "minimal_regressions_original": original, "minimal_regressions_patched_in_memory": patched,
        "audit_environment": {"python": "3.12.14", "numpy": "2.3.5", "torch": "2.5.1+cpu",
                              "pytest": "8.3.5", "tokenizers": "0.21.4", "cuda_available": False},
        "original_run_environment": {"python": "3.12.15", "torch": "2.12.1+cu130", "device": "RTX A6000"},
        "trained_checkpoint_used": False, "gpu_training_or_latency_rerun": False,
        "remote_repository_changed": False,
        "limits": "Micro-model counterexamples and record recomputations establish scoped findings, not corrected real-checkpoint performance."
    })
    # Parse artifact files and verify links to avoid delivering unusable evidence.
    local_links, source_links = 0, 0
    for path in delivery_files():
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == ".json":
            json.loads(path.read_text())
        elif path.suffix == ".md":
            body = path.read_text()
            assert len(re.findall(r"^```", body, re.M)) % 2 == 0, f"Unclosed fence: {path.name}"
            definitions = dict(re.findall(r"^\[([^\]]+)\]:\s+(\S+)", body, re.M))
            for ref in re.findall(r"\[[^\]\n]+\]\[([^\]]+)\]", body):
                assert ref in definitions, f"Undefined Markdown reference {ref}: {path.name}"
            urls = re.findall(r"\[[^\]\n]*\]\(([^)]+)\)", body) + list(definitions.values())
            for url in urls:
                if url.startswith("https://github.com/kkc25/GB_RL_MoE_Loop/"):
                    parts = urlparse(url)
                    for mode in ("blob", "tree"):
                        marker = f"/{mode}/{COMMIT}/"
                        if marker in parts.path:
                            target = repo / unquote(parts.path.split(marker, 1)[1])
                            assert target.exists(), f"Missing cited repository target: {url}"
                            if parts.fragment.startswith("L") and target.is_file():
                                match = re.fullmatch(r"L(\d+)(?:-L(\d+))?", parts.fragment)
                                if match:
                                    count = len(target.read_text().splitlines())
                                    assert max(int(x) for x in match.groups() if x) <= count, f"Citation exceeds file: {url}"
                            source_links += 1
                elif not urlparse(url).scheme and not url.startswith("#"):
                    target = (path.parent / unquote(url.split("#", 1)[0])).resolve()
                    assert target.exists(), f"Missing artifact link: {path.name}: {url}"
                    local_links += 1
    check = subprocess.run(["git", "apply", "--check", str(ROOT / "patches/minimal_correctness_fixes_with_tests.patch")],
                           cwd=repo, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    matrix = (ROOT / "01_plan_implementation_matrix.md").read_text()
    assert all(f"E{i:02d}" in matrix for i in range(13))
    write_json(ROOT / "artifact_validation.json", {
        "source_commit": COMMIT, "repository_git_blob_matches": len(verified),
        "production_sources_unchanged": True, "python_syntax_valid": True, "json_parse_valid": True,
        "markdown_code_fences_balanced": True, "local_link_checks": local_links,
        "fixed_commit_source_link_checks": source_links, "matrix_experiments": 13,
        "patch_git_apply_check": "passed", "new_training_executed": False,
        "archive_policy": "Only audit deliverables; excludes repository copy, dependencies, checkpoints, pycache and pytest temporary files."
    })
    files = [p for p in delivery_files() if p.name != "MANIFEST_SHA256.json"]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(ROOT / "MANIFEST_SHA256.json", manifest)
    files = delivery_files()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(f"{ROOT.name}/{name}")).hexdigest() == digest
    print(json.dumps({"archive": str(args.output.resolve()), "files": len(files),
                      "markdown_files": sum(p.suffix == '.md' for p in files),
                      "archive_bytes": args.output.stat().st_size,
                      "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "local_links_checked": local_links, "source_links_checked": source_links,
                      "all_repository_blobs_unchanged": True, "zip_integrity": "passed"}, indent=2))


if __name__ == "__main__":
    main()
