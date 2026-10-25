#!/usr/bin/env python3
"""Validate local research deliverables; does not run model training or access the network."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys


def validate(root):
    errors = []
    stats = {"files": 0, "markdown": 0, "experiments": 0, "python": 0, "bytes": 0}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        stats["files"] += 1
        stats["bytes"] += p.stat().st_size
        if p.suffix == ".md":
            stats["markdown"] += 1
            stats["experiments"] += int(p.parent.name == "experiments")
            s = p.read_text(encoding="utf-8")
            if len(re.findall(r"^```", s, re.M)) % 2:
                errors.append(f"{p.relative_to(root)}: unmatched code fence")
            for a, b, label in [(r"\(", r"\)", "inline math"),
                                (r"\[", r"\]", "display math")]:
                if s.count(a) != s.count(b):
                    errors.append(f"{p.relative_to(root)}: unmatched {label}")
            if any(ord(c) < 32 and c not in "\n\r\t" for c in s):
                errors.append(f"{p.relative_to(root)}: unexpected control character")
            for target in re.findall(r"\]\(([^\s]+)\)", s):
                if target.startswith(("https:", "http:", "mailto:", "#")):
                    continue
                target = target.split("#")[0]
                if target and not (p.parent / target).exists():
                    errors.append(f"{p.relative_to(root)}: missing target {target}")
        elif p.suffix == ".py":
            stats["python"] += 1
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                errors.append(f"{p.relative_to(root)}: {exc}")
        elif p.suffix == ".json":
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"{p.relative_to(root)}: {exc}")
    if stats["experiments"] != 13:
        errors.append("Expected 13 standalone experiment Markdown files")
    manifest = root / "MANIFEST_SHA256.json"
    if manifest.exists():
        for rel, digest in json.loads(manifest.read_text())["sha256"].items():
            path = root / rel
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f"Hash mismatch: {rel}")
    return {"status": "passed" if not errors else "failed", "stats": stats, "errors": errors,
            "scope": "structure, local links, math delimiters, syntax, JSON, optional manifest"}


if __name__ == "__main__":
    report = validate(Path(__file__).resolve().parents[1])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["status"] == "passed" else 1)
