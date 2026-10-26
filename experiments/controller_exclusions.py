from __future__ import annotations

import json
from pathlib import Path


def write_used_document_exclusions(source_dir: str | Path, output_path: str | Path) -> Path:
    source_dir = Path(source_dir)
    used = set()
    for split in ("train", "validation", "test"):
        with (source_dir / f"{split}.doc_map.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                used.update(int(segment["doc"]) for segment in row["segments"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (source_dir / "documents.jsonl").open() as handle, output_path.open("w") as output:
        for line in handle:
            row = json.loads(line)
            if int(row["rank"]) in used:
                output.write(json.dumps(row) + "\n")
    return output_path
