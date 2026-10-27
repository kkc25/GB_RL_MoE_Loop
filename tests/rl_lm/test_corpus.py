"""End-to-end tests for rl_lm.corpus.build_rl_corpus (spec 4.6, task card B).

Synthetic fixtures only: an original LM corpus, a four-role controller
corpus, a confirmation corpus, and an additional exclusion JSONL act as
historical exposure; a local raw JSONL acts as the pinned new source.  Covers
variant-before-original near_hash rebuild, development_only marking, doc_hash
dedup, staging/commit behaviour, and the four-role token guarantee.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.fineweb import doc_fingerprint, sha256_file, text_fingerprint
from data.packed_lm import PackedLMDataset
from rl_lm.config import CorpusConfig
from rl_lm.corpus import CORPUS_BUILD_VERSION, CorpusBuildError, build_rl_corpus
from tests.rl_lm.test_corpus_manifest import (
    SEQ_LEN,
    make_confirmation_corpus,
    make_controller_corpus,
    make_lm_corpus,
)
from tests.test_controller_corpus import synthetic_docs, synthetic_tokenizer

ROLES = ("controller_train", "selection", "calibration", "newtest")
TOKENS = {"controller_train": 48, "selection": 32, "calibration": 32, "newtest": 32}


def write_source(path: Path, docs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for doc_id, text in docs:
            handle.write(json.dumps({"id": doc_id, "text": text}) + "\n")
    return path


def make_config(root: Path, source: Path, exclude_dirs, *, output="out", tokens=None,
                additional=()) -> CorpusConfig:
    tokenizer_path = root / "tokenizer.json"
    if not tokenizer_path.exists():
        synthetic_tokenizer().save(str(tokenizer_path))
    return CorpusConfig({
        "spec_version": "RL-LM-Corpus-v1.0",
        "source_jsonl": str(source),
        "source_id": "unit-test/synthetic:v0",
        "source_revision": "synthetic-rev-1",
        "tokenizer_path": str(tokenizer_path),
        "exclude_corpus_dirs": [str(p) for p in exclude_dirs],
        "additional_exclusion_jsonl": [str(p) for p in additional],
        "output_dir": str(root / output),
        "sequence_length": SEQ_LEN,
        "tokens": dict(tokens or TOKENS),
        "allow_existing_output": False,
    })


def document_rows(corpus_dir: Path):
    return [json.loads(line) for line in (corpus_dir / "documents.jsonl").read_text().splitlines()]


class BuildCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Historical exposures over the first 26 synthetic documents.
        self.lm = make_lm_corpus(self.root / "hist_lm", synthetic_docs(6)[:6], n_train=3, n_val=2)
        self.controller = make_controller_corpus(
            self.root / "hist_controller", synthetic_docs(40)[6:26]
        )
        self.confirmation = make_confirmation_corpus(
            self.root / "hist_confirmation", synthetic_docs(30)[26:30], used_ranks={0, 2}
        )
        self.additional = self.root / "additional.jsonl"
        extra = synthetic_docs(31)[30]
        self.additional.write_text(json.dumps({
            "doc_hash": doc_fingerprint(*extra),
            "text_hash": text_fingerprint(extra[1]),
        }) + "\n")
        self.history = [self.lm, self.controller, self.confirmation]

    def build(self, docs=None, **kwargs):
        docs = docs if docs is not None else synthetic_docs(80)
        source = write_source(self.root / "source.jsonl", docs)
        config = make_config(self.root, source, self.history,
                             additional=[self.additional], **kwargs)
        return build_rl_corpus(config, tokenizer=synthetic_tokenizer()), config

    def test_end_to_end_excludes_history_and_meets_tokens(self):
        coverage, config = self.build()
        out = Path(config.data["output_dir"])
        self.assertFalse(Path(str(out) + ".staging").exists())
        self.assertEqual(coverage["version"], CORPUS_BUILD_VERSION)
        self.assertFalse(coverage["development_only"])
        self.assertGreaterEqual(coverage["near_hash_rebuild"]["n_rebuilt"], 6)  # LM format
        for role in ROLES:
            self.assertGreaterEqual(
                coverage["role_tokens"][role]["actual"],
                coverage["role_tokens"][role]["requested"],
            )
            dataset = PackedLMDataset(out, role)
            self.assertGreater(len(dataset), 0)
        # Historical documents are absent from every role.  Only *actually
        # packed* historical documents count as exposed; derive them from the
        # collected exclusion identities rather than hardcoding fixture budgets.
        from rl_lm.corpus import collect_exclusions
        exclusions = collect_exclusions(self.history, [self.additional])
        source_docs = synthetic_docs(80)
        exposed_ids = {
            doc_id for doc_id, text in source_docs
            if doc_fingerprint(doc_id, text) in exclusions.doc_hashes
            or text_fingerprint(text) in exclusions.text_hashes
        }
        self.assertIn("doc-0", exposed_ids)    # LM format
        self.assertIn("doc-26", exposed_ids)   # confirmation used rank
        self.assertIn("doc-30", exposed_ids)   # additional exclusion jsonl
        self.assertTrue(any(f"doc-{i}" in exposed_ids for i in range(6, 26)))
        rows = document_rows(out)
        self.assertTrue(rows)
        self.assertFalse(exposed_ids & {row["id"] for row in rows})
        for role, counts in coverage["role_intersections"].items():
            self.assertEqual(counts, {"doc_hash": 0, "text_hash": 0, "near_hash": 0}, role)
        # Identity and coverage files are committed with the corpus.
        self.assertTrue((out / "exclusion_coverage.json").is_file())
        manifest = json.loads((out / "manifest.json").read_text())
        identity = manifest["rl_lm_corpus"]
        self.assertEqual(identity["source_jsonl_sha256"], sha256_file(self.root / "source.jsonl"))
        self.assertEqual(identity["exclusion_coverage_sha256"], sha256_file(out / "exclusion_coverage.json"))
        self.assertEqual(identity["config_sha256"], config.sha256)
        self.assertEqual(manifest["revision"], "synthetic-rev-1")
        formats = {s["format"] for s in coverage["exclusion_sources"]}
        self.assertEqual(
            formats, {"fineweb-pipeline-v1", "controller-corpus-v1", "fixed-depth-confirmation-v1"}
        )

    def test_deterministic_rebuild(self):
        coverage_a, config_a = self.build(output="out_a")
        coverage_b, config_b = self.build(output="out_b")
        for role in ROLES:
            for key in ("input_ids_sha256", "segment_ids_sha256", "doc_map_sha256"):
                self.assertEqual(
                    json.loads((Path(config_a.data["output_dir"]) / "manifest.json").read_text())["splits"][role][key],
                    json.loads((Path(config_b.data["output_dir"]) / "manifest.json").read_text())["splits"][role][key],
                )

    def test_variant_before_original_is_excluded_via_rebuilt_near_hash(self):
        original = ("orig-1", "The Quick Brown Fox jumps over thirteen lazy dogs")
        variant = ("var-1", "the quick brown fox, jumps over thirteen lazy dogs.")
        lm = make_lm_corpus(self.root / "hist_lm2", [original], n_train=1, n_val=0)
        docs = [variant] + synthetic_docs(70) + [original]
        source = write_source(self.root / "source2.jsonl", docs)
        config = make_config(self.root, source, [lm], output="out2")
        coverage = build_rl_corpus(config, tokenizer=synthetic_tokenizer())
        ids = {row["id"] for row in document_rows(Path(config.data["output_dir"]))}
        self.assertNotIn("orig-1", ids)
        # The variant's doc/text hashes are not in the historical record; only
        # the rebuilt near_hash group can exclude it.
        self.assertNotIn(variant[0], ids)
        self.assertFalse(coverage["development_only"])
        self.assertEqual(coverage["near_hash_rebuild"]["n_unresolved"], 0)
        self.assertGreaterEqual(coverage["near_hash_rebuild"]["n_rebuilt"], 1)

    def test_unresolvable_near_hash_marks_development_only(self):
        absent = ("ghost-1", "historical document absent from the new source")
        lm = make_lm_corpus(self.root / "hist_lm3", [absent], n_train=1, n_val=0)
        source = write_source(self.root / "source3.jsonl", synthetic_docs(80))
        config = make_config(self.root, source, [lm], output="out3")
        coverage = build_rl_corpus(config, tokenizer=synthetic_tokenizer())
        self.assertTrue(coverage["development_only"])
        self.assertGreaterEqual(coverage["near_hash_rebuild"]["n_unresolved"], 1)
        unresolved = {entry["doc_hash"] for entry in coverage["unresolved_near_hash"]}
        self.assertIn(doc_fingerprint(*absent), unresolved)
        manifest = json.loads((Path(config.data["output_dir"]) / "manifest.json").read_text())
        self.assertTrue(manifest["rl_lm_corpus"]["development_only"])

    def test_duplicate_doc_hash_keeps_first_and_counts(self):
        kept = synthetic_docs(80)[70]  # doc-70 is outside every history fixture
        docs = synthetic_docs(80) + [kept]
        source = write_source(self.root / "source4.jsonl", docs)
        config = make_config(self.root, source, self.history,
                             additional=[self.additional], output="out4")
        coverage = build_rl_corpus(config, tokenizer=synthetic_tokenizer())
        self.assertEqual(coverage["n_duplicate_doc_hash"], 1)
        self.assertEqual(coverage["n_source_lines"], 81)
        self.assertEqual(coverage["n_source_documents"], 80)
        rows = document_rows(Path(config.data["output_dir"]))
        dup_hash = doc_fingerprint(*kept)
        self.assertEqual(sum(1 for row in rows if row["doc_hash"] == dup_hash), 1)

    def test_nonempty_output_refused(self):
        out = self.root / "occupied"
        out.mkdir()
        (out / "stale.txt").write_text("keep me")
        source = write_source(self.root / "source5.jsonl", synthetic_docs(80))
        config = make_config(self.root, source, self.history, output="occupied")
        with self.assertRaises(CorpusBuildError):
            build_rl_corpus(config, tokenizer=synthetic_tokenizer())
        self.assertEqual((out / "stale.txt").read_text(), "keep me")

    def test_controller_train_shortfall_fails_without_commit(self):
        source = write_source(self.root / "source6.jsonl", synthetic_docs(80))
        config = make_config(
            self.root, source, self.history, output="out6",
            tokens={**TOKENS, "controller_train": 10**7},
        )
        with self.assertRaisesRegex(CorpusBuildError, "shortfall"):
            build_rl_corpus(config, tokenizer=synthetic_tokenizer())
        self.assertFalse((self.root / "out6").exists())
        self.assertFalse((self.root / "out6.staging").exists())

    def test_exact_role_shortfall_fails_without_commit(self):
        source = write_source(self.root / "source7.jsonl", synthetic_docs(30))
        config = make_config(self.root, source, [], output="out7",
                             additional=[self.additional],
                             tokens={**TOKENS, "newtest": 10**6})
        with self.assertRaises(RuntimeError):
            build_rl_corpus(config, tokenizer=synthetic_tokenizer())
        self.assertFalse((self.root / "out7").exists())
        self.assertFalse((self.root / "out7.staging").exists())

    def test_bad_source_rows_fail(self):
        bad_cases = [
            json.dumps({"id": "x"}),                      # missing text
            json.dumps({"id": 1, "text": "ok"}),          # non-string id
            json.dumps({"id": "x", "text": None}),        # non-string text
            "{not json",
        ]
        for index, line in enumerate(bad_cases):
            source = self.root / f"bad{index}.jsonl"
            source.write_text(line + "\n")
            config = make_config(self.root, source, self.history, output=f"out_bad{index}")
            with self.assertRaises(CorpusBuildError, msg=line):
                build_rl_corpus(config, tokenizer=synthetic_tokenizer())

    def test_missing_source_or_tokenizer_fail(self):
        config = make_config(self.root, self.root / "missing.jsonl", self.history, output="out8")
        with self.assertRaises(CorpusBuildError):
            build_rl_corpus(config, tokenizer=synthetic_tokenizer())
        source = write_source(self.root / "source9.jsonl", synthetic_docs(80))
        config = make_config(self.root, source, self.history, output="out9")
        config.data["tokenizer_path"] = str(self.root / "missing_tokenizer.json")
        with self.assertRaises(CorpusBuildError):
            build_rl_corpus(config)


class CliTests(unittest.TestCase):
    def test_cli_overrides_and_refuses_second_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lm = make_lm_corpus(root / "hist_lm", synthetic_docs(6))
            source = write_source(root / "source.jsonl", synthetic_docs(80))
            tokenizer_path = root / "tokenizer.json"
            synthetic_tokenizer().save(str(tokenizer_path))
            config_path = root / "corpus.json"
            config_path.write_text(json.dumps({
                "spec_version": "RL-LM-Corpus-v1.0",
                "source_jsonl": None,
                "source_id": "unit-test/synthetic:v0",
                "source_revision": None,
                "tokenizer_path": None,
                "exclude_corpus_dirs": [],
                "additional_exclusion_jsonl": [],
                "output_dir": "data/rl_lm_pair_v1/corpus",
                "sequence_length": SEQ_LEN,
                "tokens": TOKENS,
                "allow_existing_output": False,
            }))
            repo = Path(__file__).resolve().parents[2]
            command = [
                sys.executable, "-m", "experiments.rl_lm_corpus",
                "--config", str(config_path),
                "--source-jsonl", str(source),
                "--source-revision", "cli-rev-9",
                "--tokenizer", str(tokenizer_path),
                "--output-dir", str(root / "built"),
                "--exclude-corpus-dir", str(lm),
            ]
            first = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            coverage = json.loads(first.stdout)
            self.assertEqual(coverage["source"]["revision"], "cli-rev-9")
            self.assertTrue((root / "built" / "manifest.json").is_file())
            second = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("not empty", second.stderr)


if __name__ == "__main__":
    unittest.main()
