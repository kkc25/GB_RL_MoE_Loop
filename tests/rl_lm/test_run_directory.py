"""Fresh runs must not corrupt committed evidence when .pt files are absent."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments import rl_lm_prepare, rl_lm_preflight, rl_lm_train, rl_lm_single_train, rl_lm_single_adapt
from rl_lm.run_directory import guard_run_directory, PREFLIGHT_FILES
from tests.rl_lm import test_resume, test_single_rloo


class RunDirectoryTests(unittest.TestCase):
    def test_new_and_preflight_only_directories_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard_run_directory(root / "new")
            self.assertFalse((root / "new").exists())
            for name in PREFLIGHT_FILES:
                (root / name).write_text("{}")
            guard_run_directory(root)

    def test_each_archived_artifact_blocks_fresh_run_without_checkpoint(self):
        for name in ("metrics.jsonl", "evaluations.jsonl", "selection_manifest.json",
                     "checkpoint_manifest.json", "cost_ledger.json", "final_summary.md",
                     "final_summary.json", "unrelated.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / name
                path.write_bytes(b"historical evidence\n")
                with self.assertRaises(FileExistsError):
                    guard_run_directory(tmp)
                self.assertEqual(path.read_bytes(), b"historical evidence\n")
                guard_run_directory(tmp, resume=True)

    def test_all_cli_guards_preserve_archives_before_model_load_or_write(self):
        for module, resolver in ((rl_lm_prepare, "resolve_config"),
                                 (rl_lm_single_adapt, "resolve_single_config"),
                                 (rl_lm_preflight, "resolve_config"),
                                 (rl_lm_train, "resolve_config"),
                                 (rl_lm_single_train, "resolve_single_config")):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                before = {"metrics.jsonl": b"old metrics\n", "resolved_config.json": b"old config\n",
                          "preflight.json": b'{"status":"ready"}', "final_summary.md": b"old summary\n"}
                for name, data in before.items():
                    (root / name).write_bytes(data)
                config = {"output_dir": tmp}
                with patch.object(module, resolver, return_value=config), \
                     patch.object(module, "load_backbone") as load, \
                     patch("sys.argv", [module.__name__, "--config", "unused.json"]):
                    with self.assertRaisesRegex(FileExistsError, "fresh directory"):
                        module.main()
                    load.assert_not_called()
                self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_direct_trainers_reject_archive_without_checkpoint(self):
        for fixture in (test_resume, test_single_rloo):
            with self.subTest(fixture=fixture.__name__), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "metrics.jsonl"
                path.write_bytes(b"old evidence\n")
                with self.assertRaises(FileExistsError):
                    fixture.make_trainer(fixture.toy_config_dict(tmp), stop_after=1)
                self.assertEqual(path.read_bytes(), b"old evidence\n")

    def test_single_prepared_reference_cache_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("reference_ce.npy", "reference_ce.json"):
                (root / name).write_bytes(b"prepared reference")
            guard_run_directory(root, additional_allowed=("reference_ce.npy", "reference_ce.json"))

    def test_single_exact_resume_rejects_changed_input_identity_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_data = test_single_rloo.toy_config_dict(tmp)
            original = test_single_rloo.make_trainer(config_data, stop_after=1)
            checkpoint = original._save_checkpoint("latest")
            before = {p.name: p.read_bytes() for p in Path(tmp).iterdir()}
            for field in ("backbone_identity", "adapted_identity", "data_identity"):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    rl_lm_single_train.SingleTrainer(
                        test_single_rloo.SingleRLConfig(config_data), test_single_rloo.build_tiny_model(),
                        train_contexts=test_single_rloo.ToyProvider(64),
                        device="cpu", resume_path=checkpoint,
                        **{field: {"actual_bytes_sha256": "changed"}},
                    )
            self.assertEqual({p.name: p.read_bytes() for p in Path(tmp).iterdir()}, before)

    def test_single_exact_resume_rejects_changed_numerical_and_feature_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = test_single_rloo.make_trainer(test_single_rloo.toy_config_dict(tmp), stop_after=1)
            for field in ("numerical_contract", "action_contract", "feature_schema", "feature_dim"):
                payload = trainer._checkpoint_payload()
                payload[field] = None
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    trainer._validate_resume_identities(payload)

    def test_single_identical_resume_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = test_single_rloo.toy_config_dict(tmp)
            trainer = test_single_rloo.make_trainer(config, stop_after=1)
            checkpoint = trainer._save_checkpoint("latest")
            resumed = test_single_rloo.make_trainer(config, resume_path=checkpoint, stop_after=1)
            self.assertEqual(resumed.run_id, trainer.run_id)
            for key, value in trainer.actor.state_dict().items():
                torch.testing.assert_close(value, resumed.actor.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
