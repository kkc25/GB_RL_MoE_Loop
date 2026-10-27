import unittest

from rl_lm.config import CorpusConfig, RLConfig


class ConfigAndRoleTests(unittest.TestCase):
    def test_default_configs_validate(self):
        for name in ("pair_r1", "pair_r2"):
            config = RLConfig.load(f"configs/rl_lm/{name}.json")
            self.assertTrue(config.validate())
            self.assertEqual(config.to_dict(), config.data)
            self.assertEqual(len(config.sha256), 64)

    def test_unknown_root_and_nested_fields_fail(self):
        config = RLConfig.load("configs/rl_lm/pair_r1.json").to_dict()
        config["unknown"] = True
        with self.assertRaises(ValueError):
            RLConfig.from_dict(config)
        config = RLConfig.load("configs/rl_lm/pair_r1.json").to_dict()
        config["optimization"]["critic"] = {}
        with self.assertRaises(ValueError):
            RLConfig.from_dict(config)

    def test_invalid_groups_stage_and_training_role_fail(self):
        config = RLConfig.load("configs/rl_lm/pair_r1.json").to_dict()
        config["optimization"]["groups"] = 1
        with self.assertRaises(ValueError):
            RLConfig.from_dict(config).validate()
        config = RLConfig.load("configs/rl_lm/pair_r1.json").to_dict()
        config["stage"] = "r3"
        with self.assertRaises(ValueError):
            RLConfig.from_dict(config).validate()
        config = RLConfig.load("configs/rl_lm/pair_r1.json").to_dict()
        config["data"]["train_split"] = "newtest"
        with self.assertRaises(ValueError):
            RLConfig.from_dict(config).validate()

    def test_mixed_dtype_and_checkpoint_hash_fail(self):
        config = RLConfig.load("configs/rl_lm/pair_r1.json").to_dict()
        config["environment"]["frozen_compute_dtype"] = "float32"
        with self.assertRaises(ValueError):
            RLConfig.from_dict(config).validate()
        config = RLConfig.load("configs/rl_lm/pair_r1.json")
        with self.assertRaises(ValueError):
            config.validate_checkpoint_hash("0" * 64)

    def test_required_paths_and_corpus_exclusions(self):
        config = RLConfig.load("configs/rl_lm/pair_r1.json")
        with self.assertRaises(ValueError):
            config.validate_required_paths()
        # Null paths/revision and empty exclusion sources must always be
        # rejected; the checked-in corpus.json holds the filled task-E run
        # values, so the negative case is constructed in memory.
        corpus = CorpusConfig.load("configs/rl_lm/corpus.json").to_dict()
        corpus["source_jsonl"] = None
        with self.assertRaises(ValueError):
            CorpusConfig(corpus).validate_required()
        corpus = CorpusConfig.load("configs/rl_lm/corpus.json").to_dict()
        corpus["exclude_corpus_dirs"] = []
        with self.assertRaises(ValueError):
            CorpusConfig(corpus).validate_required()
        # The filled real-run config must pass the same validation.
        self.assertTrue(CorpusConfig.load("configs/rl_lm/corpus.json").validate_required())


if __name__ == "__main__":
    unittest.main()
