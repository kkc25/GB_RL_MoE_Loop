"""Exact resume must pin content identities, not only file paths (no training)."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_lm.checkpoint import save_atomic
from rl_lm.config import RLConfig
from rl_lm.trainer import Trainer
from tests.rl_lm.test_resume import toy_config_dict, toy_provider
from tests.rl_lm.tiny_lm import build_tiny_model


class ResumeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = RLConfig.from_dict(toy_config_dict(self.root))
        self.identity = dict(
            source_identity={'commit': 'a', 'tree': 'tree-a', 'untracked_python_sha256': {}},
            backbone_identity={'config_sha256': 'cfg-a', 'tokenizer_sha256': 'tok-a'},
            data_identity={'controller_train': {'packed_sha256': 'data-a', 'source': {'commit': 'a', 'tree': 'tree-a'}}},
        )
        self.original = self.make()
        self.path = save_atomic(self.root / 'checkpoint_latest.pt', self.original._checkpoint_payload())

    def make(self, *, resume=None, identities=None, model=None):
        return Trainer(self.config, model or build_tiny_model(), train_contexts=toy_provider(64, 1234),
                       device='cpu', resume_path=resume, **(identities or self.identity))

    def assert_rejected_unchanged(self, *, identities=None, model=None, match):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with self.assertRaisesRegex(ValueError, match):
            self.make(resume=self.path, identities=identities, model=model)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_identical_resume_and_metadata_only_commit_allowed(self):
        changed = copy.deepcopy(self.identity)
        changed['source_identity']['commit'] = 'b'
        changed['data_identity']['controller_train']['source']['commit'] = 'b'
        resumed = self.make(resume=self.path, identities=changed)
        self.assertTrue(resumed.resumed)
        self.assertEqual(resumed.successful_updates, 0)

    def test_changed_packed_data_rejected_before_restore(self):
        changed = copy.deepcopy(self.identity)
        changed['data_identity']['controller_train']['packed_sha256'] = 'data-b'
        with patch('rl_lm.trainer.restore_rng_state', side_effect=AssertionError('must reject before RNG restore')):
            self.assert_rejected_unchanged(identities=changed, match='data_identity mismatch')

    def test_changed_tokenizer_or_backbone_config_rejected(self):
        for field in ('tokenizer_sha256', 'config_sha256'):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.identity)
                changed['backbone_identity'][field] = 'changed'
                self.assert_rejected_unchanged(identities=changed, match='backbone_identity mismatch')

    def test_changed_gamma_rejected_with_identical_tensor_weights(self):
        model = build_tiny_model()
        model.gammas[model.config.adaptive_layers[-1]] += 0.1
        self.assert_rejected_unchanged(model=model, match='numerical_contract mismatch')

    def test_changed_code_rejected(self):
        changed = copy.deepcopy(self.identity)
        changed['source_identity']['tree'] = 'tree-b'
        self.assert_rejected_unchanged(identities=changed, match='source_identity mismatch')

    def test_missing_identity_not_silently_migrated(self):
        payload = self.original._checkpoint_payload()
        del payload['data_identity']
        save_atomic(self.path, payload)
        self.assert_rejected_unchanged(match='data_identity mismatch')

    def test_default_source_recomputed_not_copied_from_checkpoint(self):
        identities = copy.deepcopy(self.identity)
        identities['source_identity'] = None
        with patch('experiments.r1_closed_loop.source_identity', return_value={'tree': 'different-current-tree'}):
            self.assert_rejected_unchanged(identities=identities, match='source_identity mismatch')


if __name__ == '__main__':
    unittest.main()
