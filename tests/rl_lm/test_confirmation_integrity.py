import json, tempfile, sys, contextlib, io
from pathlib import Path
from unittest.mock import patch
import numpy as np, torch
from data.fineweb import SplitWriter
from tests.rl_lm.test_resume import toy_config_dict, toy_provider
from tests.rl_lm.tiny_lm import build_tiny_model
from rl_lm.config import RLConfig
from rl_lm.cache import cache_identity
from rl_lm.trainer import Trainer
from rl_lm.checkpoint import save_atomic
from train.recovery_common import sha256_file
from experiments.rl_lm_evaluate import main, verified_evaluation_scope
from lm_config import LMExperimentConfig, LMTrainConfig
import unittest

class ConfirmationIntegrityTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = self.tmp.name
        self.root = root = Path(t)
        corpus = root / 'corpus'
        corpus.mkdir()
        out = root / 'run'
        model = build_tiny_model()
        torch.save({'model': model.state_dict()}, root / 'model.pt')
        LMExperimentConfig(model=model.config, train=LMTrainConfig(data_dir=str(corpus))).save(root / 'model.json')
        (root / 'tokenizer.json').write_text('{}')
        splits = {}
        docs = []
        for rank, split in enumerate(['controller_train', 'selection', 'newtest']):
            w = SplitWriter(corpus, split, 8)
            w.add_document(rank, [5, 6, 7, 8, 9, 10, 11, 12])
            splits[split] = w.finish()
            docs.append({'rank': rank, 'role': split, 'near_hash': split})
        (corpus / 'documents.jsonl').write_text(''.join((json.dumps(d) + '\n' for d in docs)))
        (corpus / 'manifest.json').write_text(json.dumps({'splits': splits, 'rl_lm_corpus': {'development_only': False}, 'document_list_file': 'documents.jsonl', 'document_list_sha256': sha256_file(corpus / 'documents.jsonl')}))
        d = toy_config_dict(out)
        d['backbone'].update(checkpoint_path=str(root / 'model.pt'), config_path=str(root / 'model.json'))
        d['data'].update(corpus_dir=str(corpus), tokenizer_path=str(root / 'tokenizer.json'))
        d['evaluation']['bootstrap_repetitions'] = 10
        config = RLConfig.from_dict(d)
        data = {role: cache_identity(model, model.config, checkpoint_path=root / 'model.pt', config_path=root / 'model.json', tokenizer_path=root / 'tokenizer.json', data_dir=corpus, split=split, sampling_seed=config.get('data.sampling_seed'), backend_version=config.get('environment.training_backend'), head_contract=config.get('environment.head_contract'), expert_chunk=config.get('environment.expert_chunk')) for role, split in [('controller_train', 'controller_train'), ('selection', 'selection')]}
        backbone = {k: sha256_file(p) for k, p in [('checkpoint_sha256', root / 'model.pt'), ('config_sha256', root / 'model.json'), ('tokenizer_sha256', root / 'tokenizer.json')]}
        tr = Trainer(config, model, train_contexts=toy_provider(64, 1234), device='cpu', data_identity=data, backbone_identity=backbone)
        actor = save_atomic(out / 'candidate.pt', tr._checkpoint_payload())
        candidate = root / 'candidate.json'
        candidate.write_text(json.dumps({'confirmation_scope': verified_evaluation_scope(corpus, 'newtest', 1)[0], 'actor_checkpoint_sha256': sha256_file(actor), 'price_lambda': 0, 'backend': config.get('environment.deployment_backend'), 'head_contract': config.get('environment.head_contract'), 'selection_rule': 'fixed before newtest'}))
        self.config, self.model, self.corpus = (config, model, corpus)
        self.actor, self.candidate = (actor, candidate)

    def evaluate(self, sequences=1, overwrite=True):
        with patch('experiments.rl_lm_evaluate.resolve_config', return_value=self.config), patch('experiments.rl_lm_evaluate.load_backbone', return_value=(self.model, {})), patch.object(sys, 'argv', ['evaluate', '--config', 'unused', '--actor-checkpoint', str(self.actor), '--split', 'newtest', '--candidate-manifest', str(self.candidate), '--sequences', str(sequences), '--output', str(self.root / 'report.json')] + (['--overwrite-output'] if overwrite else [])), contextlib.redirect_stdout(io.StringIO()):
            main()
        return json.loads((self.root / 'report.json').read_text())

    def test_verified_real_evaluation_and_same_size_tampering(self):
        before = self.evaluate()
        self.assertEqual(before['sequences'], 1)
        self.assertEqual(before['candidate_manifest_sha256'], sha256_file(self.candidate))
        self.assertEqual(before['candidate_manifest_content']['confirmation_scope'], before['evaluated_data_identity'])
        report = (self.root / 'report.json').read_bytes()
        for name in ('newtest.input_ids.u16.bin', 'newtest.segment_ids.i32.bin', 'newtest.doc_map.jsonl', 'documents.jsonl'):
            with self.subTest(name=name):
                path = self.corpus / name
                original = path.read_bytes()
                changed = bytearray(original)
                changed[2] ^= 1
                path.write_bytes(changed)
                try:
                    with patch('experiments.rl_lm_evaluate._legacy_nll') as load, self.assertRaisesRegex(ValueError, 'integrity check|hash mismatch'):
                        self.evaluate()
                    load.assert_not_called()
                    self.assertEqual(report, (self.root / 'report.json').read_bytes())
                finally:
                    path.write_bytes(original)
        self.assertEqual(before['online'], self.evaluate()['online'])

    def test_legacy_candidate_and_changed_denominator_fail(self):
        with self.assertRaisesRegex(ValueError, 'frozen confirmation scope'):
            self.evaluate(sequences=2)
        payload = json.loads(self.candidate.read_text())
        del payload['confirmation_scope']
        self.candidate.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'legacy candidate'):
            self.evaluate()

    def test_scope_rejects_invalid_count_or_changed_sequence_ids(self):
        for count in (0, -1, 2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                verified_evaluation_scope(self.corpus, 'newtest', count)
        payload = json.loads(self.candidate.read_text())
        payload['confirmation_scope']['sequence_ids'] = [9]
        self.candidate.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'confirmation scope mismatch'):
            self.evaluate()

    def test_updated_data_manifest_does_not_change_frozen_scope(self):
        path = self.corpus / 'newtest.input_ids.u16.bin'
        tokens = np.fromfile(path, dtype=np.uint16)
        tokens[1] = 17
        tokens.tofile(path)
        manifest_path = self.corpus / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['splits']['newtest']['input_ids_sha256'] = sha256_file(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'confirmation scope mismatch'):
            self.evaluate()

    def test_existing_report_requires_explicit_overwrite_before_model_load(self):
        report = self.root / 'report.json'
        report.write_bytes(b'historical report')
        with patch('experiments.rl_lm_evaluate.load_backbone') as load:
            with self.assertRaisesRegex(FileExistsError, 'report already exists'):
                self.evaluate(overwrite=False)
            load.assert_not_called()
        self.assertEqual(report.read_bytes(), b'historical report')

if __name__ == '__main__':
    unittest.main()
