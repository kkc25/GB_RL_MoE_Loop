"""A stale approval must never authorize different config, counts or bytes."""
import copy
import json
import shutil
import unittest
from unittest.mock import patch

from experiments import rl_lm_preflight, rl_lm_train
from rl_lm import run_contract
from rl_lm.config import RLConfig
from tests.rl_lm import test_cache_integrity
from tests.rl_lm.test_resume import toy_config_dict


class RunContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_cache_integrity.CacheIntegrityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        d = toy_config_dict(f.root / 'run')
        d['backbone'].update(checkpoint_path=str(f.root / 'model.pt'), config_path=str(f.root / 'config.json'))
        d['data'].update(corpus_dir=str(f.root), tokenizer_path=str(f.root / 'tokenizer.json'),
                         cache_root=str(f.root / 'pair'), train_split='train', selection_split='selection',
                         calibration_split='calibration', train_states=8, selection_states=8, calibration_states=8)
        self.config = RLConfig.from_dict(d)
        for split in ('selection', 'calibration'):
            shutil.copytree(f.path, f.root / 'pair' / split)
        self.identity_patch = patch.object(run_contract, 'cache_identity', return_value=f.identity)
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)

    def preflight(self):
        with patch.object(rl_lm_preflight, 'resolve_config', return_value=self.config), \
             patch.object(rl_lm_preflight, 'load_backbone', return_value=(self.fixture.model, None)), \
             patch('sys.argv', ['preflight', '--config', 'unused']), patch('builtins.print'):
            rl_lm_preflight.main()
        return json.loads((self.fixture.root / 'run' / 'preflight.json').read_text())

    def train_rejects(self, pattern):
        with patch.object(rl_lm_train, 'resolve_config', return_value=self.config), \
             patch.object(rl_lm_train, 'load_backbone', return_value=(self.fixture.model, None)), \
             patch.object(rl_lm_train, 'Trainer') as trainer, \
             patch('sys.argv', ['train', '--config', 'unused']):
            with self.assertRaisesRegex((RuntimeError, ValueError), pattern):
                rl_lm_train.main()
            trainer.assert_not_called()

    def test_fresh_preflight_failure_invalidates_old_ready_before_trainer(self):
        self.assertEqual(self.preflight()['status'], 'ready')
        self.config.data['data']['train_states'] = 64
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            self.preflight()
        report = json.loads((self.fixture.root / 'run' / 'preflight.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertNotIn('contract_sha256', report)
        self.train_rejects('requires')

    def test_trainer_independently_rechecks_all_counts(self):
        for key in ('train_states', 'selection_states', 'calibration_states'):
            with self.subTest(key=key):
                self.preflight()
                self.config.data['data'][key] = 64
                self.train_rejects('count mismatch')
                self.config.data['data'][key] = 8

    def test_changed_config_and_payload_cannot_reuse_approval(self):
        report = self.preflight()
        caches, contract = run_contract.load_role_caches(self.config, self.fixture.model)
        run_contract.require_approval(report, contract)
        changed = copy.deepcopy(contract)
        changed['roles']['selection']['verified_identity']['content_sha256'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            run_contract.require_approval(report, changed)
        self.config.data['environment']['head_chunk'] += 1
        self.train_rejects('does not match')

    def test_resealed_payload_replacement_requires_new_approval(self):
        import numpy as np
        self.preflight()
        path = self.fixture.path / 'targets.npy'
        targets = np.load(path).copy()
        targets[0] = (targets[0] + 1) % self.fixture.model.config.vocab_size
        np.save(path, targets)
        self.fixture.reseal()
        self.train_rejects('does not match')

    def test_missing_cache_clears_approval(self):
        self.preflight()
        (self.fixture.root / 'pair' / 'calibration' / 'manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.preflight()
        self.train_rejects('requires')

    def test_device_only_resume_preserves_approval_but_numerics_do_not(self):
        report = self.preflight()
        self.config.data['runtime']['device'] = 'cuda'
        _, contract = run_contract.load_role_caches(self.config, self.fixture.model)
        run_contract.require_approval(report, contract)
        self.config.data['environment']['head_chunk'] += 1
        _, changed = run_contract.load_role_caches(self.config, self.fixture.model)
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            run_contract.require_approval(report, changed)

    def test_old_ready_or_tampered_approval_is_rejected(self):
        report = self.preflight()
        for invalid in ({'status': 'ready'}, report | {'contract_sha256': 'bad'},
                        report | {'contract_schema': 'legacy'}):
            with self.subTest(invalid=invalid.get('contract_schema')):
                with self.assertRaisesRegex(RuntimeError, 'contract'):
                    run_contract.require_approval(invalid)

    def test_single_loader_enforces_every_role_count(self):
        # Both single entry points use this loader; exercise the real payload
        # verification while isolating only the environment identity builder.
        from tests.rl_lm.test_single_adapt import _toy_config
        config = _toy_config(self.fixture.root / 'adapt')
        config.data['data'].update(self.config.data['data'])
        config.data['data']['pair_cache_root'] = str(self.fixture.root / 'pair')
        config.data['backbone'].update(self.config.data['backbone'])
        with patch('rl_lm.single_config.single_cache_identity', return_value=self.fixture.identity):
            run_contract.load_role_caches(config, self.fixture.model, single=True)
            for key in ('train_states', 'selection_states', 'calibration_states'):
                with self.subTest(key=key):
                    config.data['data'][key] = 64
                    with self.assertRaisesRegex(ValueError, 'count mismatch'):
                        run_contract.load_role_caches(config, self.fixture.model, single=True)
                    config.data['data'][key] = 8

    def test_single_derivation_rejects_unconfigured_role_descriptively(self):
        import torch
        from experiments.rl_lm_single_adapt import derive_single_cache
        from tests.rl_lm.test_single_adapt import _toy_config
        config = _toy_config(self.fixture.root / 'adapt')
        config.data['data'].update(train_split='controller_train',
            pair_cache_root=str(self.fixture.root / 'pair'), cache_root=str(self.fixture.root / 'single'))
        with self.assertRaisesRegex(ValueError, 'unconfigured training role: train'):
            derive_single_cache(config, self.fixture.model, 'train', torch.device('cpu'))
        self.assertFalse((self.fixture.root / 'single').exists())

    def test_single_derivation_rejects_wrong_count_before_writing(self):
        import torch
        from experiments.rl_lm_single_adapt import derive_single_cache
        from tests.rl_lm.test_single_adapt import _toy_config
        config = _toy_config(self.fixture.root / 'adapt')
        config.data['data'].update(train_split='train', train_states=64,
            pair_cache_root=str(self.fixture.root / 'pair'), cache_root=str(self.fixture.root / 'single'))
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            derive_single_cache(config, self.fixture.model, 'train', torch.device('cpu'))
        self.assertFalse((self.fixture.root / 'single').exists())

    def test_adaptation_checkpoint_binds_all_role_contract(self):
        import torch
        from experiments.rl_lm_single_adapt import AdaptTrainer
        from tests.rl_lm.test_single_adapt import _toy_config
        config = _toy_config(self.fixture.root / 'adapt', batch=8)
        config.data['data'].update(self.config.data['data'])
        caches, contract = run_contract.load_role_caches(config, self.fixture.model)
        identity = run_contract.run_contract_identity(contract)
        kwargs = dict(train_cache=caches['controller_train'], device=torch.device('cpu'),
                      run_dir=self.fixture.root / 'adapt', run_contract_identity=identity)
        trainer = AdaptTrainer(config, self.fixture.model, **kwargs)
        checkpoint = trainer._save()
        self.assertEqual(trainer._payload()['run_contract_identity'], identity)
        AdaptTrainer(config, self.fixture.model, resume_path=checkpoint, **kwargs)
        changed = copy.deepcopy(contract)
        changed['roles']['calibration']['verified_identity']['content_sha256'] = 'replacement'
        kwargs['run_contract_identity'] = run_contract.run_contract_identity(changed)
        with self.assertRaisesRegex(ValueError, 'run contract mismatch'):
            AdaptTrainer(config, self.fixture.model, resume_path=checkpoint, **kwargs)

    def test_adaptation_cli_rejects_changed_contract_before_ready_or_references(self):
        import torch
        from experiments import rl_lm_single_adapt as adapt
        from tests.rl_lm.test_single_adapt import _toy_config
        output = self.fixture.root / 'adapt'
        config = _toy_config(output, batch=8)
        config.data['data'].update(self.config.data['data'])
        caches, contract = run_contract.load_role_caches(config, self.fixture.model)
        trainer = adapt.AdaptTrainer(config, self.fixture.model,
            train_cache=caches['controller_train'], device=torch.device('cpu'), run_dir=output,
            run_contract_identity=run_contract.run_contract_identity(contract))
        checkpoint = trainer._save()
        reference = output / 'pair_teacher_selection.json'
        reference.write_bytes(b'historical reference')
        metrics = output / 'metrics.jsonl'
        metrics.write_bytes(b'historical metrics')
        protected = {path: path.read_bytes() for path in (checkpoint, reference, metrics)}
        (output / 'preflight.json').write_text(json.dumps(run_contract.approval_record(contract)))
        changed = copy.deepcopy(contract)
        changed['roles']['calibration']['verified_identity']['content_sha256'] = 'changed'
        with patch.object(adapt, 'resolve_single_config', return_value=config), \
             patch.object(adapt, 'load_backbone', return_value=(self.fixture.model, None)), \
             patch.object(config, 'validate'), patch.object(adapt, 'derive_single_cache'), \
             patch.object(adapt, 'load_role_caches', return_value=(caches, changed)), \
             patch.object(adapt, '_selection_tensors') as selection, \
             patch.object(adapt, 'pair_teacher_selection_nll') as reference_nll, \
             patch('sys.argv', ['adapt', '--config', 'unused', '--resume', str(checkpoint)]):
            with self.assertRaisesRegex(ValueError, 'run contract mismatch'):
                adapt.main()
            selection.assert_not_called()
            reference_nll.assert_not_called()
        self.assertEqual(json.loads((output / 'preflight.json').read_text())['status'], 'failed')
        for path, expected in protected.items():
            self.assertEqual(path.read_bytes(), expected)

    def test_single_training_checkpoint_binds_contract_in_data_identity(self):
        from experiments.rl_lm_single_train import SingleTrainer
        from tests.rl_lm import test_single_rloo
        _, contract = run_contract.load_role_caches(self.config, self.fixture.model)
        identity = run_contract.run_contract_identity(contract)
        path = self.fixture.root / 'single_run'
        config_data = test_single_rloo.toy_config_dict(path)
        trainer = test_single_rloo.make_trainer(config_data)
        trainer.data_identity['run_contract'] = identity
        checkpoint = trainer._save_checkpoint('latest')
        changed = identity | {'contract_sha256': 'changed-selection-or-calibration'}
        with self.assertRaisesRegex(ValueError, 'data_identity'):
            SingleTrainer(test_single_rloo.SingleRLConfig(config_data), test_single_rloo.build_tiny_model(),
                          train_contexts=test_single_rloo.ToyProvider(64), device='cpu',
                          resume_path=checkpoint, data_identity={'run_contract': changed})

    def test_model_failure_replaces_old_approval(self):
        self.preflight()
        with patch.object(rl_lm_preflight, 'resolve_config', return_value=self.config), \
             patch.object(rl_lm_preflight, 'load_backbone', side_effect=ValueError('bad model')), \
             patch('sys.argv', ['preflight', '--config', 'unused']):
            with self.assertRaisesRegex(ValueError, 'bad model'):
                rl_lm_preflight.main()
        self.train_rejects('requires')
