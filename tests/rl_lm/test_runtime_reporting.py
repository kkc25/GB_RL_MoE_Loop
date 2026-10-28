"""Runtime cost reporting is measured, incremental, resume-safe and not utilization."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.r1_closed_loop import source_identity
from rl_lm.ledger import RLLedger
from rl_lm.trainer import Trainer
from tests.rl_lm.test_resume import make_trainer, toy_config_dict


class RuntimeReportingTests(unittest.TestCase):
    def clock_trainer(self, device):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device(device)
        trainer.ledger = RLLedger()
        trainer._cost_clock = 10.0
        trainer._wall_time_seconds = 0.0
        trainer._cost_finished = False
        return trainer

    def test_cpu_elapsed_never_claims_gpu_time(self):
        trainer = self.clock_trainer('cpu')
        with patch('rl_lm.trainer.time.monotonic', return_value=3610.0):
            self.assertEqual(trainer._account_runtime(), 3600.0)
        self.assertEqual(trainer.ledger.gpu_hours, 0.0)

    def test_mock_cuda_incremental_and_finish_idempotent(self):
        trainer = self.clock_trainer('cuda:3')
        with patch('rl_lm.trainer.time.monotonic', side_effect=[1810., 3610.]):
            trainer._account_runtime()
            self.assertEqual(trainer.ledger.gpu_hours, .5)
            trainer._account_runtime(finish=True)
            trainer._account_runtime(finish=True)  # clock no longer read
        self.assertEqual(trainer.ledger.gpu_hours, 1.)
        self.assertEqual(trainer._wall_time_seconds, 3600.)
        self.assertEqual(trainer._runtime_accounting()['current_segment_reserved_cuda_devices'], 1)

    def test_real_checkpoint_resume_cost_and_final_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = toy_config_dict(tmp)
            trainer = make_trainer(config, stop_after=1)
            # Keep each mocked segment wholly synthetic: adding an integer to a
            # real fractional monotonic origin can round across float exponents.
            trainer._cost_clock = 0.
            with patch('rl_lm.trainer.time.monotonic', return_value=trainer._cost_clock + 12.):
                trainer._initialize_run()
            checkpoint = Path(tmp) / 'checkpoint_latest.pt'
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload['runtime_accounting']['wall_time_seconds'], 12.)
            # Previously measured CUDA costs survive a CPU resume; no fabricated history.
            payload['ledger']['gpu_hours'] = .25
            torch.save(payload, checkpoint)
            resumed = make_trainer(config, resume_path=checkpoint, stop_after=1)
            resumed._cost_clock = 0.
            with patch('rl_lm.trainer.time.monotonic', return_value=resumed._cost_clock + 6.):
                resumed._finish('stopped_at_cap')
                resumed._finish('stopped_at_cap')
            report = json.loads((Path(tmp) / 'cost_ledger.json').read_text())
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(report['wall_time_seconds'], 18.)
            self.assertEqual(report['gpu_hours'], .25)
            self.assertEqual(saved['runtime_accounting']['wall_time_seconds'], 18.)
            summary = (Path(tmp) / 'final_summary.md').read_text()
            self.assertIn('device: `cpu`', summary)
            self.assertNotIn('真实 100M checkpoint 上的 GPU 训练', summary)
            self.assertEqual(trainer.source_identity['entrypoint'], 'rl_lm/trainer.py')
            # Mock only accounting's device; no CUDA allocation or training involved.
            cuda_resumed = make_trainer(config, resume_path=checkpoint, stop_after=1)
            cuda_resumed.device = torch.device('cuda:3')
            cuda_resumed._cost_clock = 0.
            with patch('rl_lm.trainer.time.monotonic', return_value=cuda_resumed._cost_clock + 3600.):
                cuda_resumed._account_runtime(finish=True)
                cuda_resumed._account_runtime(finish=True)
            self.assertEqual(cuda_resumed.ledger.gpu_hours, 1.25)
            self.assertEqual(cuda_resumed._wall_time_seconds, 3618.)
            # Legacy checkpoints lack measured wall duration: retain their saved
            # cost but do not infer time from zero GPU-hours or old report files.
            saved.pop('runtime_accounting')
            torch.save(saved, checkpoint)
            legacy = make_trainer(config, resume_path=checkpoint, stop_after=1)
            self.assertEqual(legacy._wall_time_seconds, 0.)
            self.assertEqual(legacy.ledger.gpu_hours, .25)

    def test_entrypoint_can_be_explicit_without_changing_helper_default(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(source_identity()['entrypoint'], 'experiments/r1_closed_loop.py')
        identity = source_identity(entrypoint=root / 'experiments/rl_lm_train.py')
        self.assertEqual(identity['entrypoint'], 'experiments/rl_lm_train.py')
        import hashlib
        self.assertEqual(identity['entrypoint_sha256'], hashlib.sha256((root / identity['entrypoint']).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
