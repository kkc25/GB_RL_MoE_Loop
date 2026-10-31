"""Failure-path integration for the frozen v2 experiment CLI; CPU only."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments import mixtral_loop_scale as cli
from mixtral_lab.loop_scale_data import digest, sha256_file
from mixtral_lab.loop_scale import digest as checkpoint_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ScaleCLIGuards(unittest.TestCase):
    def test_historical_original_ids_also_exclude_newly_named_source_slices(self):
        path='runs/mixtral_trainable_loop_v1/trainable_loop_v1_seed17_exportfix_20261030T080210Z/fit/report.json'
        prior=cli.read_json(path)
        exclusions=cli.historical_exclusions(path)
        for role in ('train','development','selection','calibration'):
            for original in prior['identity'][role]['selected_document_ids']:
                self.assertIn(original,exclusions['excluded_document_ids'])
                self.assertIn(original,exclusions['excluded_source_ids'])

    def test_delivered_configs_share_scientific_protocol_and_protect_inputs(self):
        generic=cli.validate_config(cli.read_json('configs/mixtral/loop_scale_v2.json'))
        local=cli.validate_config(cli.read_json('local_configs/loop_scale_v2.json'))
        for key in ('model_path','tokenizer_path','source_files','freeze_dir','run_root'):
            generic.pop(key);local.pop(key)
        self.assertEqual(generic,local)
        for setting,value in [('run_root','/models/Mixtral-8x7B-v0.1/run'),
                              ('freeze_dir','/data'),('run_root','/data/mixtral-loop-scale-v2')]:
            bad=cli.read_json('configs/mixtral/loop_scale_v2.json');bad[setting]=value
            with self.assertRaises(ValueError):cli.validate_config(bad)

    def test_final_test_requires_canonical_valid_lock_before_model_loading(self):
        code={'commit':'a'*40,'sources':{}}
        with tempfile.TemporaryDirectory() as td:
            c=dict(run_root=td)
            index=dict(freeze_sha256='f'*64)
            prepared=dict(code=code,config=c,freeze_sha256=index['freeze_sha256'])
            write(Path(td)/'preparation.json',prepared)
            args=SimpleNamespace(stage='final-test',lock=None)
            with patch.object(cli,'frozen',return_value=(Path(td)/'freeze.json',index)), \
                 patch.object(cli,'gpu_adapter',side_effect=AssertionError('model loaded before final lock')):
                with self.assertRaisesRegex(ValueError,'canonical final lock'):
                    cli.execute_prepared(args,c,code)

    def test_locked_checkpoint_cannot_be_swapped_after_finalization(self):
        code={'commit':'a'*40,'sources':{}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'cells'/'100000'/'r2_full'/'seed17';out.mkdir(parents=True)
            checkpoint=out/'gated-epoch1.pt';checkpoint.write_bytes(b'first frozen bytes')
            c=dict(run_root=td)
            index=dict(freeze_sha256='f'*64)
            lock=dict(code=code,config_sha256=digest(c),freeze_sha256=index['freeze_sha256'],
                cells=[dict(output=str(out),artifacts={checkpoint.name:sha256_file(checkpoint)})])
            lock['sha256']=digest(lock)
            path=root/'final_selection.json';write(path,lock)
            self.assertEqual(cli.check_lock(path,c,code,index),lock)
            checkpoint.write_bytes(b'replacement bytes')
            with self.assertRaisesRegex(ValueError,'checkpoint/report changed'):
                cli.check_lock(path,c,code,index)

    def test_receipt_preserves_unknown_attempt_and_refuses_changed_run_identity(self):
        with tempfile.TemporaryDirectory() as td:
            c=dict(run_root=td,suite_wall_seconds=100)
            code={'commit':'a'*40,'sources':{}}
            receipt=cli.StageReceipt(c,code,'cache')
            original=deepcopy(receipt.value)
            resumed=cli.StageReceipt(c,code,'cache',resume=True)
            self.assertEqual(resumed.value['created'],original['created'])
            self.assertEqual(resumed.value['attempts'][0]['status'],'unknown_interruption')
            self.assertGreaterEqual(resumed.value['attempts'][0]['wall_seconds'],0.)
            resumed.finish('failed',error='controlled test')
            changed=dict(c,suite_wall_seconds=101)
            with self.assertRaisesRegex(ValueError,'run_root identity changed'):
                cli.StageReceipt(changed,code,'cache',resume=True)

    def test_preflight_compares_json_normalized_identity_without_ignoring_runtime_changes(self):
        with tempfile.TemporaryDirectory() as td:
            c=dict(run_root=td)
            code={'commit':'a'*40,'sources':{}}
            index=dict(freeze_sha256='f'*64)
            model=dict(model_config=dict(id2label={0:'LABEL_0',1:'LABEL_1'}),
                runtime=dict(cudnn_tf32=False))
            write(Path(td)/'preflight.json',dict(status='complete',code=code,
                freeze_sha256=index['freeze_sha256'],model=model))
            self.assertEqual(cli.require_preflight(c,code,index,model)['status'],'complete')
            changed=deepcopy(model);changed['runtime']['cudnn_tf32']=True
            with self.assertRaisesRegex(ValueError,'runtime-bound preflight required'):
                cli.require_preflight(c,code,index,changed)

    def test_actual_loaded_tokenizer_must_match_frozen_bytes(self):
        identity=dict(model=dict(files={'config.json':'a'*64,'weights.safetensors':'b'*64}),
            tokenizer=dict(files={'tokenizer.json':'c'*64}))
        index=dict(model=dict(files={k:dict(sha256=v) for k,v in identity['model']['files'].items()}),
            tokenizer=dict(files={'tokenizer.json':dict(sha256='d'*64)}))
        c=dict(require_gpu_name='A6000')
        with ExitStack() as stack:
            for name,value in [('is_available',True),('device_count',4),('get_device_name','RTX A6000')]:
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            stack.enter_context(patch('experiments.mixtral_experiment.load_local_inputs',return_value={}))
            stack.enter_context(patch('experiments.mixtral_experiment.load_local_model',
                return_value=(None,None,SimpleNamespace(),identity)))
            with self.assertRaisesRegex(ValueError,'tokenizer'):
                cli.gpu_adapter(c,index)

    def test_formal_main_enforces_and_restores_frozen_tf32_flags(self):
        with tempfile.TemporaryDirectory() as td:
            c=dict(run_root=str(Path(td)/'run'),suite_wall_seconds=100,allow_tf32=False,
                   model_path=str(Path(td)/'model'),tokenizer_path=str(Path(td)/'tokenizer'),
                   freeze_dir=str(Path(td)/'frozen'),source_files=[str(Path(td)/'source.jsonl')])
            path=Path(td)/'config.json';write(path,c)
            old_matmul=torch.backends.cuda.matmul.allow_tf32
            old_cudnn=torch.backends.cudnn.allow_tf32
            old_precision=torch.get_float32_matmul_precision()
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=True
            def checked(*args):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                return {'status':'complete'}
            try:
                with patch.object(cli,'validate_config',return_value=c), \
                     patch.object(cli,'source_identity',return_value={'commit':'a'*40,'sources':{}}), \
                     patch.object(cli,'execute_prepared',side_effect=checked):
                    cli.main(['--config',str(path),'--stage','fit','--expected-commit','a'*40])
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
            finally:
                torch.backends.cuda.matmul.allow_tf32=old_matmul
                torch.backends.cudnn.allow_tf32=old_cudnn
                torch.set_float32_matmul_precision(old_precision)


def result(nll):
    return dict(nll=nll,per_document=[dict(document_id=f'd{i}',tokens=10,
        candidate_nll=nll,native_nll=1.) for i in range(4)])


def cell_fixture(c,budget,cell,seed=17,*,forced=.95,gate=.99,matched=False):
    path=cli.cell_output(c,budget,cell,seed,matched)
    report=dict(status='complete',identity=dict(config_sha256=digest(c),budget=budget,cell=cell,seed=seed),
        selection_results=dict(native=result(1.),forced=result(forced),gate=result(gate)),
        adapter_exposures=180000,adapter_updates=704,gate_updates=80)
    report['sha256']=checkpoint_digest(report)
    write(path/'report.json',report)
    (path/'candidate-epoch1.pt').write_bytes(b'fixed checkpoint'+str(seed).encode())
    return report


class ScaleScreenAndFinalization(unittest.TestCase):
    def config(self,root):
        return dict(run_root=str(root),minimum_gain=.001,seeds=[17,29,43])

    def cohort(self,c,budget,*,gain=.05,gate_gain=0.):
        for cell in cli.CELLS:
            cell_fixture(c,budget,cell,forced=1.-gain if cell=='r2_full' else 1.01,
                         gate=1.-gate_gain if cell=='r2_reduced' else 1.)

    def test_negative_100k_allows_predeclared_300k_and_selectors_keep_distinct_winners(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td)/'negative');self.cohort(c,100000,gain=-.01)
            screen=cli.screen(c,100000)
            self.assertTrue(screen['allow_next_budget'])
            args=SimpleNamespace(budget=300000,seed=17,cell='r2_full',matched_exposure=False)
            cli.fit_guard(args,c)
            c=self.config(Path(td)/'positive');self.cohort(c,100000,gain=.05,gate_gain=.02)
            screen=cli.screen(c,100000)
            self.assertTrue(screen['allow_next_budget'])
            self.assertEqual((screen['winner']['cell'],screen['winner']['mode']),('r2_reduced','gate'))
            self.assertEqual((screen['forced_winner']['cell'],screen['forced_winner']['mode']),('r2_full','forced'))
            self.assertEqual(cli.screen(c,100000),screen)

    def test_300k_requires_complete_three_cell_100k_cohort(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(td)
            cell_fixture(c,100000,'r2_full')
            cell_fixture(c,100000,'r2_reduced')
            args=SimpleNamespace(budget=300000,seed=17,cell='r2_full',matched_exposure=False)
            with self.assertRaises(FileNotFoundError):cli.fit_guard(args,c)
            cell_fixture(c,100000,'parameter_control')
            cli.fit_guard(args,c)

    def test_positive_but_declining_300k_cannot_escalate_to_million(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(td);self.cohort(c,100000,gain=.05);cli.screen(c,100000)
            self.cohort(c,300000,gain=.04)
            self.assertFalse(cli.screen(c,300000)['allow_next_budget'])
            args=SimpleNamespace(budget=1000000,seed=17,cell='r2_full',matched_exposure=False)
            with self.assertRaisesRegex(ValueError,'escalation criterion failed'):
                cli.fit_guard(args,c)

    def test_300k_without_same_cell_mode_improvement_stops_even_when_negative(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(td);self.cohort(c,100000,gain=-.05);cli.screen(c,100000)
            self.cohort(c,300000,gain=-.05)
            self.assertFalse(cli.screen(c,300000)['allow_next_budget'])
            args=SimpleNamespace(budget=1000000,seed=17,cell='r2_full',matched_exposure=False)
            with self.assertRaisesRegex(ValueError,'escalation criterion failed'):
                cli.fit_guard(args,c)

    def test_improving_negative_same_cell_mode_can_advance_but_final_test_stays_sealed(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(td);self.cohort(c,100000,gain=-.10);cli.screen(c,100000)
            # Full R2 improves by .02 although all conditional paths still STOP
            # and the absolute best dev NLL remains the native value 1.0.
            self.cohort(c,300000,gain=-.08)
            screen=cli.screen(c,300000)
            self.assertTrue(screen['allow_next_budget'])
            self.assertGreater(min(x['nll'] for x in screen['candidates'] if x['mode']=='forced'),1.)
            args=SimpleNamespace(budget=1000000,seed=17,cell='r2_full',matched_exposure=False)
            cli.fit_guard(args,c)
            args=SimpleNamespace(budget=300000,seed=29,cell='r2_full',matched_exposure=False)
            with self.assertRaisesRegex(ValueError,'conditional screen'):
                cli.fit_guard(args,c)
            with self.assertRaisesRegex(ValueError,'conditional screen'):
                cli.finalize(c,300000,{'commit':'a'*40,'sources':{}},dict(freeze_sha256='f'*64))
            self.assertFalse((Path(td)/'final_selection.json').exists())

    def test_confirmation_cell_frozen_and_finalization_requires_parameter_control_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            c=self.config(td);self.cohort(c,100000,gain=.05,gate_gain=.02);cli.screen(c,100000)
            self.cohort(c,300000,gain=.05,gate_gain=.02)
            args=SimpleNamespace(budget=100000,seed=29,cell='r2_full',matched_exposure=False)
            with self.assertRaisesRegex(ValueError,'confirmation seeds limited'):
                cli.fit_guard(args,c)
            for seed in (29,43):cell_fixture(c,100000,'r2_reduced',seed)
            cell_fixture(c,100000,'parameter_control',29)
            code={'commit':'a'*40,'sources':{}};index=dict(freeze_sha256='f'*64)
            with self.assertRaises(FileNotFoundError):cli.finalize(c,100000,code,index)
            self.assertFalse((Path(td)/'final_selection.json').exists())
            cell_fixture(c,100000,'parameter_control',43)
            lock=cli.finalize(c,100000,code,index)
            self.assertEqual(len(lock['cells']),6)
            self.assertEqual({(x['cell'],x['seed']) for x in lock['cells']},
                {(cell,seed) for cell in ('r2_reduced','parameter_control') for seed in (17,29,43)})
            self.assertEqual(cli.check_lock(Path(td)/'final_selection.json',c,code,index),lock)
            cell_fixture(c,100000,'parameter_control',43,forced=.8)
            with self.assertRaisesRegex(ValueError,'cohort already locked'):
                cli.finalize(c,100000,code,index)


if __name__=='__main__':
    unittest.main()
