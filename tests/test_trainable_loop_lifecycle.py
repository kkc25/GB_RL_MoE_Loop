"""CPU lifecycle and evidence guards; no downloaded weights or CUDA execution."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from experiments import mixtral_trainable_loop as cli
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.protocol import sha256_file
from mixtral_lab.fit_protocol import seal_report
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_bridge import rows


def config():
    c=json.loads(Path('local_configs/document_adapter_pilot_seed43.json').read_text())
    c.pop('document');c.update(schema=cli.SCHEMA,seed=17,total_wall_seconds=7200,selection_min_gain=.001)
    c['loop']=asdict(replace(cli.core.LoopConfig(),rank=2,epochs=1,positions_per_document=2,
        max_updates=20,gate_updates=2,gate_batch=4,eligibility_fraction=0,eligibility_top_mean=0))
    return c


class LifecycleTests(unittest.TestCase):
    def test_config_and_final_test_rejected(self):
        c=config();self.assertEqual(cli.validate_config(c)['schema'],cli.SCHEMA)
        for key,value in [('schema','old'),('total_wall_seconds',7201),('selection_min_gain',float('nan'))]:
            bad=deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):cli.validate_config(bad)
        c['data_files']['newtest']='x'
        with self.assertRaises(ValueError):cli.validate_config(c)

    def test_stop_wins_ties_and_no_continuations(self):
        def fake(*a,mode,**kw):return dict(nll=1.,continued_supervised_tokens=0 if mode=='stop' else 10)
        with patch.object(cli.core,'evaluate',side_effect=fake):
            choice,candidates=cli.choose_threshold(None,[],None,None,cli.core.LoopConfig(),{})
        self.assertEqual(choice['mode'],'stop');self.assertEqual(len(candidates),6)

    def test_quality_pairing_and_negative_control(self):
        def result(nll):return dict(continued_tokens=3,continued_supervised_tokens=3,per_document=[dict(document_id=str(i),tokens=4,candidate_nll=nll) for i in range(3)])
        r=dict(gate=result(1.),stop=result(1.1),matched_random=result(1.02),parameter_control=result(.9))
        q=cli.quality(r,.001,17);self.assertTrue(q['passed']);self.assertLess(q['comparisons']['parameter_control']['gain_nats'],0)
        r['matched_random']=result(.99);self.assertFalse(cli.quality(r,.001,17)['passed'])
        r['stop']['per_document'][0]['document_id']='changed'
        with self.assertRaises(ValueError):cli.quality(r,.001,17)

    def test_fresh_failure_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            c=config();c['output_dir']=str(Path(t)/'run');c['schema']='invalid'
            p=Path(t)/'config.json';p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):cli.main(['--config',str(p),'--stage','fit'])
            report=json.loads((Path(c['output_dir'])/'report.json').read_text())
            self.assertEqual(report['status'],'failed');self.assertEqual(report['failure_phase'],'preflight')
            with self.assertRaises(FileExistsError):cli.main(['--config',str(p),'--stage','fit'])

    def test_actual_cpu_fit_stage_roundtrip_and_tamper(self):
        a=MixtralAdapter(tiny());train=rows();dev=deepcopy(rows())
        for i,r in enumerate(dev):r['id']='dev-'+str(i)
        identity=dict(model={'id':'tiny'},tokenizer={'id':'tiny'},code={'sha':'unit'},
            train={'selected_document_ids':[r['id'] for r in train]},development={'selected_document_ids':[r['id'] for r in dev]})
        with tempfile.TemporaryDirectory() as t:
            report={};c=config()
            # Actual fitting and validation remain real. Force only the *eligibility*
            # decision for lifecycle coverage; this is not a quality result.
            original=cli.core.fit_branch
            def force_eligible(*args,**kwargs):
                branch,summary=original(*args,**kwargs);summary['selected_train_probe']['gate_eligible']=True
                return branch,summary
            with patch.object(cli.core,'fit_branch',side_effect=force_eligible):
                cli.fit_stage(a,{'train':train,'development':dev},c,identity,Path(t),{},report)
            self.assertEqual(report['r2_fit']['epochs_completed'],1)
            self.assertEqual(report['control_fit']['epochs_completed'],1)
            self.assertEqual(len(report['threshold_candidates']),6)
            cli.check_files(report)
            p=Path(report['checkpoints']['r2']['path']);p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaises(ValueError):cli.check_files(report)

    def test_failed_or_changed_reports_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'report.json'
            r=seal_report(dict(schema=cli.REPORT,stage='fit',status='failed',config=cli.validate_config(config())))
            p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):cli.read_report(p,'fit')
            r['status']='complete';p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):cli.read_report(p,'fit')

    def test_three_stage_cpu_runtime_and_frozen_source_guard(self):
        adapter=MixtralAdapter(tiny()); train=rows()
        populations={}
        for role in ('train','development','selection','calibration'):
            populations[role]=deepcopy(train)
            for i,row in enumerate(populations[role]):row['id']=role+'-'+str(i)
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            root=Path(t);c=config();c.update(warmup=1,repeats=1,measurements=1)
            identities={}
            for role,rs in populations.items():
                source=root/(role+'.jsonl');source.write_text(role)
                c['data_files'][role]=str(source)
                identities[role]={'selected_document_ids':[r['id'] for r in rs],'file_sha256':sha256_file(source)}
            path=root/'config.json';path.write_text(json.dumps(c));sha='a'*40
            stack.enter_context(patch.object(cli,'read_documents'))
            stack.enter_context(patch.object(cli,'load_local_inputs',return_value={'tokenizer':None,'identity':{'local':'tiny'},'metadata':SimpleNamespace(hidden_size=16,vocab_size=32)}))
            stack.enter_context(patch.object(cli,'prepare_data',return_value=(populations,identities)))
            stack.enter_context(patch.object(cli,'source_identity',return_value={'commit':sha,'sources':{'tiny':'digest'}}))
            stack.enter_context(patch.object(cli,'runtime_versions',return_value={'runtime':'cpu-tiny-test'}))
            stack.enter_context(patch.object(cli,'normalized_artifact_stats',return_value={}))
            stack.enter_context(patch.object(cli.subprocess,'run'))
            stack.enter_context(patch.object(cli.torch.cuda,'is_available',return_value=True))
            stack.enter_context(patch.object(cli.torch.cuda,'is_current_stream_capturing',return_value=False))
            stack.enter_context(patch.object(cli,'load_local_model',return_value=(adapter.model,None,adapter,{'model':{'tiny':True},'tokenizer':{'tiny':True},'cuda_devices':[]})))
            original=cli.core.fit_branch
            def force_eligible(*args,**kwargs):
                branch,summary=original(*args,**kwargs);summary['selected_train_probe']['gate_eligible']=True
                return branch,summary
            stack.enter_context(patch.object(cli.core,'fit_branch',side_effect=force_eligible))
            # Isolate orchestration from scientific outcome: force eligibility and
            # threshold choice but execute all actual tiny expert paths/checkpoints.
            def choose(a,rs,b,g,cfg,w):
                with torch.no_grad():g.weight.zero_();g.bias.fill_(1.)
                result=cli.core.evaluate(a,rs,b,gate=g,threshold=0.,mode='gate',config=cfg,work=w)
                return {'mode':'gate','threshold':0.,'development':result},[]
            stack.enter_context(patch.object(cli,'choose_threshold',side_effect=choose))
            common=['--config',str(path),'--expected-commit',sha]
            fit=cli.main(common+['--stage','fit','--output-dir',str(root/'fit')])
            self.assertEqual(fit['status'],'complete')
            self.assertGreater(fit['evidence_artifacts']['tokens']['records'],0)
            evidence=[json.loads(line) for line in Path(fit['evidence_artifacts']['tokens']['path']).read_text().splitlines()]
            self.assertTrue(all(len(r['record']['native_ce'])==r['record']['tokens'] for r in evidence))
            q=cli.quality
            def allow_profile(*args):
                result=q(*args);result['passed']=True;return result
            stack.enter_context(patch.object(cli,'quality',side_effect=allow_profile))
            ev=cli.main(common+['--stage','evaluate','--output-dir',str(root/'evaluate'),'--fit-report',str(root/'fit/report.json')])
            prof=cli.main(common+['--stage','profile','--output-dir',str(root/'profile'),'--fit-report',str(root/'fit/report.json'),'--evaluation-report',str(root/'evaluate/report.json')])
            self.assertGreater(ev['cumulative_wall_seconds'],fit['cumulative_wall_seconds'])
            self.assertGreater(prof['cumulative_wall_seconds'],ev['cumulative_wall_seconds'])
            self.assertEqual(len(prof['timing']['variants']),5)
            ev_evidence=Path(ev['evidence_artifacts']['tokens']['path'])
            ev_evidence.write_text(ev_evidence.read_text()+'{}\n')
            with self.assertRaises(ValueError):cli.main(common+['--stage','profile','--output-dir',str(root/'tampered_profile'),'--fit-report',str(root/'fit/report.json'),'--evaluation-report',str(root/'evaluate/report.json')])
            exhausted=deepcopy(fit);exhausted.update(wall_seconds=7200.,prior_wall_seconds=0.,cumulative_wall_seconds=7200.)
            budget_report=root/'exhausted_fit.json';budget_report.write_text(json.dumps(seal_report(exhausted)))
            with self.assertRaises(TimeoutError):cli.main(common+['--stage','evaluate','--output-dir',str(root/'budget'),'--fit-report',str(budget_report)])
            failed=json.loads((root/'budget/report.json').read_text())
            self.assertEqual(failed['failure_kind'],'budget_exhausted')
            self.assertGreater(failed['cumulative_wall_seconds'],7200.)

            with patch.object(cli,'source_identity',return_value={'commit':sha,'sources':{'tiny':'CHANGED'}}):
                with self.assertRaises(ValueError):cli.main(common+['--stage','evaluate','--output-dir',str(root/'changed'),'--fit-report',str(root/'fit/report.json')])

    def test_immutable_source_output_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as t:
            c=config();c['model_path']=str(Path(t)/'model');c['output_dir']=str(Path(t)/'model/run')
            p=Path(t)/'config.json';p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):cli.main(['--config',str(p),'--stage','fit'])
            self.assertFalse(Path(c['model_path']).exists())

    def test_durable_cost_reservations(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'ledger.json';work=cli.DurableWork({},p)
            work['forward_attempts']=1
            self.assertEqual(json.loads(p.read_text()),{'forward_attempts':1})
            work['forward_completed']=1
            self.assertEqual(json.loads(p.read_text())['forward_completed'],1)

    def test_parity_must_use_counts(self):
        with self.assertRaises(ValueError):cli.require_parity({'status':'complete','passed':True,'documents':[]})
        with self.assertRaises(ValueError):cli.require_parity({'status':'complete','passed':True,'documents':[{'comparisons':{'ce':{'elements':2,'mismatched_elements':1}}}]})

    def test_token_evidence_accepts_integer_dictionary_keys(self):
        row={'document_id':'d','tokens':1,'valid_position_indices':[0],
             'native_ce':[1.0],'candidate_ce':[0.5],'continue_mask':[True]}
        with tempfile.TemporaryDirectory() as t:
            report={0:{'per_document':[row]}}
            cli.export_token_evidence(report,Path(t))
            evidence=Path(report['evidence_artifacts']['tokens']['path'])
            self.assertEqual(len(evidence.read_text().splitlines()),1)

if __name__=='__main__':unittest.main()
