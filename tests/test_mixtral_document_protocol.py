"""CPU tiny-model protocol integration, preflight and immutable-evidence tests."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments import mixtral_document as runner
from mixtral_lab import document_protocol as protocol
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.protocol import canonical_hash
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_protocol import config as base_config, Tokenizer


def document_config(root, kind='adapter'):
    c = base_config(root)
    for key in ('training', 'fit', 'oracle', 'second_round', 'max_train_contexts', 'context_sampling'):
        c.pop(key, None)
    c['schema'] = protocol.SCHEMA
    path = root/'development.jsonl'
    path.write_text(json.dumps({'id': 'development', 'text': 'document 3 independent checkpoint development'})+'\n')
    c['data_files']['development'] = str(path)
    c['sequences']['development'] = 1
    training = dict(updates=2, eval_interval=1, min_updates=0, development_patience=2,
                    learning_rate=.01, seed=17, parity_atol=0., parity_rtol=0.)
    if kind == 'adapter':
        training.update(rank=2, positions_per_document=2)
    else:
        training.update(positions_per_update=2, max_table_calls=100, max_train_tokens=100)
    c['document'] = dict(kind=kind, training=training, development_min_gain=.001, selection_min_gain=.001)
    return c


def paired(rows, changed, role=None, score=1.8):
    records=[]
    for r in rows:
        tokens=int((r['attention_mask'][:-1]*r['attention_mask'][1:]).sum())
        nll=score if changed else 2.
        records.append(dict(document_id=r['id'],tokens=tokens,native_nll=2.,controlled_nll=2.,candidate_nll=nll,
            candidate_minus_native_nll=nll-2.,candidate_minus_controlled_nll=nll-2.))
    return {'per_document':records}


class DocumentProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.raw=document_config(self.root)

    def runtime(self, score=1.8):
        stack=self.enterContext(ExitStack())
        model=tiny();adapter=MixtralAdapter(model)
        self.adapter=adapter
        info={'model':{'content_sha256':'a'*64},'tokenizer':{'content_sha256':'b'*64},'cuda_devices':[]}
        inputs={'metadata':model.config,'tokenizer':Tokenizer(),'identity':{'local':'stable'}}
        stack.enter_context(patch.object(runner,'load_local_inputs',return_value=inputs))
        self.heavy=stack.enter_context(patch.object(runner,'load_local_model',return_value=(model,inputs['tokenizer'],adapter,info)))
        stack.enter_context(patch.object(runner,'normalized_artifact_stats',return_value={'stable':True}))
        stack.enter_context(patch.object(runner,'document_code_identity',return_value={'commit':'stable','sources':{'x':'a'*64}}))
        stack.enter_context(patch.object(runner,'runtime_versions',return_value={'runtime':'cpu-test'}))
        stack.enter_context(patch.object(runner.torch.cuda,'is_available',return_value=True))
        stack.enter_context(patch.object(runner.torch.cuda,'is_current_stream_capturing',return_value=False))
        self.timing=stack.enter_context(patch.object(runner,'measure_calls',return_value={'variants':{}}))
        from mixtral_lab import document_training
        self.roles=[]
        def evaluator(a,rows,candidate=None,**kwargs):
            self.roles.extend(r['id'] for r in rows)
            state=getattr(candidate,'up',getattr(candidate,'weight',torch.zeros(1)))
            return paired(rows,bool(torch.count_nonzero(state)),score=score)
        stack.enter_context(patch.object(document_training,'evaluate_documents',side_effect=evaluator))

    def run_stage(self, stage, config=None, tune=None, evaluation=None):
        c=copy.deepcopy(self.raw if config is None else config)
        output=self.root/f'{stage}-{len(list(self.root.glob("cfg-*.json")))}'
        c['output_dir']=str(output)
        path=self.root/f'cfg-{len(list(self.root.glob("cfg-*.json")))}.json';path.write_text(json.dumps(c))
        args=['--config',str(path),'--stage',stage]
        if tune:
            args+=['--tune-report',str(Path(tune['config']['output_dir'])/'report.json'),'--checkpoint',tune['checkpoint']['path']]
        if evaluation:
            args+=['--evaluation-report',str(Path(evaluation['config']['output_dir'])/'report.json')]
        return runner.main(args)

    def test_config_rejects_legacy_and_unknown_roles(self):
        for key in ('recovery','fit','training','max_train_contexts','context_sampling'):
            c=copy.deepcopy(self.raw);c[key]={}
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.validate_document_config(c)
        c=copy.deepcopy(self.raw);c['data_files']['newtest']='x'
        with self.assertRaises(ValueError):protocol.validate_document_config(c)

    def test_config_does_not_mutate_input_and_normalizes_training(self):
        before=copy.deepcopy(self.raw);c=protocol.validate_document_config(self.raw)
        self.assertEqual(before,self.raw);self.assertIn('max_cache_bytes',c['document']['training'])
        self.assertNotIn('max_train_contexts',c)

    def test_seed_mismatch_rejected(self):
        c=copy.deepcopy(self.raw);c['document']['training']['seed']=88
        with self.assertRaisesRegex(ValueError,'seeds'):protocol.validate_document_config(c)

    def test_duplicate_development_source_preflight_before_model(self):
        self.runtime();c=copy.deepcopy(self.raw);c['data_files']['development']=c['data_files']['train']
        with self.assertRaises(ValueError):self.run_stage('tune',c)
        self.heavy.assert_not_called()

    def test_token_leakage_before_model(self):
        self.runtime()
        p=Path(self.raw['data_files']['development'])
        train=json.loads(Path(self.raw['data_files']['train']).read_text().splitlines()[0])
        p.write_text(json.dumps({'id':'different-id','text':train['text']})+'\n')
        with self.assertRaises(ValueError):self.run_stage('tune')
        self.heavy.assert_not_called()

    def test_logit_quota_before_model(self):
        self.runtime();c=copy.deepcopy(self.raw);c['document']['training']['max_logit_elements']=1
        with self.assertRaises(ValueError):self.run_stage('tune',c)
        self.heavy.assert_not_called()

    def test_stage_arguments(self):
        for args in (['--config','x','--stage','evaluate'],['--config','x','--stage','profile','--tune-report','t','--checkpoint','c'],
                     ['--config','x','--stage','tune','--checkpoint','c']):
            with self.assertRaises(ValueError):runner.validate_stage_arguments(runner.parser().parse_args(args))

    def test_adapter_real_cache_checkpoint_pipeline(self):
        self.runtime();tune=self.run_stage('tune')
        self.assertEqual(tune['status'],'complete');self.assertTrue(tune['development_gate']['passed'])
        self.assertEqual(set(self.roles),{'development'})
        self.assertTrue(tune['replay_initial']['passed']);self.assertTrue(tune['replay_frozen']['passed'])
        evaluation=self.run_stage('evaluate',tune=tune)
        self.assertTrue(evaluation['evaluation']['quality_gate']['passed'])
        profile=self.run_stage('profile',tune=tune,evaluation=evaluation)
        self.assertEqual(profile['status'],'complete');self.timing.assert_called_once()

    def test_update_zero_fallback_blocks_outer_before_model(self):
        self.runtime(score=2.1);tune=self.run_stage('tune')
        self.assertEqual(tune['tune_summary']['best_update'],0)
        self.assertFalse(tune['development_gate']['passed']);n=self.heavy.call_count
        with self.assertRaises(ValueError):self.run_stage('evaluate',tune=tune)
        self.assertEqual(self.heavy.call_count,n)

    def test_router_real_cache_checkpoint_pipeline(self):
        self.raw=document_config(self.root,'router');self.runtime()
        tune=self.run_stage('tune');self.assertEqual(tune['status'],'complete')
        self.assertTrue(tune['development_gate']['passed'])
        self.run_stage('evaluate',tune=tune)

    def test_report_boolean_and_candidate_binding_tampering(self):
        self.runtime();tune=self.run_stage('tune')
        for alter in ('count','missing','binding','candidate','protocol','absolute','zero_probe'):
            report=copy.deepcopy(tune)
            replay=report['replay_frozen']
            if alter=='count':
                metric=next(iter(replay['documents'][0]['comparisons'].values()))
                metric['mismatched_elements']=1;metric['max_excess']=1.
            elif alter=='missing':del replay['documents'][0]['comparisons']['cached_vs_full.h0']
            elif alter=='binding':replay['binding']['checkpoint']['update']+=1
            elif alter=='candidate':replay['candidate_checked']=False
            elif alter=='absolute':next(iter(replay['documents'][0]['comparisons'].values()))['max_abs']=1.
            elif alter=='zero_probe':replay['documents'][0]['nonzero_probe']['effective_changed_elements']=0
            else:replay['protocol']='old-singleton'
            report=protocol.seal_report(report)
            p=self.root/f'tamper-{alter}.json';p.write_text(json.dumps(report))
            with self.subTest(alter=alter),self.assertRaises(ValueError):protocol.read_report(p,'tune')

    def test_changed_checkpoint_bytes_preflight(self):
        self.runtime();tune=self.run_stage('tune');n=self.heavy.call_count
        p=Path(tune['checkpoint']['path']);p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaises(ValueError):self.run_stage('evaluate',tune=tune)
        self.assertEqual(self.heavy.call_count,n)

    def test_changed_scientific_config_preflight(self):
        self.runtime();tune=self.run_stage('tune');n=self.heavy.call_count
        c=copy.deepcopy(self.raw);c['document']['training']['native_kl_beta']=.2
        with self.assertRaises(ValueError):self.run_stage('evaluate',c,tune=tune)
        self.assertEqual(self.heavy.call_count,n)

    def test_actual_full_document_evaluation_retains_work_counts(self):
        from mixtral_lab.document_training import DocumentAdapterConfig
        from mixtral_lab.second_round import SecondRoundResidual
        config=protocol.validate_document_config(self.raw)
        data,_=protocol.prepare_data(config,Tokenizer())
        adapter=MixtralAdapter(tiny())
        candidate=SecondRoundResidual(adapter.hidden_size,DocumentAdapterConfig(rank=2),adapter.device)
        work={}
        result=runner.evaluate_documents(adapter,data['development'],candidate,config,role='development',work=work)
        self.assertEqual(result['native'],result['candidate'])
        counters=work['development_execution']
        self.assertEqual(counters['native_full_model_attempted'],1)
        self.assertEqual(counters['candidate_tail_with_head_completed'],1)

    def test_timed_candidates_use_explicit_logit_bound(self):
        config=protocol.validate_document_config(self.raw)
        config['document']['training']['max_logit_elements']=1234
        adapter=MixtralAdapter(tiny())
        with patch('mixtral_lab.document_training.online_logits',return_value=torch.ones(1,2,3)) as forward:
            calls=runner.profile_functions(adapter,None,config)
            calls['controlled_original_r1']({})
            calls['frozen_candidate_adapter']({})
        self.assertEqual(forward.call_count,2)
        self.assertTrue(all(c.kwargs['max_logit_elements']==1234 for c in forward.call_args_list))

    def test_failed_preflight_report_retained(self):
        self.runtime();c=copy.deepcopy(self.raw);c['schema']='wrong'
        with self.assertRaises(ValueError):self.run_stage('tune',c)
        reports=list(self.root.glob('tune-*/report.json'));self.assertEqual(len(reports),1)
        failure=json.loads(reports[0].read_text());self.assertEqual(failure['status'],'failed')
        self.assertFalse(failure['model_load_started'])


if __name__=='__main__':unittest.main()
