"""Whole-document router correctness; tiny random HF, CPU, no downloads."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.document_training import cache_documents, document_forward, document_losses
from mixtral_lab.document_router import (DocumentRouterConfig, train_document_router,
    load_document_router_checkpoint, build_document_cost_table, _cache_binding, _probe,
    _digest, _tensor_hashes)
from mixtral_lab.recovery_router import RecoveryPairRouter
from tests.test_mixtral_adapter import tiny


IDENTITIES={key:'toy-'+key for key in ('model','tokenizer','code')}
IDENTITIES.update(train={'selected_document_ids':['train-a','train-b']},
                  development={'selected_document_ids':['dev-a']})


def rows():
    return [dict(id='train-a',input_ids=torch.tensor([0,1,2,3,4,0]),attention_mask=torch.tensor([0,1,1,1,1,0])),
            dict(id='train-b',input_ids=torch.tensor([5,6,7,8,9,0]),attention_mask=torch.tensor([1,1,0,1,1,0]))]


class Development:
    def __init__(self,nlls=(3.,2.,1.)):
        self.contract={'role':'development','protocol':'full_document_online_nll','document_ids':['dev-a']}
        self.nlls,self.calls=list(nlls),0
    def __call__(self,actor):
        assert not torch.is_grad_enabled()
        assert not actor.training
        nll=self.nlls[min(self.calls,len(self.nlls)-1)]
        self.calls+=1
        return {**self.contract,'status':'ok','nll':nll}


def setup(dtype=torch.float32):
    adapter=MixtralAdapter(tiny(dtype),head_chunk=1)
    return adapter,cache_documents(adapter,rows())


def fit(path,*,dtype=torch.float32,config=None,development=None):
    adapter,caches=setup(dtype)
    actor,summary=train_document_router(adapter,caches,development or Development(),
        config or DocumentRouterConfig(updates=4,positions_per_update=2,eval_interval=2,min_updates=0),
        copy.deepcopy(IDENTITIES),path)
    return adapter,caches,actor,summary


class DocumentRouterTests(unittest.TestCase):
    def test_labels_and_probes_preserve_entire_padded_expert_and_head_shape(self):
        for dtype in (torch.float32,torch.bfloat16):
            with self.subTest(dtype=dtype),tempfile.TemporaryDirectory() as directory:
                adapter,caches=setup(dtype)
                shapes=[]
                original=adapter._forward_r1
                def traced(h,policy):
                    shapes.append(tuple(h.shape))
                    return original(h,policy)
                with patch.object(adapter,'_forward_r1',side_effect=traced),patch.object(adapter,'score_pairs',side_effect=AssertionError('singleton path forbidden')):
                    actor,summary=train_document_router(adapter,caches,Development(),
                        DocumentRouterConfig(updates=2,positions_per_update=2,eval_interval=1),
                        copy.deepcopy(IDENTITIES),Path(directory)/'fit')
                self.assertTrue(shapes)
                self.assertEqual(set(shapes),{(1,6,16)})
                self.assertEqual(summary['work_totals']['table_completed_document_forwards'],58)
                self.assertEqual(summary['work_totals']['table_head_rows'],58*6)
                self.assertEqual(summary['work_totals']['probe_completed_document_forwards'],6)
                self.assertEqual(summary['initial_train_metrics']['tokens'],5)
                self.assertEqual(summary['initial_train_metrics']['native_agreement'],1.)
                self.assertFalse(summary['surrogate_is_deployment_cost'])
                self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_rotating_positions_cover_all_and_population_is_cumulative(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,_,summary=fit(Path(directory)/'fit')
            self.assertEqual(summary['training_coverage']['unique_positions_visited'],5)
            self.assertEqual(summary['training_coverage']['optimizer_position_visits'],7)
            self.assertEqual(summary['initial_train_metrics']['populations']['optimizer_seen_positions']['tokens'],0)
            latest=summary['latest_train_metrics']['populations']
            self.assertEqual(latest['optimizer_seen_positions']['tokens'],5)
            self.assertEqual(latest['optimizer_unseen_positions'],{'tokens':0,'native_nll':None,'nll':None})

    def test_best_includes_zero_and_strict_checkpoint_roundtrip(self):
        for dtype in (torch.float32,torch.bfloat16):
            with self.subTest(dtype=dtype),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'fit'
                adapter,caches,actor,summary=fit(root,dtype=dtype,development=Development((1.,2.,3.)))
                self.assertEqual(summary['best_update'],0)
                loaded,metadata=load_document_router_checkpoint(root/'checkpoint_best.pt',IDENTITIES)
                self.assertEqual(metadata['update'],0)
                self.assertEqual(summary['initial_development_metrics'],metadata['development_metrics'])
                self.assertFalse(metadata['resume_supported'])
                for key,value in actor.state_dict().items():
                    torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)
                for cache in caches:
                    logits,_=document_forward(adapter,cache,loaded,kind='router')
                    native,_=document_forward(adapter,cache,None,kind='router')
                    torch.testing.assert_close(logits,native,rtol=0,atol=0)
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                self.assertTrue((root/'checkpoint_initial.pt').is_file())
                self.assertTrue((root/'checkpoint_latest.pt').is_file())

    def test_surrogate_probe_is_not_used_as_actual_nll(self):
        adapter,caches=setup()
        cfg=DocumentRouterConfig(updates=0)
        work={p+'_'+s:0 for p in ('table','probe') for s in ('attempted_document_forwards','completed_document_forwards','head_rows')}
        table,_=build_document_cost_table(adapter,caches,cfg,work)
        table['costs'].fill_(9.)
        actor=RecoveryPairRouter(adapter.feature_dim)
        result=_probe(adapter,caches,actor,table,torch.zeros(5,dtype=torch.long),work)
        self.assertEqual(result['surrogate_nll'],9.)
        self.assertNotEqual(result['nll'],9.)
        self.assertEqual(result['gain_nats'],0.)
        self.assertAlmostEqual(result['surrogate_minus_actual_nll'],9.-result['nll'])

    def test_probe_action_diagnostics_use_executed_full_shape_not_masked_batch(self):
        adapter,caches=setup()
        work={p+'_'+s:0 for p in ('table','probe') for s in ('attempted_document_forwards','completed_document_forwards','head_rows')}
        table,_=build_document_cost_table(adapter,caches,DocumentRouterConfig(updates=0),work)
        table['costs'].fill_(1.)
        table['costs'][:,27]=9.
        actor=RecoveryPairRouter(adapter.feature_dim)
        def shape_sensitive(features,pairs):
            logits=torch.zeros(len(features),28,device=features.device)
            logits[:,27 if len(features)==6 else 0]=10.
            return logits.log_softmax(-1)
        with patch.object(actor,'log_probs',side_effect=shape_sensitive):
            result=_probe(adapter,caches,actor,table,torch.zeros(5,dtype=torch.long),work)
        self.assertEqual(result['surrogate_nll'],9.)
        self.assertEqual(result['greedy_action_counts'][27],5)
        self.assertEqual(sum(result['greedy_action_counts']),5)

    def test_train_labels_and_document_roles_are_validated(self):
        adapter,caches=setup()
        with tempfile.TemporaryDirectory() as directory:
            for mutation in ('target','mask','order','duplicate'):
                with self.subTest(mutation=mutation):
                    bad=copy.deepcopy(caches)
                    if mutation=='target': bad[0].targets[1]+=1
                    if mutation=='mask': bad[0].valid_mask[0]=True
                    if mutation=='order': bad.reverse()
                    if mutation=='duplicate': bad[1].document_id=bad[0].document_id
                    with self.assertRaises(ValueError):
                        train_document_router(adapter,bad,Development(),DocumentRouterConfig(),IDENTITIES,Path(directory)/mutation)
            dev=Development();dev.contract['role']='selection'
            with self.assertRaises(ValueError):
                train_document_router(adapter,caches,dev,DocumentRouterConfig(),IDENTITIES,Path(directory)/'role')

    def test_budgets_rejected_before_table_execution(self):
        adapter,caches=setup()
        with tempfile.TemporaryDirectory() as directory,patch.object(adapter,'_forward_r1',side_effect=AssertionError('must preflight')):
            for config in (DocumentRouterConfig(max_table_calls=57),DocumentRouterConfig(max_train_tokens=4,positions_per_update=1),
                           DocumentRouterConfig(max_cache_bytes=1),DocumentRouterConfig(max_logit_elements=1)):
                with self.subTest(config=config),self.assertRaises(ValueError):
                    train_document_router(adapter,caches,Development(),config,IDENTITIES,Path(directory)/'fit')

    def test_failed_table_keeps_attempted_and_completed_counts(self):
        adapter,caches=setup()
        original=adapter._forward_r1
        calls=0
        def fail(h,policy):
            nonlocal calls
            calls+=1
            if calls==3: raise RuntimeError('synthetic hardware failure')
            return original(h,policy)
        with tempfile.TemporaryDirectory() as directory,patch.object(adapter,'_forward_r1',side_effect=fail):
            out=Path(directory)/'fit'
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                train_document_router(adapter,caches,Development(),DocumentRouterConfig(),IDENTITIES,out)
            report=json.loads((out/'summary.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['work_totals']['table_attempted_document_forwards'],3)
            self.assertEqual(report['work_totals']['table_completed_document_forwards'],2)
            self.assertFalse((out/'checkpoint_best.pt').exists())

    def test_adaptive_checkpoint_and_patience_wait_for_coverage(self):
        cfg=DocumentRouterConfig(updates=6,positions_per_update=1,eval_interval=1,min_updates=0,development_patience=1,
            adaptive={'enabled':True,'eval_interval':1,'patience':1,'min_delta':100.,'min_updates':0,'stop_patience':2,
                      'min_lr':1e-5,'max_lr':.02})
        with tempfile.TemporaryDirectory() as directory:
            _,_,_,summary=fit(Path(directory)/'fit',config=cfg,development=Development((1.,1.)))
            self.assertEqual(summary['successful_updates'],5)
            self.assertEqual(summary['stop_reason'],'development_patience')
            self.assertEqual(summary['training_coverage']['coverage_fraction'],1.)
            load_document_router_checkpoint(Path(directory)/'fit'/'checkpoint_latest.pt',IDENTITIES)

    def test_checksum_source_identity_and_state_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'fit'
            fit(out,config=DocumentRouterConfig(updates=0))
            path=out/'checkpoint_best.pt'
            original=torch.load(path,weights_only=True)
            for kind in ('checksum','tensor','identity','source','development'):
                bad=copy.deepcopy(original)
                if kind=='checksum': bad['update']=1
                elif kind=='tensor': bad['actor_state_dict']['bias'][0]=float('nan')
                elif kind=='identity': bad['binding']['identities']['model']='wrong'
                elif kind=='source': bad['binding']['sources']['adapter.py']='bad'
                else: bad['development_metrics']['status']='failed'
                if kind not in ('checksum','tensor'):
                    binding=bad['binding'];binding['sha256']=_digest({k:v for k,v in binding.items() if k!='sha256'})
                    bad['payload_sha256']=_digest({k:v for k,v in bad.items() if k not in ('payload_sha256','actor_state_dict')})
                p=out/f'{kind}.pt';torch.save(bad,p)
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    load_document_router_checkpoint(p,IDENTITIES)

    def test_invalid_config_and_rng_independence(self):
        for cfg in (DocumentRouterConfig(epsilon=1.),DocumentRouterConfig(temperature=0.),
                    DocumentRouterConfig(updates=True),DocumentRouterConfig(max_table_calls=False),
                    DocumentRouterConfig(adaptive={'enabled':True,'eval_interval':3})):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):cfg.validate()
        adapter,caches=setup()
        state=torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            train_document_router(adapter,caches,Development(),DocumentRouterConfig(updates=2,eval_interval=1),IDENTITIES,Path(directory)/'fit')
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))


if __name__=='__main__': unittest.main()
