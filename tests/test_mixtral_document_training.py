"""CPU tiny-HF full-document shape, masks, selection and checkpoint contracts."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.document_training import (DocumentAdapterConfig,cache_documents,document_forward,
    document_losses,verify_document_replay,evaluate_documents,online_logits,
    train_document_adapter,load_document_adapter_checkpoint,_loss_components)
from mixtral_lab.second_round import SecondRoundResidual
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_bridge import rows


def identities():
    return dict(model={'id':'tiny'},tokenizer={'id':'tiny'},code={'sha':'unit'},
        train={'selected_document_ids':['train-a','train-b']},development={'selected_document_ids':['dev']})


def evaluator(scores=None):
    values=iter(scores) if scores else None
    contract=dict(role='development',document_ids=['dev'],protocol='full_document_online_nll')
    def call(module):
        assert not torch.is_grad_enabled() and not module.training
        return dict(contract,status='ok',nll=next(values) if values else 1.)
    call.contract=contract
    return call


def cfg(**kw):return replace(DocumentAdapterConfig(rank=2,updates=4,positions_per_document=1,eval_interval=1,min_updates=99,learning_rate=.02),**kw)


class DocumentTrainingTests(unittest.TestCase):
    def test_full_shapes_native_zero_nonzero_and_rng(self):
        for dtype in (torch.float32,torch.bfloat16):
            adapter=MixtralAdapter(tiny(dtype));caches=cache_documents(adapter,rows())
            rng=torch.random.get_rng_state().clone();module=SecondRoundResidual(16,cfg(),adapter.device)
            with torch.no_grad():module.up.fill_(.02)
            result=verify_document_replay(adapter,caches,module)
            self.assertTrue(result['passed']);self.assertEqual(rng.tolist(),torch.random.get_rng_state().tolist())
            self.assertIn('probe_cached_vs_full.logits',result['documents'][0]['comparisons'])
            self.assertIn('cached_vs_full.h0',result['documents'][0]['comparisons'])
            calls=[];handle=adapter.model.lm_head.register_forward_pre_hook(lambda _,args:calls.append(tuple(args[0].shape)))
            try:
                parts=_loss_components(adapter,module,caches[0],cfg())
                parts['loss'][1:2].mean().backward()
            finally:handle.remove()
            self.assertEqual(calls,[(1,6,16),(1,6,16)])
            self.assertGreater(float(module.up.grad.abs().sum()),0.)
            self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))

    def test_masks_next_token_and_no_label_compaction(self):
        adapter=MixtralAdapter(tiny());caches=cache_documents(adapter,rows())
        self.assertEqual(torch.nonzero(caches[0].valid_mask).flatten().tolist(),[1,2,3])
        self.assertEqual(torch.nonzero(caches[1].valid_mask).flatten().tolist(),[0,3])
        self.assertEqual(caches[0].targets.tolist(),[1,2,3,4,0,0])
        logits,_=document_forward(adapter,caches[0])
        self.assertEqual(document_losses(logits,caches[0]).shape,(6,))
        report=evaluate_documents(adapter,rows(),role='development')
        self.assertEqual(report['tokens'],5)
        self.assertEqual(report['native']['nll'],report['candidate']['nll'])

    def test_allocation_and_duplicate_validation_pre_capture(self):
        adapter=MixtralAdapter(tiny())
        for changes in ({'max_cache_bytes':1},{'max_logit_elements':1}):
            with patch.object(adapter,'capture',side_effect=AssertionError('must not capture')):
                with self.assertRaises(ValueError):cache_documents(adapter,rows(),**changes)
        with self.assertRaises(ValueError):cache_documents(adapter,[rows()[0],rows()[0]])
        c=cache_documents(adapter,rows());c[0].targets[1]=31
        with self.assertRaises(ValueError):verify_document_replay(adapter,c)

    def test_online_helper_one_native_tail_and_head(self):
        adapter=MixtralAdapter(tiny());calls=[]
        h=adapter.model.lm_head.register_forward_pre_hook(lambda _,args:calls.append(tuple(args[0].shape)))
        try:online_logits(adapter,rows()[0])
        finally:h.remove()
        self.assertEqual(calls,[(1,6,16)])

    def test_development_selection_rotating_masks_checkpoint_roundtrip(self):
        adapter=MixtralAdapter(tiny());caches=cache_documents(adapter,rows())
        before={k:v.clone() for k,v in adapter.model.state_dict().items()}
        rng=torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run'
            module,summary=train_document_adapter(adapter,caches,evaluator([3.,2.,1.,2.,3.]),cfg(),identities(),path)
            self.assertEqual(summary['best_update'],2)
            self.assertEqual(summary['unique_contexts_visited'],4)
            self.assertEqual(summary['context_visits'],4)
            self.assertEqual(summary['latest_train_probe']['unselected']['tokens'],1)
            loaded,meta=load_document_adapter_checkpoint(path/'checkpoint_best.pt',identities(),expected_config=cfg())
            self.assertEqual(meta['update'],2)
            self.assertEqual(meta['checkpoint_sha256'],hashlib.sha256((path/'checkpoint_best.pt').read_bytes()).hexdigest())
            self.assertTrue(torch.equal(loaded.up,module.up))
            self.assertEqual(json.loads((path/'summary.json').read_text()),summary)
            wrong=identities();wrong['model']['id']='other'
            with self.assertRaises(ValueError):load_document_adapter_checkpoint(path/'checkpoint_best.pt',wrong)
            with self.assertRaises(FileExistsError):train_document_adapter(adapter,caches,evaluator(),cfg(),identities(),path)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in adapter.model.state_dict().items()))

    def test_update_zero_and_empty_never_supervised_population(self):
        adapter=MixtralAdapter(tiny());caches=cache_documents(adapter,rows())
        with tempfile.TemporaryDirectory() as tmp:
            module,s=train_document_adapter(adapter,caches,evaluator(),cfg(updates=2,positions_per_document=6),identities(),Path(tmp)/'run')
            self.assertEqual(s['best_update'],0)
            self.assertTrue(s['coverage_complete'])
            self.assertEqual(s['latest_train_probe']['unselected'],dict(tokens=0,nll=None))
            self.assertFalse(bool(module.up.any()))

    def test_callback_mutation_and_direct_cache_bound_fail_closed(self):
        adapter=MixtralAdapter(tiny());caches=cache_documents(adapter,rows())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):train_document_adapter(adapter,caches,evaluator(),cfg(max_cache_bytes=1),identities(),Path(tmp)/'small')
            self.assertFalse((Path(tmp)/'small').exists())
            def bad(module):
                module.requires_grad_(False)
                return dict(role='development',document_ids=['dev'],protocol='full_document_online_nll',status='ok',nll=1.)
            bad.contract=evaluator().contract
            with self.assertRaises(ValueError):train_document_adapter(adapter,caches,bad,cfg(),identities(),Path(tmp)/'bad')
            self.assertTrue((Path(tmp)/'bad'/'failure.json').exists())
            self.assertFalse((Path(tmp)/'bad'/'checkpoint_best.pt').exists())

    def test_work_counts_survive_failed_candidate(self):
        adapter=MixtralAdapter(tiny());work={}
        def bad(h):raise RuntimeError('injected candidate failure')
        with self.assertRaises(RuntimeError):evaluate_documents(adapter,rows(),bad,work=work)
        self.assertEqual(work['prefix_capture_attempted'],1)
        self.assertEqual(work['prefix_capture_completed'],1)
        self.assertEqual(work['candidate_tail_with_head_attempted'],1)
        self.assertNotIn('candidate_tail_with_head_completed',work)

    def test_checkpoint_corruption_rejected(self):
        adapter=MixtralAdapter(tiny());caches=cache_documents(adapter,rows())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'r'
            train_document_adapter(adapter,caches,evaluator(),cfg(updates=1),identities(),path)
            cp=path/'checkpoint_best.pt';data=torch.load(cp,weights_only=True)
            data['state']['up'][0,0]+=1.;torch.save(data,cp)
            with self.assertRaises(ValueError):load_document_adapter_checkpoint(cp,identities())

    @unittest.skipUnless(torch.cuda.device_count()>=2,'requires two CUDA devices')
    def test_sharded_full_document_parity_gradient_and_masks(self):
        from accelerate import dispatch_model
        model=tiny(torch.bfloat16)
        model=dispatch_model(model,device_map={'model.embed_tokens':0,'model.layers.0':0,
            'model.layers.1':0,'model.norm':1,'lm_head':1},force_hooks=True)
        adapter=MixtralAdapter(model);caches=cache_documents(adapter,rows())
        module=SecondRoundResidual(16,cfg(),adapter.device)
        with torch.no_grad():module.up.fill_(.01)
        report=verify_document_replay(adapter,caches,module)
        self.assertTrue(report['passed'])
        part=_loss_components(adapter,module,caches[0],cfg())
        part['loss'][caches[0].valid_mask.to(part['loss'].device)].mean().backward()
        self.assertGreater(float(module.up.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(evaluate_documents(adapter,rows(),module)['tokens'],5)

    def test_config_rejects_legacy_chunking_and_invalid_values(self):
        with self.assertRaises(TypeError):DocumentAdapterConfig(batch_contexts=4)
        for change in (dict(positions_per_document=0),dict(parity_atol=float('nan')),dict(updates=True),dict(native_kl_beta=-1)):
            with self.assertRaises(ValueError):cfg(**change).validate()

if __name__=='__main__':unittest.main()
