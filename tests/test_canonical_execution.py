"""Fixed-row candidate: boundaries, true selection, padding cost and diagnostics."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

import torch
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.canonical_execution import MODE,TILE,tiled,diagnose_path
from mixtral_lab.document_training import cache_documents
from mixtral_lab.loop_scale import ScaleBranch,_ce,preflight,digest
from tests.test_mixtral_adapter import tiny
from tests.test_loop_scale import config
from tests.test_mixtral_bridge import rows


class CanonicalExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.a=MixtralAdapter(tiny())
        self.cfg=config(r2_precision=MODE)
        self.b=ScaleBranch(16,self.cfg)

    def test_all_boundary_sizes_dense_compact_and_offsets(self):
        for n in (1,31,32,33,63,64,65):
            h=torch.randn(1,n,16)
            for selected in (torch.arange(n),torch.arange(0,n,2),torch.tensor([n-1])):
                mask=torch.zeros(n,dtype=torch.bool);mask[selected]=True
                dense=self.b(self.a,h,mask,compact=False)
                compact=self.b(self.a,h,mask,compact=True)
                torch.testing.assert_close(dense,compact,atol=1e-4,rtol=1e-4)
        # Move the very same token through each lane of a fixed tile.
        token=torch.randn(1,1,16);single=self.b(self.a,token)
        for offset in range(TILE):
            h=torch.randn(1,TILE,16);h[:,offset]=token[:,0]
            torch.testing.assert_close(self.b(self.a,h)[:,offset],single[:,0],atol=1e-4,rtol=1e-4)

    def test_permutation_and_repartition_forward_and_readout(self):
        h=torch.randn(1,65,16);target=torch.randint(0,32,(65,));permutation=torch.randperm(65)
        reference=self.b(self.a,h)
        permuted=self.b(self.a,h[:,permutation])[:,torch.argsort(permutation)]
        split=torch.cat([self.b(self.a,h[:,start:end]) for start,end in ((0,1),(1,32),(32,33),(33,65))],dim=1)
        torch.testing.assert_close(permuted,reference,atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(split,reference,atol=1e-4,rtol=1e-4)
        loss=_ce(self.a,h,target,self.b)
        parts=torch.cat([_ce(self.a,h[:,start:end],target[start:end],self.b) for start,end in ((0,1),(1,32),(32,33),(33,65))])
        torch.testing.assert_close(loss,parts,atol=1e-4,rtol=1e-4)

    def test_pre_post_update_gradients_same_scalar_objective(self):
        h=torch.randn(1,65,16);target=torch.randint(0,32,(65,))
        opt=torch.optim.SGD(self.b.parameters(),lr=.01)
        for state in range(2):
            for selected in (torch.tensor([0]),torch.tensor([64]),torch.tensor([0,31,32,64]),torch.arange(1,65,2)):
                mask=torch.zeros(65,dtype=torch.bool);mask[selected]=True
                gradients=[]
                for compact in (False,True):
                    loss=_ce(self.a,h,target,self.b,mask,compact=compact)[selected].mean()
                    gradients.append(torch.autograd.grad(loss,tuple(self.b.parameters())))
                for a,b in zip(*gradients):torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
            opt.zero_grad();_ce(self.a,h,target,self.b).mean().backward();opt.step()
        self.assertGreater(float(self.b.correction.up.abs().sum()),0)

    def test_padded_expert_cost_and_no_unselected_work(self):
        h=torch.randn(1,65,16);mask=torch.zeros(65,dtype=torch.bool);mask[40]=True;work={}
        trace={};out=self.b(self.a,h,mask,work=work,trace=trace)
        self.assertEqual(work['extra_expert_token_calls'],2)
        self.assertEqual(work['canonical_expert_logical_row_attempts'],2)
        self.assertEqual(work['canonical_expert_padded_row_attempts'],64)
        self.assertEqual(work['canonical_norm_padded_row_attempts'],32)
        self.assertEqual(trace['expert_input'].shape,(1,16))
        changed=h.clone();changed[:,~mask]*=100
        other=self.b(self.a,changed,mask)
        self.assertTrue(torch.equal(out[:,mask],other[:,mask]))
        self.assertTrue(torch.equal(out[:,~mask],h[:,~mask]))
        stopped={};self.assertTrue(torch.equal(self.b(self.a,h,torch.zeros_like(mask),work=stopped),h))
        self.assertEqual(stopped,{})

    def test_expert_bucket_padding_formula_and_exact_routes(self):
        h=torch.randn(1,65,16);work={};trace={}
        self.b(self.a,h,work=work,trace=trace)
        counts=torch.bincount(trace['router_pairs'].flatten(),minlength=8)
        expected=sum(((int(n)+31)//32)*32 for n in counts if n)
        self.assertEqual(work['canonical_expert_logical_row_attempts'],130)
        self.assertEqual(work['canonical_expert_padded_row_attempts'],expected)
        for index in (0,31,32,64):
            small={};self.b(self.a,h[:,index:index+1],trace=small)
            self.assertTrue(torch.equal(small['router_pairs'][0],trace['router_pairs'][index]))
            torch.testing.assert_close(small['router_weights'][0],trace['router_weights'][index],atol=0,rtol=0)

    def test_cpu_bf16_all_singleton_offsets(self):
        self.a.model.bfloat16()
        h=torch.randn(1,33,16,dtype=torch.bfloat16)
        full=self.b(self.a,h)
        for index in range(33):
            small=self.b(self.a,h[:,index:index+1])
            torch.testing.assert_close(small,full[:,index:index+1],atol=1e-4,rtol=1e-4)

    def test_precision_scope_restores_after_later_tile_failure(self):
        old=torch.get_float32_matmul_precision();calls=[]
        try:
            torch.set_float32_matmul_precision('medium')
            def fn(x):
                self.assertEqual(torch.get_float32_matmul_precision(),'highest');calls.append(1)
                if len(calls)==2:raise RuntimeError('second tile')
                return x
            with self.assertRaisesRegex(RuntimeError,'second tile'):tiled(torch.randn(33,4),fn)
            self.assertEqual(torch.get_float32_matmul_precision(),'medium')
        finally:torch.set_float32_matmul_precision(old)

    def test_shared_readout_stop_native_and_gate_baseline(self):
        h=torch.randn(1,33,16);target=torch.randint(0,32,(33,));mask=torch.zeros(33,dtype=torch.bool)
        base=_ce(self.a,h,target,config=self.cfg)
        stopped=_ce(self.a,h,target,self.b,mask)
        self.assertTrue(torch.equal(base,stopped))
        work={};_ce(self.a,h,target,config=self.cfg,work=work)
        self.assertEqual(work['canonical_head_logical_row_attempts'],33)
        self.assertEqual(work['canonical_head_padded_row_attempts'],64)
        cfg=replace(self.cfg,max_logit_elements=33*32)
        with self.assertRaisesRegex(ValueError,'allocation bound'):_ce(self.a,h,target,config=cfg)
        cfg=replace(self.cfg,max_logit_elements=32*31)
        with self.assertRaisesRegex(ValueError,'allocation bound'):_ce(self.a,h[:,:1],target[:1],config=cfg)

    def test_mixed_precision_identity_rejected(self):
        from mixtral_lab.loop_scale import evaluate
        h=torch.randn(1,3,16);target=torch.zeros(3,dtype=torch.long)
        with self.assertRaisesRegex(ValueError,'protocols differ'):_ce(self.a,h,target,self.b,config=config())
        with self.assertRaisesRegex(ValueError,'protocols differ'):evaluate(self.a,[],self.b,config=config())

    def test_native_fidelity_gate_is_retained(self):
        cache=cache_documents(self.a,rows())[0]
        result=preflight(self.a,cache,self.cfg)
        self.assertTrue(result['passed'])
        self.assertEqual(result['comparisons']['native_ce']['mismatched_elements'],0)
        from mixtral_lab import canonical_execution as ce
        original=ce.readout
        def biased(*args,**kwargs):
            logits=original(*args,**kwargs).clone();logits[...,0]+=.5;return logits
        with patch.object(ce,'readout',side_effect=biased):
            result=preflight(self.a,cache,self.cfg)
        self.assertFalse(result['passed'])
        self.assertGreater(result['comparisons']['native_ce']['mismatched_elements'],0)

    def test_path_trace_matches_actual_and_no_model_mutation(self):
        cache=cache_documents(self.a,rows())[0];mask=torch.zeros(cache.native_h1.shape[-2],dtype=torch.bool);mask[-1]=True
        state=digest(self.a.model.state_dict());branch=digest(self.b.state_dict())
        result=diagnose_path(self.a,cache,self.b,mask)
        self.assertTrue(result['observer_matches_untraced']);self.assertIsNone(result['first_tolerance_failure'])
        self.assertTrue(all(v['mismatched_elements']==0 for v in result['fixed_hidden_readout'].values()))
        json.dumps(result,allow_nan=False)
        self.assertEqual(state,digest(self.a.model.state_dict()));self.assertEqual(branch,digest(self.b.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.b.parameters()))

    def test_trace_catches_native_shape_sensitive_router(self):
        cache=cache_documents(self.a,rows())[0];mask=torch.zeros(cache.native_h1.shape[-2],dtype=torch.bool);mask[-1]=True
        native=ScaleBranch(16,config());original=self.a.moe.gate.forward
        def biased(h):
            logits,weights,pairs=original(h)
            return logits+len(h)*.1,weights,pairs
        with patch.object(self.a.moe.gate,'forward',side_effect=biased):
            result=diagnose_path(self.a,cache,native,mask)
        self.assertEqual(result['first_tolerance_failure'],'router_logits')

    def test_observer_mismatch_suppresses_localization(self):
        cache=cache_documents(self.a,rows())[0];mask=torch.zeros(cache.native_h1.shape[-2],dtype=torch.bool);mask[-1]=True
        original=self.b.forward
        def changed(*args,**kwargs):
            value=original(*args,**kwargs)
            return value+.1 if kwargs.get('trace') is not None else value
        with patch.object(self.b,'forward',side_effect=changed):result=diagnose_path(self.a,cache,self.b,mask)
        self.assertEqual(result['status'],'observer_mismatch');self.assertIsNone(result['first_tolerance_failure'])

    def test_path_failure_keeps_completed_conditionals_and_budget(self):
        from mixtral_lab.loop_scale import verify_conditional
        cache=cache_documents(self.a,rows())[0];native=ScaleBranch(16,config());original=self.a.moe.experts.forward
        budget={'remaining':1}
        with patch.object(self.a.moe.experts,'forward',side_effect=lambda h,p,w:original(h,p,w)+len(h)*.1),patch(
                'mixtral_lab.canonical_execution.diagnose_path',side_effect=RuntimeError('trace error')):
            with self.assertRaisesRegex(RuntimeError,'trace error') as caught:
                verify_conditional(self.a,[cache],native,path_probe_budget=budget)
        self.assertEqual(budget['remaining'],0)
        evidence=caught.exception.diagnostic_evidence
        self.assertTrue(evidence['mask'].startswith('singleton'))
        self.assertGreater(len(evidence['conditional_records']),2)
        json.dumps(evidence,allow_nan=False)

    def test_trace_cleanup_after_readout_failure(self):
        cache=cache_documents(self.a,rows())[0];mask=torch.zeros(cache.native_h1.shape[-2],dtype=torch.bool);mask[-1]=True
        native=ScaleBranch(16,config())
        with patch.object(self.a.model.lm_head,'forward',side_effect=RuntimeError('head failure')):
            with self.assertRaisesRegex(RuntimeError,'head failure'):diagnose_path(self.a,cache,native,mask)
        self.assertFalse(self.a.model.model.norm._forward_hooks)

    def test_tiled_preserves_gradients_and_bounds(self):
        x=torch.randn(33,4,requires_grad=True);work={};sizes=[]
        def fn(x):sizes.append(len(x));return x.square()
        y=tiled(x,fn,work=work);y.sum().backward()
        self.assertEqual(sizes,[32,32]);torch.testing.assert_close(x.grad,2*x.detach())
        self.assertEqual(work['canonical_padded_row_attempts'],64)
        with self.assertRaises(ValueError):tiled(torch.empty(0,4),fn)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA canonical protocol requires GPU')
    def test_cuda_bf16_singleton_permutation(self):
        self.a=MixtralAdapter(tiny().cuda().bfloat16());self.b=ScaleBranch(16,self.cfg,device='cuda')
        h=torch.randn(1,33,16,device='cuda',dtype=torch.bfloat16)
        full=self.b(self.a,h)
        for i in (0,1,31,32):
            torch.testing.assert_close(self.b(self.a,h[:,i:i+1]),full[:,i:i+1],atol=1e-4,rtol=1e-4)


class CanonicalScreenTests(unittest.TestCase):
    def test_three_modes_config_and_24_cases_no_certificate(self):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from experiments import mixtral_loop_scale as cli
        import time
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            c=dict(run_root=td,r2_precision=MODE,preflight_documents=4,amplitudes=[1.,.1],preflight_wall_seconds=300,max_document_bytes=4096)
            stack.enter_context(patch.object(cli,'suite_deadline',return_value=time.time()+300))
            stack.enter_context(patch.object(cli,'load_manifest',return_value=[{}]*4))
            stack.enter_context(patch.object(cli,'scale_config',return_value=config()))
            stack.enter_context(patch('torch.cuda.device_count',return_value=0))
            stack.enter_context(patch('mixtral_lab.document_training.cache_documents',return_value=[SimpleNamespace(document_id='unit')]))
            budgets=[]
            def run(a,cache,cfg,**kw):
                budgets.append(kw['path_probe_budget'])
                return dict(passed=cfg.r2_precision==MODE,r2_precision=cfg.r2_precision,backbone_gradients_absent=True,backbone_parameter_versions_unchanged=True)
            stack.enter_context(patch('mixtral_lab.loop_scale.preflight',side_effect=run))
            result=cli.precision_screen(object(),{},c,{},'x',{'freeze_sha256':'b'*64})
            self.assertEqual(result['completed_cases'],24);self.assertEqual(result['expected_cases'],24)
            self.assertEqual(set(result['numerical_checks_passed']),{'native_bf16','fp32_experts',MODE})
            self.assertTrue(result['numerical_checks_passed'][MODE])
            self.assertTrue(all(x is budgets[0] for x in budgets));self.assertEqual(budgets[0]['remaining'],12)
            self.assertFalse((Path(td)/'preflight.json').exists())


if __name__=='__main__':unittest.main()
