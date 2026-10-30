"""First-order candidate tests; real GPU numerical acceptance remains separate."""
from dataclasses import replace, asdict
from pathlib import Path
from unittest.mock import patch
import copy
import json
import tempfile
import time
import unittest

import torch
import torch.nn.functional as F

from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.fp32_experts import frozen_linear, precise_linear, experts_fp32, precision_scope
from mixtral_lab.loop_scale import ScaleBranch, preflight, verify_gradient_parity, digest, _effective_expert_input
from mixtral_lab.document_training import cache_documents
from tests.test_mixtral_adapter import tiny
from tests.test_loop_scale import config
from tests.test_mixtral_bridge import rows


def independent(experts, x, pairs, weights):
    # Independent per-token/slot equation, not the implementation's expert loop.
    outputs=[]
    for t in range(len(x)):
        value=torch.zeros_like(x[t])
        for slot in range(2):
            e=int(pairs[t,slot])
            gate,up=F.linear(x[t:t+1],experts.gate_up_proj[e].float()).chunk(2,-1)
            term=F.linear(F.silu(gate)*up,experts.down_proj[e].float())[0]
            value=value+weights[t,slot]*term
        outputs.append(value)
    return torch.stack(outputs)


class FP32ExpertTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.a=MixtralAdapter(tiny().bfloat16())
        self.e=self.a.moe.experts
        self.x=torch.randn(7,16,requires_grad=True)
        self.p=torch.tensor([[0,1],[2,0],[1,2],[7,1],[0,7],[2,1],[1,7]])
        self.w=torch.softmax(torch.randn(7,2),dim=-1)

    def test_forward_and_input_gradient_match_independent_fp32(self):
        x2=self.x.detach().clone().requires_grad_()
        before=digest(self.e.state_dict())
        with precision_scope('cpu'):
            y=experts_fp32(self.e,self.x,self.p,self.w)
            reference=independent(self.e,x2,self.p,self.w)
            g=torch.randn_like(y)
            actual=torch.autograd.grad(y,self.x,g)[0]
            expected=torch.autograd.grad(reference,x2,g)[0]
        torch.testing.assert_close(y,reference,atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)
        self.assertEqual(y.dtype,torch.float32)
        self.assertEqual(before,digest(self.e.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.e.parameters()))

    def test_saved_expert_weights_are_original_bf16_not_full_fp32_bank(self):
        saved=[]
        def pack(t):
            saved.append((tuple(t.shape),t.dtype,t.untyped_storage().data_ptr()))
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            y=experts_fp32(self.e,self.x,self.p,self.w)
            y.sum().backward()
        weight_shapes={(48,16),(16,24)}
        projections=[x for x in saved if x[0] in weight_shapes]
        self.assertTrue(projections)
        storage={self.e.gate_up_proj.untyped_storage().data_ptr(),self.e.down_proj.untyped_storage().data_ptr()}
        self.assertTrue(all(dtype==torch.bfloat16 and ptr in storage for _,dtype,ptr in projections))

    def test_trainable_linear_gradient_matches_fp32_autograd(self):
        w=torch.randn(5,16,requires_grad=True);w2=w.detach().clone().requires_grad_()
        x2=self.x.detach().clone().requires_grad_()
        with precision_scope('cpu'):
            y=precise_linear(self.x,w);z=F.linear(x2,w2);g=torch.randn_like(y)
            actual=torch.autograd.grad(y,(self.x,w),g)
            expected=torch.autograd.grad(z,(x2,w2),g)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_autocast_disabled_forward_and_backward(self):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            y=frozen_linear(self.x,self.e.gate_up_proj[0]);self.assertEqual(y.dtype,torch.float32)
            y.sum().backward();self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(self.x.grad.dtype,torch.float32)

    def test_precision_restored_all_modes_on_forward_and_backward_error(self):
        old=torch.get_float32_matmul_precision()
        try:
            for mode in ('highest','high','medium'):
                torch.set_float32_matmul_precision(mode)
                y=frozen_linear(self.x,self.e.gate_up_proj[0])
                self.assertEqual(torch.get_float32_matmul_precision(),mode)
                with patch('torch.matmul',side_effect=RuntimeError('backward injected')):
                    with self.assertRaisesRegex(RuntimeError,'backward injected'):y.sum().backward()
                self.assertEqual(torch.get_float32_matmul_precision(),mode)
                with patch('torch.nn.functional.linear',side_effect=RuntimeError('forward injected')):
                    with self.assertRaisesRegex(RuntimeError,'forward injected'):frozen_linear(self.x,self.e.gate_up_proj[0])
                self.assertEqual(torch.get_float32_matmul_precision(),mode)
        finally:torch.set_float32_matmul_precision(old)

    def test_frozen_guard_invalid_routes_and_deadline(self):
        with self.assertRaisesRegex(ValueError,'frozen'):
            frozen_linear(self.x,torch.randn(5,16,requires_grad=True))
        with self.assertRaises(TimeoutError):
            experts_fp32(self.e,self.x,self.p,self.w,deadline=time.time()-1)
        p=self.p.clone();p[:,1]=p[:,0]
        with self.assertRaisesRegex(ValueError,'distinct'):experts_fp32(self.e,self.x,p,self.w)

    def test_compaction_full_subset_and_singleton(self):
        full=experts_fp32(self.e,self.x,self.p,self.w)
        for ids in (torch.tensor([0,2,5]),torch.tensor([6]),torch.arange(7)):
            actual=experts_fp32(self.e,self.x[ids],self.p[ids],self.w[ids])
            torch.testing.assert_close(actual,full[ids],atol=1e-6,rtol=1e-5)


class FP32BranchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.a=MixtralAdapter(tiny())
        self.cache=cache_documents(self.a,rows())[0]
        self.cfg=config(r2_precision='fp32_experts')
        self.branch=ScaleBranch(16,self.cfg)

    def test_candidate_preflight_enforces_gradient_checks_and_no_legacy_trace(self):
        with patch('mixtral_lab.loop_parity_diagnostics.diagnose_compaction',side_effect=AssertionError('wrong backend observer')):
            result=preflight(self.a,self.cache,self.cfg,gradient_checks=False)
        self.assertTrue(result['passed'])
        self.assertTrue(result['gradient_parity_before_update']['passed'])
        self.assertTrue(result['gradient_parity_after_update']['passed'])
        self.assertGreater(result['work']['fp32_r2_forward_completed'],0)
        self.assertEqual(result['tolerances'],{'atol':1e-4,'rtol':1e-4})
        self.assertIsNotNone(result['train_document_loss_snapshot']['r2_after_update'])

    def test_failed_gradient_parity_cannot_be_bypassed(self):
        with patch('mixtral_lab.loop_scale.verify_gradient_parity',return_value={'passed':False}) as check:
            result=preflight(self.a,self.cache,self.cfg,gradient_checks=False)
        self.assertFalse(result['passed']);self.assertEqual(check.call_count,2)

    def test_late_backbone_mutation_cannot_escape_integrity_check(self):
        calls=[]
        def probe(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:
                with torch.no_grad():next(self.a.model.parameters()).add_(.001)
            return {'passed':True}
        with patch('mixtral_lab.loop_scale.verify_gradient_parity',side_effect=probe):
            result=preflight(self.a,self.cache,self.cfg)
        self.assertFalse(result['backbone_parameter_versions_unchanged'])
        self.assertFalse(result['passed'])

    def test_preflight_deadline_applies_to_both_precision_modes(self):
        for mode in ('native_bf16','fp32_experts'):
            with self.assertRaises(TimeoutError):
                preflight(self.a,self.cache,replace(self.cfg,r2_precision=mode,absolute_deadline_utc=time.time()-1))

    def test_gradient_probe_preserves_existing_gradients_and_parameters(self):
        for p in self.branch.parameters():p.grad=torch.ones_like(p)
        state=digest(self.branch.state_dict())
        result=verify_gradient_parity(self.a,self.cache,self.branch)
        self.assertTrue(result['passed'])
        self.assertEqual(state,digest(self.branch.state_dict()))
        self.assertTrue(all(torch.equal(p.grad,torch.ones_like(p)) for p in self.branch.parameters()))

    def test_stop_zero_experts_and_unselected_rows_unchanged(self):
        h=self.cache.native_h1;work={};n=h.shape[-2]
        stopped=self.branch(self.a,h,torch.zeros(n,dtype=torch.bool),work=work)
        self.assertTrue(torch.equal(h,stopped));self.assertEqual(work,{})
        mask=torch.zeros(n,dtype=torch.bool);mask[0]=True
        out=self.branch(self.a,h,mask,work=work)
        self.assertTrue(torch.equal(out.reshape(n,-1)[1:],h.reshape(n,-1)[1:]))
        self.assertEqual(work['fp32_r2_expert_token_calls'],2)

    def test_effective_input_uses_fp32_and_protocol_changes_identity(self):
        x=torch.ones(2,16,dtype=torch.bfloat16)
        with torch.no_grad():self.branch.correction.up.fill_(1e-4)
        actual=_effective_expert_input(self.branch,x)
        self.assertEqual(actual.dtype,torch.float32)
        self.assertFalse(torch.equal(actual,x.float()))
        self.assertNotEqual(digest(asdict(self.cfg)),digest(asdict(config())))
        self.assertEqual(ScaleBranch(16,config()).scale_config.r2_precision,'native_bf16')

    def test_native_default_preflight_does_not_add_gradient_work(self):
        result=preflight(self.a,self.cache,config())
        self.assertIsNone(result['gradient_parity_before_update'])
        self.assertNotIn('fp32_r2_forward_attempts',result['work'])

    def test_config_validation(self):
        from experiments.mixtral_loop_scale import validate_config
        path=Path(__file__).resolve().parents[1]/'configs/mixtral/loop_scale_fp32_experts.json'
        c=validate_config(json.loads(path.read_text()))
        self.assertEqual(c['r2_precision'],'fp32_experts')
        c['r2_precision']='silent_fallback'
        with self.assertRaises(ValueError):validate_config(c)
        with self.assertRaises(ValueError):replace(self.cfg,r2_precision='silent_fallback').validate()


class PrecisionScreenTests(unittest.TestCase):
    def exercise(self, root, *, fail=False, unsafe=False):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from experiments import mixtral_loop_scale as cli
        c=dict(run_root=str(root),preflight_wall_seconds=300,preflight_documents=4,
               amplitudes=[1.,.1],max_document_bytes=4096)
        calls=[]
        def run(a,cache,cfg,**kwargs):
            calls.append((cfg.r2_precision,kwargs))
            if fail and len(calls)==3:raise RuntimeError('candidate exception')
            return dict(passed=cfg.r2_precision=='fp32_experts',r2_precision=cfg.r2_precision,
                backbone_gradients_absent=not unsafe,backbone_parameter_versions_unchanged=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(cli,'suite_deadline',return_value=time.time()+200))
            stack.enter_context(patch.object(cli,'load_manifest',return_value=[{}]*4))
            stack.enter_context(patch.object(cli,'scale_config',return_value=config()))
            stack.enter_context(patch('torch.cuda.device_count',return_value=0))
            stack.enter_context(patch('mixtral_lab.document_training.cache_documents',return_value=[SimpleNamespace(document_id='unit')]))
            stack.enter_context(patch('mixtral_lab.loop_scale.preflight',side_effect=run))
            try:return cli.precision_screen(object(),{},c,{'commit':'a'*40},'path',{'freeze_sha256':'b'*64}),calls
            finally:self.calls=calls

    def test_paired_coverage_no_training_certificate(self):
        with tempfile.TemporaryDirectory() as td:
            result,calls=self.exercise(Path(td))
            self.assertEqual(len(calls),16)
            self.assertEqual(result['numerical_checks_passed'],{'native_bf16':False,'fp32_experts':True})
            self.assertTrue(all(k['gradient_checks'] and not k['diagnostics'] for _,k in calls))
            self.assertFalse((Path(td)/'preflight.json').exists())
            self.assertTrue((Path(td)/'precision_screen.json').exists())
            self.assertEqual([x[0] for x in calls[:4]],['native_bf16','fp32_experts','fp32_experts','native_bf16'])

    def test_exception_preserves_partial_and_no_certificate(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,'candidate exception'):self.exercise(Path(td),fail=True)
            p=json.loads((Path(td)/'precision_screen_progress.json').read_text())
            self.assertEqual(p['status'],'failed');self.assertEqual(p['completed_cases'],2)
            self.assertFalse((Path(td)/'precision_screen.json').exists())
            self.assertFalse((Path(td)/'preflight.json').exists())

    def test_backbone_violation_stops_screen(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'backbone integrity'):self.exercise(Path(td),unsafe=True)
            self.assertEqual(len(self.calls),1)

    def test_attempted_screen_rejected_before_model_load(self):
        from types import SimpleNamespace
        from experiments import mixtral_loop_scale as cli
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c={'run_root':td,'r2_precision':'fp32_experts'};code={'commit':'a'*40};index={'freeze_sha256':'b'*64}
            (root/'preparation.json').write_text(json.dumps(dict(code=code,config=c,freeze_sha256=index['freeze_sha256'])))
            (root/'precision_screen_progress.json').write_text('{}')
            with patch.object(cli,'frozen',return_value=('path',index)),patch.object(cli,'gpu_adapter') as load:
                with self.assertRaises(FileExistsError):cli.execute_prepared(SimpleNamespace(stage='precision-screen'),c,code)
                load.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA candidate screen requires GPU')
    def test_cuda_bf16_frozen_linear_backward(self):
        x=torch.randn(8,16,device='cuda',requires_grad=True)
        w=torch.randn(12,16,device='cuda',dtype=torch.bfloat16)
        y=frozen_linear(x,w);y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(y.dtype,torch.float32)


if __name__=='__main__':unittest.main()
