"""CPU tests validate observation/control flow, never certify CUDA mitigation."""
import copy
import json
import time
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.expert_precision_diagnostics import diagnose_expert_precision, _flags
from mixtral_lab.loop_scale import digest, preflight
from tests.test_mixtral_adapter import tiny
from tests.test_loop_scale import config
from tests.test_mixtral_bridge import rows
from mixtral_lab.document_training import cache_documents


class ShapeBias(TorchDispatchMode):
    def __init__(self, stage):
        self.stage = stage
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        hit = ((self.stage == 'gate_up' and name == 'aten.mm.default' and args[1].shape[-1] == 48) or
               (self.stage == 'activation' and name == 'aten.silu.default') or
               (self.stage == 'product' and name == 'aten.mul.Tensor' and args[0].shape[-1] == 24) or
               (self.stage == 'down' and name == 'aten.mm.default' and args[1].shape[-1] == 16) or
               (self.stage == 'weighted' and name == 'aten.mul.Tensor' and isinstance(args[1],torch.Tensor) and args[1].ndim > 0 and args[1].shape[-1] == 1) or
               (self.stage == 'accumulation' and name == 'aten.index_add_.default'))
        if hit and result.dtype == torch.float32:
            if self.stage == 'accumulation':
                result.add_(len(result) * .01)
            else:
                result = result + len(result) * .01
        return result


class ExpertPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.a = MixtralAdapter(tiny())
        self.e = self.a.moe.experts
        self.h = torch.randn(12, 16)
        # Slot-major ordering; expert 7 occurs only outside selected rows.
        self.p = torch.tensor([[0,1],[7,0],[1,2],[0,2]]*3)
        self.w = torch.softmax(torch.randn(12,2), dim=-1)
        self.idx = torch.tensor([0,2,4,6,8,10])

    def run_probe(self, **kwargs):
        return diagnose_expert_precision(self.e,self.h,self.p,self.w,self.idx,
            dense_output=self.e(self.h,self.p,self.w),
            compact_output=self.e(self.h[self.idx],self.p[self.idx],self.w[self.idx]), **kwargs)

    def test_actual_trace_alignment_reference_and_no_mutation(self):
        before = digest(self.a.model.state_dict()); work = {}
        result = self.run_probe(work=work)
        self.assertEqual(result['status'],'complete')
        self.assertTrue(result['observer_matches_untraced'])
        self.assertEqual(result['sampled_tokens'],6)
        self.assertEqual({x['expert'] for x in result['experts']},{0,1,2})
        self.assertEqual(result['reduction_probe']['status'],'not_cuda_bf16_tested')
        self.assertEqual(work['precision_expert_forward_attempts'],2)
        self.assertEqual(work['precision_expert_forward_completed'],2)
        self.assertEqual(work['precision_reference_projection_calls'],6)
        self.assertEqual(before,digest(self.a.model.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.a.model.parameters()))
        json.dumps(result,allow_nan=False)
        for x in result['experts']:
            for metrics in x['comparisons'].values():
                self.assertEqual(metrics['compact_vs_fp32']['mismatched_elements'],0)

    def test_sample_bound_does_not_change_forward_shapes(self):
        self.idx = torch.arange(11)
        result = self.run_probe()
        self.assertEqual(result['sampled_tokens'],8)
        self.assertEqual(result['selected_tokens'],11)
        self.assertEqual(result['total_tokens'],12)
        self.assertEqual(result['baseline_output']['compact_vs_dense']['shape'],[11,16])

    def test_detects_first_differing_actual_primitive(self):
        for stage in ('gate_up','activation','product','down','weighted','accumulation'):
            with self.subTest(stage=stage), ShapeBias(stage):
                result = self.run_probe()
            self.assertTrue(result['observer_matches_untraced'])
            first = result['experts'][0]
            self.assertEqual(first['first_tolerance_failure'],stage)

    def test_observer_mismatch_cannot_localize(self):
        dense = self.e(self.h,self.p,self.w)+1
        result = diagnose_expert_precision(self.e,self.h,self.p,self.w,self.idx,
            dense_output=dense,compact_output=self.e(self.h[self.idx],self.p[self.idx],self.w[self.idx]))
        self.assertEqual(result['status'],'observer_mismatch')
        self.assertTrue(all(x['first_exact_difference'] is None for x in result['experts']))

    def test_untraced_extra_output_operation_not_attributed_to_primitive(self):
        original=self.e.forward
        with patch.object(self.e,'forward',side_effect=lambda h,p,w:original(h,p,w)+.1):
            result=self.run_probe()
        self.assertEqual(result['status'],'observer_mismatch')

    def test_bf16_reference_uses_existing_quantized_weights(self):
        self.e.to(torch.bfloat16);self.h=self.h.bfloat16()
        result=self.run_probe()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['baseline_output']['dense_vs_fp32']['left_dtype'],'torch.bfloat16')
        self.assertEqual(result['baseline_output']['dense_vs_fp32']['right_dtype'],'torch.float32')
        self.assertEqual(self.e.gate_up_proj.dtype,torch.bfloat16)

    def test_restores_all_precision_modes_and_exception(self):
        backend=torch.backends.cuda.matmul
        initial=(torch.get_float32_matmul_precision(),backend.allow_bf16_reduced_precision_reduction)
        try:
            for precision in ('highest','high','medium'):
                for flag in (True,False):
                    torch.set_float32_matmul_precision(precision)
                    backend.allow_bf16_reduced_precision_reduction=flag
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        with _flags(reduction=False,tf32=False):
                            self.assertEqual(torch.get_float32_matmul_precision(),'highest')
                            self.assertFalse(backend.allow_bf16_reduced_precision_reduction)
                            raise RuntimeError('injected')
                    self.assertEqual(torch.get_float32_matmul_precision(),precision)
                    self.assertEqual(backend.allow_bf16_reduced_precision_reduction,flag)
        finally:
            torch.set_float32_matmul_precision(initial[0])
            backend.allow_bf16_reduced_precision_reduction=initial[1]

    def test_reference_failure_restores_flags_and_has_no_completed_counter(self):
        backend=torch.backends.cuda.matmul;before=torch.get_float32_matmul_precision();work={}
        original=torch.nn.functional.linear
        # Actual native trace is already complete when the reference starts.
        def fail(*args,**kwargs):
            if not backend.allow_tf32: raise RuntimeError('reference failure')
            return original(*args,**kwargs)
        try:
            torch.set_float32_matmul_precision('high')
            with patch('torch.nn.functional.linear',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'reference failure'):self.run_probe(work=work)
            self.assertTrue(backend.allow_tf32)
            self.assertEqual(work.get('precision_probe_completed',0),0)
        finally:torch.set_float32_matmul_precision(before)

    def test_deadline_fails_closed(self):
        work={}
        with self.assertRaisesRegex(TimeoutError,'deadline'):
            self.run_probe(work=work,deadline=time.monotonic()-1)
        self.assertEqual(work.get('precision_expert_forward_attempts',0),0)

    def test_invalid_indices_and_routes(self):
        for idx in (torch.tensor([2,0]),torch.tensor([0,0]),torch.arange(12),torch.tensor([-1,0])):
            self.idx=idx
            with self.assertRaises(ValueError):self.run_probe()
        self.idx=torch.tensor([0,2]);self.p[:,1]=self.p[:,0]
        with self.assertRaisesRegex(ValueError,'unique Top-2'):self.run_probe()

    def test_unsupported_backend_rejected_before_forward(self):
        self.e.config._experts_implementation='grouped_mm'
        with self.assertRaisesRegex(ValueError,'eager implementation'):
            diagnose_expert_precision(self.e,self.h,self.p,self.w,self.idx,
                dense_output=torch.zeros_like(self.h),compact_output=torch.zeros_like(self.h[self.idx]))

    def test_shared_budget_does_not_change_failed_gate(self):
        cache=cache_documents(self.a,rows())[0]
        original=self.e.forward;budget={'remaining':2};calls=[]
        def fake(*args,**kwargs):calls.append(1);return {'status':'unit_stub'}
        with patch.object(self.e,'forward',side_effect=lambda h,p,w:original(h,p,w)+len(h)*.1), patch(
                'mixtral_lab.expert_precision_diagnostics.diagnose_expert_precision',side_effect=fake):
            first=preflight(self.a,cache,config(),expert_probe_budget=budget)
            second=preflight(self.a,cache,config(),expert_probe_budget=budget)
        self.assertFalse(first['passed']);self.assertFalse(second['passed'])
        self.assertEqual(len(calls),2);self.assertEqual(budget['remaining'],0)
        self.assertEqual(first['tolerances'],{'atol':1e-4,'rtol':1e-4})

    def test_failure_preserves_layer_evidence_and_work(self):
        from mixtral_lab.loop_parity_diagnostics import diagnose_compaction
        from mixtral_lab.loop_scale import ScaleBranch
        cache=cache_documents(self.a,rows())[0]
        branch=ScaleBranch(16,config(),amplitude=1.)
        mask=torch.zeros(cache.native_h1.shape[-2],dtype=torch.bool)
        mask[[0,2]]=True
        def failed(*args,**kwargs):
            kwargs['progress'].update(stage='fp32_reference',completed_traced_forwards=2)
            raise TimeoutError('injected deadline')
        with patch('mixtral_lab.expert_precision_diagnostics.diagnose_expert_precision',side_effect=failed):
            with self.assertRaises(TimeoutError) as caught:
                diagnose_compaction(self.a,cache,branch,mask,expert_probe_budget={'remaining':2})
        evidence=caught.exception.diagnostic_evidence
        self.assertIn('fixed_dense_inputs_routes_expert_only',evidence['layer_diagnostic'])
        self.assertEqual(evidence['expert_precision_partial']['completed_traced_forwards'],2)
        self.assertEqual(evidence['work']['diagnostic_completed'],1)
        json.dumps(evidence,allow_nan=False)
        self.assertFalse(self.e._forward_hooks)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA BF16 probe requires GPU')
    def test_cuda_bf16_probe_restores_reduction_flag(self):
        self.e.cuda().bfloat16();self.h=self.h.cuda().bfloat16()
        self.p=self.p.cuda();self.w=self.w.cuda();self.idx=self.idx.cuda()
        old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        result=self.run_probe()
        self.assertEqual(result['reduction_probe']['status'],'complete')
        self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,old)


if __name__=='__main__':unittest.main()
