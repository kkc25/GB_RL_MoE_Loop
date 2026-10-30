"""Regression tests inject batch-shape-sensitive operators, not GPU claims."""
import copy
import unittest
from unittest.mock import patch
import torch
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.document_training import cache_documents
from mixtral_lab.loop_scale import ScaleBranch, preflight, verify_conditional, digest
from mixtral_lab.loop_parity_diagnostics import diagnose_compaction
from tests.test_loop_scale import config
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_bridge import rows


class ParityDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.a=MixtralAdapter(tiny())
        self.cache=cache_documents(self.a,rows())[0]
        self.branch=ScaleBranch(16,config(),amplitude=1.)
        self.mask=torch.zeros(self.cache.native_h1.shape[-2],dtype=torch.bool)
        self.mask[[0,2]]=True

    def test_observes_actual_forward_no_mutation_and_no_hooks_left(self):
        before=digest(self.branch.state_dict()); backbone=digest(self.a.model.state_dict())
        work={}
        result=diagnose_compaction(self.a,self.cache,self.branch,self.mask,work=work)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['selected_rows'],2)
        self.assertIsNone(result['first_tolerance_failure'])
        self.assertEqual(digest(self.branch.state_dict()),before)
        self.assertEqual(digest(self.a.model.state_dict()),backbone)
        self.assertTrue(all(p.grad is None for p in self.branch.parameters()))
        self.assertEqual(work['diagnostic_attempts'],1)
        self.assertEqual(work['diagnostic_completed'],1)
        self.assertEqual(work['extra_expert_forward_attempts'],3)
        self.assertEqual(work['head_forward_attempts'],2)
        for m in (self.a.layer.post_attention_layernorm,self.branch.correction,self.a.moe.gate,self.a.moe.experts):
            self.assertFalse(m._forward_hooks)
            self.assertFalse(m._forward_pre_hooks)

    def test_pinpoints_expert_shape_sensitivity_with_identical_input(self):
        original=self.a.moe.experts.forward
        def biased(h,p,w): return original(h,p,w)+len(h)*.1
        with patch.object(self.a.moe.experts,'forward',side_effect=biased):
            result=diagnose_compaction(self.a,self.cache,self.branch,self.mask)
        self.assertEqual(result['first_tolerance_failure'],'expert_output')
        self.assertGreater(result['fixed_dense_inputs_routes_expert_only']['mismatched_elements'],0)
        self.assertEqual(result['comparisons']['router_pairs']['mismatched_elements'],0)

    def test_pinpoints_normalization_and_keeps_failure(self):
        original=self.a.layer.post_attention_layernorm.forward
        def biased(h): return original(h)+h.shape[-2]*.1
        with patch.object(self.a.layer.post_attention_layernorm,'forward',side_effect=biased):
            result=verify_conditional(self.a,[self.cache],self.branch,diagnostics=True)
        self.assertFalse(result['passed'])
        traces=[x['layer_diagnostic'] for x in result['documents'] if 'layer_diagnostic' in x]
        self.assertEqual(len(traces),1)
        self.assertEqual(traces[0]['first_tolerance_failure'],'normalized_input')
        self.assertEqual(result['tolerances'],{'atol':1e-4,'rtol':1e-4})

    def test_cleanup_after_exception(self):
        with patch.object(self.a.moe.experts,'forward',side_effect=RuntimeError('injected failure')):
            with self.assertRaisesRegex(RuntimeError,'injected failure'):
                diagnose_compaction(self.a,self.cache,self.branch,self.mask)
        for m in (self.a.layer.post_attention_layernorm,self.branch.correction,self.a.moe.gate,self.a.moe.experts):
            self.assertFalse(m._forward_hooks);self.assertFalse(m._forward_pre_hooks)

    def test_stop_and_all_not_mislabelled_as_mixed_diagnostics(self):
        for mask in (torch.zeros_like(self.mask),torch.ones_like(self.mask)):
            with self.assertRaisesRegex(ValueError,'mixed mask'):
                diagnose_compaction(self.a,self.cache,self.branch,mask)

    def test_preflight_records_both_parameter_states(self):
        result=preflight(self.a,self.cache,config())
        self.assertTrue(result['passed'])
        self.assertTrue(result['conditional_parity_before_update']['passed'])
        for key in ('conditional_parity_before_update','conditional_parity'):
            self.assertEqual({r['mask'] for r in result[key]['documents']},
                {'stop','all','mixed','mixed_odd','singleton_first','singleton_last'})

    def test_preflight_does_not_turn_known_mismatch_into_pass(self):
        original=self.a.moe.experts.forward
        with patch.object(self.a.moe.experts,'forward',side_effect=lambda h,p,w:original(h,p,w)+len(h)*.1):
            result=preflight(self.a,self.cache,config())
        self.assertFalse(result['passed'])
        self.assertFalse(result['conditional_parity']['passed'])
        self.assertIn('layer_diagnostic',result['conditional_parity']['documents'][2])


class PreflightCoverageTests(unittest.TestCase):
    def exercise(self, directory, results, *, budget_failure=False, documents=4):
        from contextlib import ExitStack
        from pathlib import Path
        from types import SimpleNamespace
        import json
        from experiments import mixtral_loop_scale as cli
        root=Path(directory)
        c=dict(run_root=str(root),preflight_documents=4,amplitudes=[1.,.1],
               preflight_wall_seconds=300,max_document_bytes=1024)
        code={'commit':'a'*40,'sources':{}};index={'freeze_sha256':'b'*64}
        (root/'preparation.json').write_text(json.dumps(dict(code=code,config=c,freeze_sha256=index['freeze_sha256'])))
        cache=SimpleNamespace(document_id='unit')
        with ExitStack() as stack:
            stack.enter_context(patch.object(cli,'frozen',return_value=(root,index)))
            stack.enter_context(patch.object(cli,'gpu_adapter',return_value=(object(),{})))
            stack.enter_context(patch.object(cli,'load_manifest',return_value=[{}]*documents))
            stack.enter_context(patch.object(cli,'scale_config',return_value=config()))
            stack.enter_context(patch.object(cli,'memory_peaks',return_value={}))
            stack.enter_context(patch.object(cli,'check_suite_deadline',side_effect=[None,None,TimeoutError('budget')] if budget_failure else None))
            stack.enter_context(patch('torch.cuda.synchronize'))
            stack.enter_context(patch('mixtral_lab.document_training.cache_documents',return_value=[cache]))
            mock=stack.enter_context(patch('mixtral_lab.loop_scale.preflight',side_effect=results))
            try: cli.execute_prepared(SimpleNamespace(stage='preflight'),c,code)
            finally: self.calls=mock.call_count

    @staticmethod
    def case(passed=False, safe=True):
        return dict(passed=passed,backbone_gradients_absent=safe,backbone_parameter_versions_unchanged=safe)

    def test_numeric_failure_collects_all_eight_but_never_certifies(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'preflight failed'):
                self.exercise(td,[self.case(False)]+[self.case(True) for _ in range(7)])
            self.assertEqual(self.calls,8)
            evidence=json.loads((Path(td)/'preflight_progress.json').read_text())
            self.assertEqual(evidence['status'],'failed');self.assertEqual(evidence['completed_cases'],8)
            self.assertFalse((Path(td)/'preflight.json').exists())

    def test_insufficient_documents_cannot_certify_empty_all(self):
        import tempfile
        from pathlib import Path
        for documents in (0,3):
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaisesRegex(ValueError,'insufficient documents'):
                    self.exercise(td,[],documents=documents)
                self.assertEqual(self.calls,0)
                self.assertFalse((Path(td)/'preflight.json').exists())

    def test_complete_success_requires_eight_cases(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            self.exercise(td,[self.case(True) for _ in range(8)])
            self.assertEqual(self.calls,8)
            evidence=json.loads((Path(td)/'preflight.json').read_text())
            self.assertEqual(evidence['status'],'complete')
            self.assertEqual(len(evidence['results']),8)

    def test_unsafe_backbone_stops_immediately(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'backbone integrity'):
                self.exercise(td,[self.case(False,False)])
            self.assertEqual(self.calls,1);self.assertFalse((Path(td)/'preflight.json').exists())

    def test_budget_failure_preserves_partial_without_certificate(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(TimeoutError,'budget'):
                self.exercise(td,[self.case(False)],budget_failure=True)
            self.assertEqual(self.calls,1)
            self.assertEqual(json.loads((Path(td)/'preflight_progress.json').read_text())['status'],'incomplete')
            self.assertFalse((Path(td)/'preflight.json').exists())

if __name__=='__main__':unittest.main()
