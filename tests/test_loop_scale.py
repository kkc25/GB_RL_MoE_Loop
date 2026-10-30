"""Real tiny HF experts, accumulation, bounded resume and protocol guards."""
from dataclasses import replace
import copy
from contextlib import contextmanager
import json
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.document_training import cache_documents
from mixtral_lab.loop_scale import (ScaleConfig,ScaleBranch,run_cell,evaluate,load_selected,
                                   load_candidate,preflight,digest,_load,_save,paired_comparison,RunLedger)
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_bridge import rows


def config(**kw):
    return replace(ScaleConfig(rank=2,epochs=2,positions_per_update=3,max_updates=100,
                               max_wall_seconds=300.,thresholds=(0.,.01)),**kw)


def populations(adapter):
    values={}
    for role in ('train','gate','calibration','dev'):
        rs=copy.deepcopy(rows())
        for index,row in enumerate(rs):
            row['id']=f'{role}-{index}'
        values[role]=cache_documents(adapter,rs)
    return values


def run(adapter,pops,out,cfg=None,**kw):
    return run_cell(adapter,pops['train'],pops['gate'],pops['dev'],cfg or config(),
                    {'model':'tiny','source':'unit'},out,calibration_caches=pops['calibration'],**kw)


class ScaleRuntimeTests(unittest.TestCase):
    def test_true_expert_scale_stop_and_frozen_backbone(self):
        for dtype in (torch.float32,torch.bfloat16):
            torch.manual_seed(19);a=MixtralAdapter(tiny(dtype));c=populations(a)['train'][0]
            before={k:v.clone() for k,v in a.model.state_dict().items()}
            m=ScaleBranch(16,config(),amplitude=.1);calls=[]
            handle=a.moe.experts.register_forward_pre_hook(lambda _,args:calls.append(len(args[0])))
            try:
                h=c.native_h1;mask=torch.zeros(h.shape[-2],dtype=torch.bool)
                self.assertTrue(torch.equal(m(a,h,mask),h));self.assertEqual(calls,[])
                mask[[0,2]]=True;out=m(a,h,mask);self.assertEqual(calls,[2])
                out.float().square().sum().backward()
                self.assertGreater(float(m.correction.up.grad.abs().sum()),0)
            finally:
                handle.remove()
            self.assertTrue(all(p.grad is None for p in a.model.parameters()))
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in a.model.state_dict().items()))
            self.assertFalse(torch.equal(out,h))

    def test_control_no_experts_and_parameter_match(self):
        a=MixtralAdapter(tiny());c=populations(a)['train'][0]
        r=ScaleBranch(16,config(),amplitude=.1)
        control=ScaleBranch(16,config(),'parameter_control',1.)
        self.assertEqual(sum(p.numel() for p in r.parameters()),sum(p.numel() for p in control.parameters()))
        with patch.object(a.moe.experts,'forward',side_effect=AssertionError('unexpected expert')):
            self.assertTrue(torch.equal(control(a,c.native_h1),c.native_h1))
        for amplitude in (0.,.2,True):
            with self.assertRaises(ValueError):ScaleBranch(16,config(),amplitude=amplitude)
        with self.assertRaises(ValueError):ScaleBranch(16,config(),'parameter_control',.1)

    def test_full_runtime_separate_selectors_and_evidence(self):
        a=MixtralAdapter(tiny());p=populations(a)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run';report=run(a,p,path)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['adapter_exposures'],10)
            self.assertEqual(report['adapter_updates'],4)
            self.assertEqual(report['gate_updates'],4)
            self.assertEqual([h['epoch'] for h in report['history']],[0,1,2])
            self.assertTrue(all(h['conditional_parity']['passed'] for h in report['history']))
            self.assertIn(report['forced_selection']['epoch'],(1,2))
            self.assertEqual(len(report['calibration']),2)
            self.assertIn(report['gated_selection']['mode'],('native','gate'))
            self.assertEqual(report['specs']['gate_fit']['unique_positions'],5)
            b,g,s=load_selected(path,config(),{'model':'tiny','source':'unit'},which='forced')
            self.assertIsNone(g);self.assertEqual(s['mode'],'forced')
            before=json.loads((path/'ledger.json').read_text())
            repeated=run(a,p,path,resume=True)
            self.assertEqual(report,repeated)
            self.assertEqual(len(before['events']),len(json.loads((path/'ledger.json').read_text())['events']))
            docs=list((path/'audit-epoch0'/'train').glob('document-*.json'))
            evidence=json.loads(docs[0].read_text())
            self.assertAlmostEqual(sum(evidence['evidence']['candidate_ce'])/evidence['evidence']['tokens'],evidence['evidence']['candidate_nll'])

    def test_resume_exact_optimizer_cursor_and_rng(self):
        torch.manual_seed(21);a=MixtralAdapter(tiny());p=populations(a)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);complete=run(a,p,root/'full')
            with self.assertRaises(InterruptedError):run(a,p,root/'resumed',interrupt_after_updates=1)
            initial=_load(root/'resumed/state.pt')
            self.assertEqual(initial['adapter_updates'],1);self.assertEqual(initial['exposure'],3)
            resumed=run(a,p,root/'resumed',resume=True)
            self.assertEqual(resumed['adapter_updates'],complete['adapter_updates'])
            for epoch in (0,1,2):
                left,_,_=load_candidate(root/f'full/candidate-epoch{epoch}.pt',config(),{'model':'tiny','source':'unit'})
                right,_,_=load_candidate(root/f'resumed/candidate-epoch{epoch}.pt',config(),{'model':'tiny','source':'unit'})
                self.assertEqual(digest(left.state_dict()),digest(right.state_dict()))
            self.assertEqual(resumed['forced_selection']['nll'],complete['forced_selection']['nll'])
            events=json.loads((root/'resumed/ledger.json').read_text())['events']
            self.assertEqual(sum(e['key']=='adapter-update:1' for e in events),1)

    def test_source_leakage_budget_and_identity_tamper(self):
        a=MixtralAdapter(tiny());p=populations(a)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            bad=copy.copy(p);bad['gate']=p['train']
            with self.assertRaises(ValueError):run(a,bad,root/'leak')
            self.assertFalse((root/'leak').exists())
            with self.assertRaises(ValueError):run(a,p,root/'budget',config(max_updates=1))
            self.assertFalse((root/'budget').exists())
            with self.assertRaises(InterruptedError):run(a,p,root/'state',interrupt_after_updates=1)
            changed=copy.deepcopy(p);changed['train'][0].native_h1.add_(.001)
            with self.assertRaises(ValueError):run(a,changed,root/'state',resume=True)
            state=_load(root/'state/state.pt');state['exposure']=999;_save(root/'state/state.pt',state)
            with self.assertRaises(ValueError):run(a,p,root/'state',resume=True)
            # Candidate identity changes also reject checkpoint loading.
            with self.assertRaises(ValueError):load_candidate(root/'state/candidate-epoch0.pt',config(seed=99),{'model':'tiny','source':'unit'})

    def test_equal_exposure_and_update_matched_arm(self):
        a=MixtralAdapter(tiny());p=populations(a)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            one=run(a,p,root/'one',config(target_exposures=18,gate_target_exposures=15))
            larger=copy.deepcopy(p)
            extra=copy.deepcopy(larger['train'][0]);extra.document_id='train-extra'
            larger['train'].append(extra)
            two=run(a,larger,root/'two',config(target_exposures=18,gate_target_exposures=15))
            self.assertNotEqual(one['unique_adapter_positions'],two['unique_adapter_positions'])
            self.assertEqual(one['adapter_exposures'],two['adapter_exposures'])
            self.assertEqual(one['adapter_updates'],two['adapter_updates'])
            self.assertEqual(one['checkpoint_exposures'],two['checkpoint_exposures'])
            self.assertEqual(one['gate_updates'],two['gate_updates'])
            self.assertEqual(one['gate_exposures_per_candidate'],15)

    def test_cached_evaluation_never_repeats_completed_docs(self):
        a=MixtralAdapter(tiny());p=populations(a);b=ScaleBranch(16,config())
        with tempfile.TemporaryDirectory() as td:
            first=evaluate(a,p['dev'],b,config=config(),output=td)
            with patch.object(a.moe.experts,'forward',side_effect=AssertionError('completed computation repeated')):
                second=evaluate(a,p['dev'],b,config=config(),output=td)
            self.assertEqual(first,second)
            path=Path(td)/'document-000000.json';data=json.loads(path.read_text());data['evidence']['candidate_ce'][0]+=1;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):evaluate(a,p['dev'],b,config=config(),output=td)

    def test_final_test_guard_native_mask_and_paired_uncertainty(self):
        a=MixtralAdapter(tiny());p=populations(a)
        with self.assertRaises(ValueError):evaluate(a,p['dev'],role='final_test',config=config())
        result=evaluate(a,p['dev'],role='final_test',mode='native',config=config(),allow_final_test=True)
        self.assertEqual(result['continued_tokens'],0)
        comparison=paired_comparison(result,result,draws=20)
        self.assertEqual(comparison['gain_nats'],0);self.assertEqual(comparison['ci95'],[0,0])
        changed=copy.deepcopy(result);changed['per_document'][0]['document_id']='bad'
        with self.assertRaises(ValueError):paired_comparison(changed,result)

    def test_actual_backend_preflight(self):
        a=MixtralAdapter(tiny());c=populations(a)['train'][0]
        result=preflight(a,c,config())
        self.assertTrue(result['passed']);self.assertGreater(result['up_gradient_norm'],0)
        self.assertTrue(result['stop_zero_experts']);self.assertTrue(result['backbone_gradients_absent'])
        self.assertEqual(result['optimizer_steps'],1)
        self.assertGreater(result['effective_expert_input_changed_elements'],0)
        self.assertTrue(result['backbone_parameter_versions_unchanged'])
        self.assertTrue(result['conditional_parity']['passed'])

    def test_unknown_attempt_and_immutable_budget_origin(self):
        a=MixtralAdapter(tiny());p=populations(a)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run'
            with self.assertRaises(InterruptedError):run(a,p,root,interrupt_after_updates=1)
            path=root/'ledger.json';ledger=json.loads(path.read_text());ledger['events'].append(dict(key='lost',status='attempted'))
            original=ledger['started_utc'];path.write_text(json.dumps(ledger))
            report=run(a,p,root,resume=True)
            self.assertEqual(report['cost']['unknown_operations'],1)
            self.assertEqual(json.loads(path.read_text())['started_utc'],original)


    def test_gate_resume_does_not_repeat_committed_update(self):
        torch.manual_seed(31);a=MixtralAdapter(tiny());p=populations(a)
        original=RunLedger.operation;fired=[]
        @contextmanager
        def interrupted(ledger,key,**kwargs):
            with original(ledger,key,**kwargs) as event:
                yield event
            if key.startswith('gate-') and not fired:
                fired.append(key)
                raise InterruptedError('controlled gate interruption')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run(a,p,root/'full')
            with patch.object(RunLedger,'operation',interrupted):
                with self.assertRaises(InterruptedError):run(a,p,root/'interrupted')
            report=run(a,p,root/'interrupted',resume=True)
            self.assertEqual(report['gate_updates'],4)
            for epoch in (1,2):
                _,left,_=load_candidate(root/f'full/gated-epoch{epoch}.pt',config(),{'model':'tiny','source':'unit'})
                _,right,_=load_candidate(root/f'interrupted/gated-epoch{epoch}.pt',config(),{'model':'tiny','source':'unit'})
                self.assertEqual(digest(left.state_dict()),digest(right.state_dict()))
            events=json.loads((root/'interrupted/ledger.json').read_text())['events']
            self.assertEqual(sum(e['key']==fired[0] for e in events),1)

    def test_suite_deadline_and_actual_attempt_cost(self):
        a=MixtralAdapter(tiny());p=populations(a)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(TimeoutError):
                run(a,p,root/'expired',config(absolute_deadline_utc=time.time()-1))
            report=run(a,p,root/'cost')
            cost=report['cost']['work']
            self.assertGreater(cost['head_forward_attempts'],0)
            self.assertEqual(cost['head_forward_attempts'],cost['head_forward_completed'])
            self.assertEqual(cost['optimizer_step_completed'],report['adapter_updates'])
            self.assertEqual(cost['gate_optimizer_completed'],report['gate_updates'])
            self.assertGreater(cost['extra_expert_token_calls'],report['adapter_exposures']*2)

    def test_bf16_optimizer_has_effective_expert_input_change(self):
        from mixtral_lab.loop_scale import _batch,_ce
        torch.manual_seed(77);a=MixtralAdapter(tiny(torch.bfloat16));c=populations(a)['train'][0]
        branch=ScaleBranch(16,config(),amplitude=.1)
        positions=c.valid_mask.nonzero().flatten();h1,target=_batch(a,c,positions,config())
        seen=[]
        hook=a.moe.experts.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
        try:
            opt=torch.optim.AdamW(branch.parameters(),lr=.02,weight_decay=0.)
            for _ in range(10):
                opt.zero_grad();_ce(a,h1,target,branch).mean().backward();opt.step()
        finally:hook.remove()
        self.assertGreater(int((seen[0]!=seen[-1]).sum()),0)
        self.assertTrue(all(p.grad is None for p in a.model.parameters()))


    def test_correlated_chunks_are_one_source_bootstrap_cluster(self):
        a=dict(per_document=[dict(document_id='chunk1',source_document_id='source-a',tokens=2,candidate_nll=1.),
                             dict(document_id='chunk2',source_document_id='source-a',tokens=3,candidate_nll=2.),
                             dict(document_id='chunk3',source_document_id='source-b',tokens=5,candidate_nll=3.)])
        b=copy.deepcopy(a)
        b['per_document'][0]['candidate_nll']+=1
        b['per_document'][1]['candidate_nll']+=2
        b['per_document'][2]['candidate_nll']+=3
        result=paired_comparison(a,b,draws=100)
        self.assertEqual(result['documents'],3)
        self.assertEqual(result['independent_source_documents'],2)
        self.assertAlmostEqual(result['gain_nats'],2.3)
        self.assertEqual(result['ci95'],[1.6,3.])


    def test_real_zero_gradient_backbone_head_is_a_failure(self):
        a=MixtralAdapter(tiny())
        with torch.no_grad():a.model.lm_head.weight.zero_()
        p=populations(a)
        self.assertFalse(preflight(a,p['train'][0],config())['passed'])
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'zero-gradient'
            with self.assertRaisesRegex(FloatingPointError,'zero adapter gradient'):
                run(a,p,out)
            failure=json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['committed_adapter_updates'],0)
            self.assertEqual(failure['work']['zero_gradient_updates'],1)
            self.assertGreater(failure['cost']['failed_operations'],0)


if __name__=='__main__':unittest.main()
