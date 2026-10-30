"""Real tiny HF expert FFNs: no downloaded models or GPU claims."""
from dataclasses import replace
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.document_training import cache_documents
from mixtral_lab.trainable_loop import (LoopConfig,LoopBranch,GainGate,losses,evaluate,fit_branch,
    fit_gate,probe,save_checkpoint,load_checkpoint,verify_conditional,verify_replay,verify_backend_backward)
from tests.test_mixtral_adapter import tiny
from tests.test_mixtral_bridge import rows


def cfg(**kw): return replace(LoopConfig(rank=2,epochs=2,positions_per_document=2,max_updates=20,gate_updates=4,gate_batch=4),**kw)
def ids(): return dict(model={'id':'tiny'},tokenizer={'id':'tiny'},code={'sha':'unit'},
    train={'selected_document_ids':['train-a','train-b']},development={'selected_document_ids':['dev']})
def dev(scores):
    it=iter(scores)
    def f(branch): return dict(status='ok',role='development',document_ids=['dev'],protocol='full_document_online_nll',nll=next(it))
    f.contract=dict(role='development',document_ids=['dev'],protocol='full_document_online_nll');return f

class TrainableLoopTests(unittest.TestCase):
    def test_real_ffn_zero_adapter_not_native_and_gradient(self):
        for dtype in (torch.float32,torch.bfloat16):
            a=MixtralAdapter(tiny(dtype));c=cache_documents(a,rows())[0];m=LoopBranch(16,cfg(),device=a.device)
            before={k:v.clone() for k,v in a.model.state_dict().items()};calls=[]
            handle=a.moe.experts.register_forward_pre_hook(lambda _,args:calls.append(args[0].shape))
            try: ce,h=losses(a,c,m);ce[c.valid_mask].mean().backward()
            finally: handle.remove()
            self.assertEqual(calls,[torch.Size([6,16])]);self.assertFalse(torch.equal(h,c.native_h1))
            self.assertGreater(float(m.correction.up.grad.abs().sum()),0.)
            self.assertTrue(all(p.grad is None for p in a.model.parameters()))
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in a.model.state_dict().items()))
            torch.testing.assert_close(h,a.step_pairs(c.native_h1,a.native_pair(c.native_h1)),rtol=0,atol=0)
    def test_stop_zero_extra_calls_and_mixed_compaction(self):
        a=MixtralAdapter(tiny());c=cache_documents(a,rows())[0];m=LoopBranch(16,cfg());calls=[]
        handle=a.moe.experts.register_forward_pre_hook(lambda _,args:calls.append(len(args[0])))
        try:
            mask=torch.zeros(6,dtype=torch.bool);ce,h=losses(a,c,m,mask=mask)
            self.assertEqual(calls,[]);self.assertTrue(torch.equal(h,c.native_h1))
            mask[[1,3]]=True;ce,h=losses(a,c,m,mask=mask);self.assertEqual(calls,[2])
            self.assertTrue(torch.equal(h[0,[0,2,4,5]],c.native_h1[0,[0,2,4,5]]))
            ce[1:2].mean().backward();self.assertGreater(float(m.correction.up.grad.abs().sum()),0.)
        finally:handle.remove()
    def test_parameter_matched_control_no_extra_ffn(self):
        a=MixtralAdapter(tiny());c=cache_documents(a,rows())[0]
        r=LoopBranch(16,cfg());m=LoopBranch(16,cfg(),'parameter_control')
        self.assertEqual(sum(p.numel() for p in r.parameters()),sum(p.numel() for p in m.parameters()))
        with patch.object(a.moe.experts,'forward',side_effect=AssertionError('control must not use experts')):
            ce,h=losses(a,c,m);self.assertTrue(torch.equal(h,c.native_h1));ce[c.valid_mask].mean().backward()
        self.assertGreater(float(m.correction.up.grad.abs().sum()),0.)
    def test_fixed_budget_complete_coverage_no_dev_patience(self):
        a=MixtralAdapter(tiny());c=cache_documents(a,rows())
        with tempfile.TemporaryDirectory() as t:
            m,s=fit_branch(a,c,dev([1.,2.,3.]),cfg(),ids(),Path(t)/'fit')
            self.assertEqual(s['epochs_completed'],2);self.assertEqual(s['updates'],6)
            self.assertEqual(s['context_visits'],10);self.assertEqual(s['min_visits'],2);self.assertEqual(s['best_epoch'],0)
            self.assertEqual(s['stop_reason'],'fixed_epoch_budget_completed')
            b,g,meta=load_checkpoint(Path(t)/'fit/checkpoint_best.pt',ids(),cfg())
            self.assertTrue(torch.equal(m.correction.up,b.correction.up));self.assertIsNone(g)
            with self.assertRaises(FileExistsError):fit_branch(a,c,dev([1.,2.,3.]),cfg(),ids(),Path(t)/'fit')
    def test_budget_and_identity_fail_before_output(self):
        a=MixtralAdapter(tiny());c=cache_documents(a,rows())
        with tempfile.TemporaryDirectory() as t:
            for configuration in (cfg(max_updates=1),cfg(max_cache_bytes=1),cfg(max_train_tokens=1)):
                with self.assertRaises(ValueError):fit_branch(a,c,dev([1.,2.,3.]),configuration,ids(),Path(t)/'bad')
                self.assertFalse((Path(t)/'bad').exists())
    def test_gate_label_free_and_frozen_branch(self):
        a=MixtralAdapter(tiny());c=cache_documents(a,rows());m=LoopBranch(16,cfg());before=copy.deepcopy(m.state_dict())
        g,s=fit_gate(a,c,m,cfg());self.assertEqual(s['updates'],4);self.assertEqual(s['tokens'],5)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()))
        with patch.object(a.moe.experts,'forward',side_effect=AssertionError('gate must not execute experts')):
            self.assertEqual(g(c[0].native_h1).shape,(1,6))
    def test_conditional_shape_replay_and_fresh_full(self):
        a=MixtralAdapter(tiny());c=cache_documents(a,rows());m=LoopBranch(16,cfg());g=GainGate(16)
        with torch.no_grad():m.correction.up.fill_(.02);g.weight[0]=1
        self.assertTrue(verify_replay(a,c,m,cfg())['passed'])
        self.assertTrue(verify_conditional(a,c,m,g,0.,cfg())['passed'])
    def test_online_controls_and_exact_count_random(self):
        a=MixtralAdapter(tiny());m=LoopBranch(16,cfg());g=GainGate(16)
        with torch.no_grad():g.weight[0]=1.
        actual=evaluate(a,rows(),m,gate=g,threshold=0.,mode='gate')
        matched=evaluate(a,rows(),m,gate=g,threshold=0.,mode='matched_random')
        self.assertEqual([r['continued_tokens'] for r in actual['per_document']],[r['continued_tokens'] for r in matched['per_document']])
        stop=evaluate(a,rows(),m,mode='stop');self.assertEqual(stop['nll'],stop['native_nll']);self.assertEqual(stop['continued_tokens'],0)
        fixed=evaluate(a,rows(),m,mode='fixed_hash_25');self.assertEqual(fixed,evaluate(a,rows(),m,mode='fixed_hash_25'))
        with self.assertRaises(ValueError):evaluate(a,rows(),m,role='newtest')
    def test_checkpoint_tamper_and_deadline(self):
        a=MixtralAdapter(tiny());c=cache_documents(a,rows());m=LoopBranch(16,cfg());g=GainGate(16)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'c.pt';save_checkpoint(p,m,ids(),0,gate=g,threshold=.01)
            _,loaded,data=load_checkpoint(p,ids(),cfg());self.assertIsNotNone(loaded)
            changed=ids();changed['code']['sha']='bad'
            with self.assertRaises(ValueError):load_checkpoint(p,changed,cfg())
            data['state']['correction.up'].add_(1);torch.save(data,p)
            with self.assertRaises(ValueError):load_checkpoint(p,ids(),cfg())
        with self.assertRaises(TimeoutError):losses(a,c[0],m,work={'_deadline_monotonic':0})
    def test_bf16_training_changes_expert_input_and_hidden(self):
        a=MixtralAdapter(tiny(torch.bfloat16));c=cache_documents(a,rows())[0];m=LoopBranch(16,cfg())
        with torch.no_grad(): initial=m(a,c.native_h1).clone()
        opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=0.)
        for _ in range(20):
            opt.zero_grad();ce,_=losses(a,c,m);ce[c.valid_mask].mean().backward();opt.step()
        with torch.no_grad(): latest=m(a,c.native_h1)
        self.assertGreater(int((initial!=latest).sum()),0)
        self.assertTrue(all(p.grad is None for p in a.model.parameters()))
    def test_checkpoint_metadata_and_nonfinite_rejected(self):
        import hashlib,json
        a=MixtralAdapter(tiny());m=LoopBranch(16,cfg());g=GainGate(16)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'c.pt';save_checkpoint(p,m,ids(),0,gate=g,threshold=.01)
            original=torch.load(p,weights_only=True)
            for key,value in [('threshold',.3),('epoch',99),('kind','not-r2')]:
                data=copy.deepcopy(original);data[key]=value;torch.save(data,p)
                with self.assertRaises(ValueError):load_checkpoint(p,ids(),cfg())
            data=copy.deepcopy(original);data['state']['correction.up'].fill_(float('nan'))
            from mixtral_lab.adapter_fit import _tensor_hash
            data['state_sha256']=_tensor_hash(data['state'])
            data['metadata_sha256']=hashlib.sha256(json.dumps({k:v for k,v in data.items() if k not in ('state','gate','metadata_sha256')},sort_keys=True,allow_nan=False).encode()).hexdigest()
            torch.save(data,p)
            with self.assertRaises(ValueError):load_checkpoint(p,ids(),cfg())
    def test_actual_backend_backward_preflight(self):
        a=MixtralAdapter(tiny(torch.bfloat16));c=cache_documents(a,rows())[0];work={}
        result=verify_backend_backward(a,c,cfg(),work)
        self.assertTrue(result['passed']);self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(work['preflight_backward_completed'],1)

    def test_raw_metrics_recompute_and_template_identity(self):
        import json
        a=MixtralAdapter(tiny());m=LoopBranch(16,cfg());g=GainGate(16)
        with torch.no_grad():g.weight[0]=1
        result=evaluate(a,rows(),m,gate=g,threshold=0.,mode='gate')
        total=0.;tokens=0
        for r in result['per_document']:
            self.assertEqual(len(r['candidate_ce']),r['tokens'])
            self.assertAlmostEqual(sum(r['candidate_ce'])/r['tokens'],r['candidate_nll'])
            self.assertEqual(sum(r['continue_mask']),r['continued_supervised_tokens'])
            self.assertEqual(sum(r['padded_continue_mask']),r['continued_tokens'])
            self.assertEqual([x>0 for x in r['gate_predictions']],r['continue_mask'])
            total+=sum(r['candidate_ce']);tokens+=r['tokens']
        self.assertAlmostEqual(total/tokens,result['nll'])
        root=Path(__file__).resolve().parents[1]
        local=json.loads((root/'local_configs/trainable_loop_pilot.json').read_text())
        generic=json.loads((root/'configs/mixtral/trainable_loop_pilot.json').read_text())
        for key in ('model_path','tokenizer_path','data_files','output_dir'):local.pop(key);generic.pop(key)
        self.assertEqual(local,generic)
        from experiments.mixtral_trainable_loop import validate_config
        validate_config(json.loads((root/'configs/mixtral/trainable_loop_pilot.json').read_text()))
        validate_config(json.loads((root/'local_configs/trainable_loop_pilot.json').read_text()))

    def test_invalid_config(self):
        for configuration in (cfg(epochs=0),cfg(learning_rate=float('nan')),cfg(thresholds=(.1,.0)),cfg(eligibility_top_fraction=0)):
            with self.assertRaises(ValueError):configuration.validate()

if __name__=='__main__':unittest.main()
