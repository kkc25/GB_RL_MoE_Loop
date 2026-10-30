"""Fit actual second expert FFN, freeze a hard gate, then evaluate and profile.

No downloads, final-test input, automatic deployment or GPU work on import.
Each invocation creates a new output directory; failures retain an audit report.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import subprocess
from pathlib import Path
import time
import torch
from experiments.mixtral_experiment import (atomic_json, code_identity, load_local_inputs,
    load_local_model, measure_calls, precision_scope, input_device)
from experiments.mixtral_fit_experiment import normalized_artifact_stats, runtime_versions
from mixtral_lab.document_protocol import validate_document_config
from mixtral_lab.recovery_protocol import ROLES, prepare_data, read_documents
from mixtral_lab.fit_protocol import finite_tree, seal_report
from mixtral_lab.protocol import canonical_hash, fresh_output, sha256_file, batch_row
from mixtral_lab import trainable_loop as core

SCHEMA = 'mixtral-trainable-loop-run-v1'
REPORT = 'mixtral-trainable-loop-report-v1'


class DurableWork(dict):
    """Persist reservation/completion counters; ledger I/O is charged to fit/eval.

    Profiling uses a pre-reserved maximum invocation ledger, not timed I/O.
    A hard kill leaves status running and incomplete counters, never success.
    """
    def __init__(self,value,path):
        super().__init__(value);self.path=path;atomic_json(path,self)
    def __setitem__(self,key,value):
        super().__setitem__(key,value);atomic_json(self.path,self)


def memory_peaks(devices):
    result={}
    for device in devices:
        try:
            result[str(device)]={'allocated':torch.cuda.max_memory_allocated(device),'reserved':torch.cuda.max_memory_reserved(device)}
        except Exception as exc:
            result[str(device)]={'unavailable':type(exc).__name__}
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=('fit','evaluate','profile'),required=True)
    p.add_argument('--output-dir')
    p.add_argument('--expected-commit',help='required full immutable Git commit SHA for formal execution')
    p.add_argument('--fit-report')
    p.add_argument('--evaluation-report')
    return p


def contract(config):
    return {k:v for k,v in config.items() if k!='output_dir'}


def validate_config(raw):
    finite_tree(raw,'config'); config=deepcopy(raw)
    if config.get('schema')!=SCHEMA: raise ValueError('requires independent true-loop schema')
    if any(k in config for k in ('document','recovery','training','fit','oracle','second_round')):
        raise ValueError('legacy experiment settings cannot be relabeled')
    loop=core.LoopConfig(**config['loop']);loop.validate()
    if loop.seed!=config['seed']: raise ValueError('runner and loop seeds differ')
    config['loop']=json.loads(json.dumps(asdict(loop)))
    for key in ('total_wall_seconds','selection_min_gain'):
        x=config[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:
            raise ValueError('finite positive '+key+' required')
    if config['total_wall_seconds']>7200: raise ValueError('total chain wall budget exceeds two hours')
    # Compatibility view validates existing model/data/runtime safety bounds only.
    view=deepcopy(config);view['schema']='mixtral-full-document-recovery-v1'
    view['document']={'kind':'adapter','training':{'seed':config['seed']},
                      'development_min_gain':.001,'selection_min_gain':.001}
    validate_document_config(view)
    return config


def source_identity():
    value=code_identity(); root=Path(__file__).resolve().parents[1]
    for name in ('experiments/mixtral_trainable_loop.py','experiments/mixtral_fit_experiment.py'):
        value['sources'][name]=sha256_file(root/name)
    return value


def read_report(path,stage):
    p=Path(path)
    if not p.is_file() or p.stat().st_size>64*1024**2: raise ValueError('missing or oversized report')
    r=json.loads(p.read_text());finite_tree(r,'report')
    if r.get('schema')!=REPORT or r.get('stage')!=stage or r.get('status')!='complete':
        raise ValueError('complete true-loop stage report required')
    if r.get('report_sha256')!=seal_report(r)['report_sha256']: raise ValueError('report digest mismatch')
    if r.get('config')!=validate_config(r['config']): raise ValueError('noncanonical config')
    elapsed=r.get('wall_seconds');prior=r.get('prior_wall_seconds');total=r.get('cumulative_wall_seconds')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in (elapsed,prior,total)):
        raise ValueError('invalid wall-time receipt')
    if not math.isclose(total,prior+elapsed,rel_tol=0,abs_tol=.01) or total>r['config']['total_wall_seconds']:
        raise ValueError('inconsistent or over-budget complete receipt')
    return r


def checkpoint_record(path):
    return {'path':str(Path(path).resolve()),'sha256':sha256_file(path)}


def check_files(report):
    for value in [*report['checkpoints'].values(),*report.get('evidence_artifacts',{}).values()]:
        if sha256_file(value['path'])!=value['sha256']: raise ValueError('frozen checkpoint changed')


def require_parity(value):
    rows=value.get('documents',[])
    if not rows or value.get('status')!='complete': raise ValueError('missing parity evidence')
    passed=all(m['elements']>0 and m['mismatched_elements']==0 for r in rows for m in r['comparisons'].values())
    if not passed or value.get('passed') is not True: raise ValueError('numerical parity failed')


def downstream(args,config,current):
    fit=read_report(args.fit_report,'fit')
    if contract(fit['config'])!=contract(config) or fit['current']!=current:
        raise ValueError('frozen source/config/data/runtime changed')
    check_files(fit)
    if not fit['eligible'] or fit['choice']['mode']!='gate':
        raise ValueError('fit did not select an eligible non-STOP gate; no outer evaluation')
    for name in ('initial','selected'):
        evidence=fit['parity'][name];require_parity(evidence)
        if [r['document_id'] for r in evidence['documents']]!=fit['identity']['train']['selected_document_ids']:
            raise ValueError('replay population changed')
        required={'h0','native_hidden','native_ce','candidate_hidden','candidate_ce'}
        if any(set(r['comparisons'])!=required for r in evidence['documents']):
            raise ValueError('replay comparisons missing')
    evaluation=None
    if args.stage=='profile':
        evaluation=read_report(args.evaluation_report,'evaluate')
        check_files(evaluation)
        if evaluation['checkpoints']!=fit['checkpoints']: raise ValueError('evaluation frozen checkpoints differ from fit')
        if evaluation['fit_report_sha256']!=fit['report_sha256'] or evaluation['identity']!=fit['identity']:
            raise ValueError('evaluation is not bound to frozen fit')
        if evaluation['current']!=current or contract(evaluation['config'])!=contract(config):
            raise ValueError('profile evidence identity changed')
        computed=quality(evaluation['results'],config['selection_min_gain'],config['seed'])
        if evaluation['quality']!=computed or not computed['passed']:
            raise ValueError('quality gate failed or inconsistent; profile forbidden')
    return fit,evaluation


def development(adapter,rows,config,work):
    def evaluate(branch):
        return core.evaluate(adapter,rows,branch,role='development',work=work,config=config)
    evaluate.contract={'role':'development','document_ids':[r['id'] for r in rows],'protocol':'full_document_online_nll'}
    return evaluate


def choose_threshold(adapter,rows,branch,gate,config,work):
    stop=core.evaluate(adapter,rows,branch,role='development',gate=gate,mode='stop',work=work,config=config)
    choice={'mode':'stop','threshold':None,'development':stop}; candidates=[]
    for threshold in config.thresholds:
        result=core.evaluate(adapter,rows,branch,role='development',gate=gate,threshold=threshold,mode='gate',work=work,config=config)
        candidates.append({'threshold':threshold,'development':result})
        if result['continued_supervised_tokens']>0 and result['nll']<choice['development']['nll']:
            choice={'mode':'gate','threshold':threshold,'development':result}
    return choice,candidates


def quality(results,minimum,seed):
    """Paired document bootstrap; historical outer development, not final test."""
    candidate=results['gate']; comparisons={}
    for name in ('stop','matched_random','parameter_control'):
        other=results[name]
        a=candidate['per_document']; b=other['per_document']
        if [(r['document_id'],r['tokens']) for r in a]!=[(r['document_id'],r['tokens']) for r in b]:
            raise ValueError('unpaired quality comparison')
        gain=torch.tensor([s['candidate_nll']-r['candidate_nll'] for r,s in zip(a,b)],dtype=torch.float64)
        weights=torch.tensor([r['tokens'] for r in a],dtype=torch.float64)
        gen=torch.Generator().manual_seed(seed); idx=torch.randint(len(a),(2000,len(a)),generator=gen)
        boots=(gain[idx]*weights[idx]).sum(1)/weights[idx].sum(1)
        comparisons[name]={'gain_nats':float((gain*weights).sum()/weights.sum()),
            'paired_document_ci95':[float(boots.quantile(.025)),float(boots.quantile(.975))]}
    passed=candidate['continued_supervised_tokens']>0 and len(candidate['per_document'])>=2 and all(
        comparisons[k]['gain_nats']>=minimum and comparisons[k]['paired_document_ci95'][0]>0 for k in ('stop','matched_random'))
    return {'passed':passed,'conditional_allocation_supported':passed,
            'loop_specific_advantage_supported':passed and comparisons['parameter_control']['gain_nats']>=minimum and comparisons['parameter_control']['paired_document_ci95'][0]>0,
            'minimum_gain':minimum,'comparisons':comparisons,
            'scope':'historically inspected outer development; parameter control reported separately, not compute matched'}


def export_token_evidence(report,output):
    """Bounded readable audit rows; sufficient to recompute CE gains/decisions."""
    sources=[('report',report)]
    for kind in ('r2','parameter_control'):
        history=output/kind/'history.json'
        if history.is_file(): sources.append((kind+'_history',json.loads(history.read_text())))
    records=[]
    def visit(value,path):
        if isinstance(value,dict):
            if 'per_document' in value:
                for row in value['per_document']:
                    required=('valid_position_indices','native_ce','candidate_ce','continue_mask')
                    if any(k not in row or len(row[k])!=row['tokens'] for k in required):
                        raise ValueError('missing bounded per-token evidence')
                    records.append({'source':path,'record':row})
            for key,item in value.items():
                if key!='per_document': visit(item,path+'/'+str(key))
        elif isinstance(value,list):
            for i,item in enumerate(value): visit(item,path+'/'+str(i))
    for label,value in sources:visit(value,label)
    if not records:return
    path=output/'token_evidence.jsonl';total=0
    with path.open('x') as handle:
        for record in records:
            text=json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n';total+=len(text.encode())
            if total>64*1024**2: raise ValueError('token evidence exceeds 64 MiB safety bound')
            handle.write(text)
    report.setdefault('evidence_artifacts',{})['tokens']={**checkpoint_record(path),'records':len(records),'bytes':total}


def phase(report,output,name):
    report['phase']=name
    report.setdefault('phase_started_monotonic',{})[name]=time.monotonic()
    if 'work' in report:report['work']['phase']=name
    atomic_json(output/'report.json',report)


def fit_stage(adapter,data,config,identity,output,work,report):
    c=core.LoopConfig(**config['loop'])
    caches=core.cache_documents(adapter,data['train'],max_cache_bytes=c.max_cache_bytes,max_logit_elements=c.max_logit_elements,work=work)
    evaluator=development(adapter,data['development'],c,work)
    report['backend_backward']=core.verify_backend_backward(adapter,caches[0],c,work)
    initial=core.LoopBranch(adapter.hidden_size,c,'r2',adapter.device)
    report['parity']={'initial':core.verify_replay(adapter,caches,initial,c,work)}
    require_parity(report['parity']['initial'])
    phase(report,output,'fit_r2')
    branch,report['r2_fit']=core.fit_branch(adapter,caches,evaluator,c,identity,output/'r2',kind='r2',work=work)
    phase(report,output,'fit_parameter_control')
    control,report['control_fit']=core.fit_branch(adapter,caches,evaluator,c,identity,output/'parameter_control',kind='parameter_control',work=work)
    report['parity']['selected']=core.verify_replay(adapter,caches,branch,c,work)
    require_parity(report['parity']['selected'])
    report['checkpoints']={'r2':checkpoint_record(output/'r2'/'checkpoint_best.pt'),
                           'parameter_control':checkpoint_record(output/'parameter_control'/'checkpoint_best.pt')}
    report['eligible']=report['r2_fit']['selected_train_probe']['gate_eligible']
    report['choice']={'mode':'stop','threshold':None,'reason':'train eligibility failed'}
    if report['eligible']:
        phase(report,output,'fit_gate')
        gate,report['gate_fit']=core.fit_gate(adapter,caches,branch,c,work)
        report['choice'],report['threshold_candidates']=choose_threshold(adapter,data['development'],branch,gate,c,work)
        if report['choice']['mode']=='gate':
            report['parity']['conditional']=core.verify_conditional(adapter,caches,branch,gate,report['choice']['threshold'],c,work=work)
            report['parity']['conditional']['gate_scope']='diagnostic only; deployed compact path is selected on actual development NLL'
            p=output/'frozen.pt'
            core.save_checkpoint(p,branch,identity,report['r2_fit']['best_epoch'],gate=gate,
                threshold=report['choice']['threshold'],selection=report['choice'])
            report['checkpoints']['frozen']=checkpoint_record(p)
    report['scientific_outcome']=('train_branch_ineligible' if not report['eligible'] else
        'no_gate_selected' if report['choice']['mode']=='stop' else 'frozen_gate_pending_outer_evaluation')
    return report


def main(argv=None):
    args=parser().parse_args(argv);raw=json.loads(Path(args.config).read_text())
    if args.output_dir: raw['output_dir']=args.output_dir
    destination=Path(raw['output_dir']).resolve()
    for name in ('model_path','tokenizer_path'):
        source=raw.get(name)
        if isinstance(source,str) and source.strip():
            immutable=Path(source).resolve()
            if destination==immutable or immutable in destination.parents:
                raise ValueError('output must not write inside immutable model/tokenizer source')
    output=fresh_output(raw['output_dir']);started=time.perf_counter()
    report={'schema':REPORT,'stage':args.stage,'status':'running','phase':'preflight','work':{},
        'limitations':['Actual GPU results are pending until collaborator execution.',
          'Selection is historically inspected outer development, not final heldout.',
          'No amplitude gate or final-test execution in this protocol.']}
    atomic_json(output/'report.json',report)
    try:
        if args.stage=='fit' and (args.fit_report or args.evaluation_report): raise ValueError('fit must start fresh')
        if args.stage!='fit' and not args.fit_report: raise ValueError('frozen fit report required')
        if args.stage=='profile' and not args.evaluation_report: raise ValueError('passing evaluation report required')
        if args.stage=='evaluate' and args.evaluation_report: raise ValueError('cannot reselect from outer evidence')
        config=validate_config(raw); report['config']=config
        work=report['work'];work['_deadline_monotonic']=time.monotonic()+config['total_wall_seconds']
        read_documents(config);inputs=load_local_inputs(config);data,data_identity=prepare_data(config,inputs['tokenizer'])
        current={'code':source_identity(),'data':data_identity,'local_inputs':inputs['identity'],
                 'artifacts':normalized_artifact_stats(config),'runtime':runtime_versions()}
        report['current']=current
        if (not isinstance(args.expected_commit,str) or len(args.expected_commit)!=40
                or any(x not in '0123456789abcdef' for x in args.expected_commit)
                or current['code']['commit']!=args.expected_commit):
            raise ValueError('formal execution requires matching full --expected-commit')
        root=Path(__file__).resolve().parents[1]
        subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--','mixtral_lab','experiments'],check=True,capture_output=True)
        for name in ('model_path','tokenizer_path'):
            if Path(config[name]).resolve() in output.resolve().parents:
                raise ValueError('output must not write inside immutable model/tokenizer source')
        fit=evaluation=None
        if args.stage!='fit': fit,evaluation=downstream(args,config,current)
        prior=0. if fit is None else (evaluation or fit)['cumulative_wall_seconds']
        if not isinstance(prior,(int,float)) or not math.isfinite(prior) or prior<0:
            raise ValueError('invalid cumulative cost receipt')
        report['prior_wall_seconds']=prior
        work['_deadline_monotonic']=time.monotonic()+config['total_wall_seconds']-prior-(time.perf_counter()-started)
        core.check_deadline(work)
        if args.stage!='profile':
            work=DurableWork(work,output/'work_ledger.json');report['work']=work
        else:
            work['profile_max_variant_invocations']=5*(config['warmup']+config['repeats']*config['measurements'])
            work['profile_reservation_scope']='upper bound, not completed calls; interrupted timing is inconclusive'
            atomic_json(output/'work_ledger.json',work)
        c=core.LoopConfig(**config['loop'])
        if len(data['train'])*config['sequence_length']*(inputs['metadata'].hidden_size*4+32)>c.max_cache_bytes:
            raise ValueError('cache quota exceeded before model load')
        if config['sequence_length']*inputs['metadata'].vocab_size>c.max_logit_elements:
            raise ValueError('logit quota exceeded before model load')
        if not torch.cuda.is_available(): raise RuntimeError('formal Mixtral stages require CUDA; tiny real HF CPU tests are correctness tests only')
        with precision_scope(config['allow_tf32']):
            report['phase']='model_load';atomic_json(output/'report.json',report)
            model,_,adapter,model_info=load_local_model(config,local_inputs=inputs)
            identity={'model':model_info['model'],'tokenizer':model_info['tokenizer'],**data_identity,
                'code':current['code'],'contract':contract(config),'runtime':current['runtime'],
                'device_map':model_info.get('device_map'),'cuda_devices':model_info.get('cuda_devices')}
            report.update(identity=identity,model=model_info,model_load_peak_memory_bytes=memory_peaks(model_info['cuda_devices']))
            for device in model_info['cuda_devices']: torch.cuda.reset_peak_memory_stats(device)
            if fit is not None and fit['identity']!=identity: raise ValueError('loaded execution identity differs')
            if args.stage=='fit': fit_stage(adapter,data,config,identity,output,work,report)
            else:
                report['fit_report_sha256']=fit['report_sha256'];report['checkpoints']=deepcopy(fit['checkpoints'])
                branch,gate,metadata=core.load_checkpoint(fit['checkpoints']['frozen']['path'],identity,c,adapter.device)
                if metadata['selection']!=fit['choice'] or metadata['threshold']!=fit['choice']['threshold'] or metadata['epoch']!=fit['r2_fit']['best_epoch']:
                    raise ValueError('checkpoint selection mismatch')
                control,_,control_metadata=core.load_checkpoint(fit['checkpoints']['parameter_control']['path'],identity,c,adapter.device)
                if control_metadata['epoch']!=fit['control_fit']['best_epoch']: raise ValueError('control checkpoint is not frozen development best')
                report['phase']=args.stage
                if args.stage=='evaluate':
                    results={}
                    for mode in ('stop','forced','gate','matched_random','fixed_hash_25','fixed_hash_50','fixed_hash_100'):
                        results[mode]=core.evaluate(adapter,data['selection'],branch,role='selection',gate=gate,
                            threshold=metadata['threshold'],mode=mode,work=work,config=c)
                    results['parameter_control']=core.evaluate(adapter,data['selection'],control,role='selection',work=work,config=c)
                    report['results']=results;report['quality']=quality(results,config['selection_min_gain'],config['seed'])
                else:
                    report['evaluation_report_sha256']=evaluation['report_sha256'];report['quality']=evaluation['quality']
                    @torch.no_grad()
                    def native_model(row):
                        core.check_deadline(work)
                        work['native_model_forward_attempts']=work.get('native_model_forward_attempts',0)+1
                        logits=model(**batch_row(row,input_device(model)),use_cache=False,output_router_logits=False).logits
                        work['native_model_forward_completed']=work.get('native_model_forward_completed',0)+1
                        return logits
                    functions={'native_model':native_model}
                    for name,b,g,t,mode in [('controlled_stop',None,None,None,'stop'),('forced_r2',branch,None,None,'forced'),
                            ('learned_gate',branch,gate,metadata['threshold'],'gate'),('parameter_control',control,None,None,'forced')]:
                        functions[name]=lambda row,b=b,g=g,t=t,mode=mode: core.online_logits(adapter,row,b,gate=g,threshold=t,mode=mode,config=c,work=work)
                    report['timing']=measure_calls(functions,data['selection'],model_info['cuda_devices'],config)
            export_token_evidence(report,output)
            core.check_deadline(work)
            if source_identity()!=current['code'] or normalized_artifact_stats(config)!=current['artifacts']:
                raise ValueError('source or model artifacts changed during stage')
            for role in ROLES:
                if sha256_file(config['data_files'][role])!=data_identity[role]['file_sha256']: raise ValueError('data changed during stage')
            check_files(report)
            core.check_deadline(work)
        report.update(status='complete',phase='complete',wall_seconds=time.perf_counter()-started,
            cumulative_wall_seconds=report.get('prior_wall_seconds',0.)+time.perf_counter()-started,
            peak_memory_bytes=memory_peaks(report.get('model',{}).get('cuda_devices',[])))
        report=seal_report(report);atomic_json(output/'report.json',report)
    except Exception as exc:
        partial={}
        for p in output.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl') and p.name!='report.json' and p.stat().st_size<=64*1024**2:
                partial[str(p.relative_to(output))]=checkpoint_record(p)
        report['partial_artifacts']=partial
        report.update(status='failed',failure_kind='budget_exhausted' if isinstance(exc,TimeoutError) else 'execution_error',
            failure_phase=report['phase'],error=f'{type(exc).__name__}: {exc}',wall_seconds=time.perf_counter()-started,
            cumulative_wall_seconds=report.get('prior_wall_seconds',0.)+time.perf_counter()-started,
            peak_memory_bytes=memory_peaks(report.get('model',{}).get('cuda_devices',[])))
        atomic_json(output/'report.json',report);raise
    print(json.dumps({'report':str(output/'report.json'),'status':report['status']}));return report

if __name__=='__main__': main()
