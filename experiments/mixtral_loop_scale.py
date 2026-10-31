"""Offline preparation and explicitly staged Mixtral loop learning curves.

No work happens on import. Preparation hashes local bytes; GPU stages require a
clean, pinned checkout and never download weights or access a remote server.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
import subprocess
import time

from mixtral_lab.loop_scale_data import (DiskDocumentCache, digest, load_manifest,
    local_artifact_identity, prepare_manifests, sha256_file, _atomic_json,
    _validate_index)

SCHEMA = 'mixtral-loop-scale-run-v2'
ROOT = Path(__file__).resolve().parents[1]
REVISION = 'fc7ac94680e38d7348cfa806e51218e6273104b0'
CELLS = {'r2_full': ('r2', 1.), 'r2_reduced': ('r2', .1),
         'parameter_control': ('parameter_control', 1.)}


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_config(raw):
    """The scientific grid is deliberately small and fixed before preparation."""
    c = json.loads(json.dumps(raw, allow_nan=False))
    if c.get('schema') != SCHEMA: raise ValueError('independent v2 schema required')
    if c.get('model_revision') != REVISION: raise ValueError('pinned Mixtral revision required')
    for key in ('model_path', 'tokenizer_path', 'freeze_dir', 'run_root', 'historical_report'):
        if not isinstance(c.get(key), str) or not c[key].strip(): raise ValueError('explicit '+key+' required')
    if not isinstance(c.get('source_files'), list) or not c['source_files']:
        raise ValueError('explicit local JSONL source_files required; no download fallback')
    if any(not isinstance(p, str) or not p.strip() for p in c['source_files']): raise ValueError('invalid source path')
    if c.get('train_budgets') != [100000, 300000, 1000000]: raise ValueError('frozen nested training budgets required')
    if (c.get('development_budget'), c.get('test_budget'), c.get('sequence_length')) != (100000,100000,256):
        raise ValueError('frozen development/test/sequence allocation required')
    if c.get('seeds') != [17,29,43] or c.get('split_seed') != 17: raise ValueError('frozen seeds required')
    if c.get('amplitudes') != [1.,.1]: raise ValueError('only full and predeclared reduced amplitude')
    if c.get('dtype') != 'bfloat16' or c.get('experts_implementation') != 'eager':
        raise ValueError('BF16 eager protocol required')
    if c.get('device_map') != 'balanced' or c.get('max_memory') != {str(i):'44GiB' for i in range(4)}:
        raise ValueError('four resident A6000 placement budgets required')
    if c.get('r2_precision','native_bf16') not in ('native_bf16','fp32_experts','fp32_canonical32'):
        raise ValueError('unknown R2 precision protocol')
    if c.get('allow_tf32') is not False: raise ValueError('TF32 must remain disabled')
    for key in ('cell_wall_seconds', 'suite_wall_seconds', 'cache_wall_seconds', 'preflight_wall_seconds'):
        n=c.get(key)
        if type(n) is not int or not 1 <= n <= 172800: raise ValueError('bounded '+key+' required')
    if c['cell_wall_seconds'] > c['suite_wall_seconds']: raise ValueError('cell exceeds suite wall cap')
    for key in ('max_shard_bytes','max_document_bytes','max_cache_disk_bytes'):
        if type(c.get(key)) is not int or c[key] < 1: raise ValueError('bounded '+key+' required')
    if c['max_document_bytes'] > c['max_shard_bytes']: raise ValueError('document exceeds shard cap')
    if c.get('minimum_gain') != .001: raise ValueError('frozen .001 nat minimum gain required')
    if c.get('minimum_source_groups') != 30: raise ValueError('at least30 independent source groups per role required')
    if c.get('preflight_documents') != 4: raise ValueError('four-document preflight required')
    if c.get('require_gpu_name') != 'A6000': raise ValueError('declared A6000 resource protocol required')
    protected=[Path(c[k]).resolve() for k in ('model_path','tokenizer_path')]
    protected_files=[Path(p).resolve() for p in [*c['source_files'],c['historical_report']]]
    for key in ('run_root','freeze_dir'):
        out=Path(c[key]).resolve()
        if any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
            raise ValueError('outputs must be disjoint from local model/tokenizer artifacts')
        if any(out==p or p.is_relative_to(out) for p in protected_files):
            raise ValueError('outputs must not contain immutable input files')
    a,b=Path(c['run_root']).resolve(),Path(c['freeze_dir']).resolve()
    if a==b or a.is_relative_to(b) or b.is_relative_to(a): raise ValueError('run and frozen manifests require disjoint directories')
    return c


def source_identity(expected=None, *, clean=False):
    def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args]).decode().strip()
    commit=git('rev-parse','HEAD')
    if expected is not None and (not re.fullmatch('[0-9a-f]{40}', expected) or expected != commit):
        raise ValueError('full expected commit differs from checkout')
    if clean and (git('diff','HEAD','--') or git('ls-files','--others','--exclude-standard')):
        raise ValueError('formal execution requires clean committed checkout')
    paths=[*sorted((ROOT/'mixtral_lab').glob('*.py')), ROOT/'experiments/mixtral_loop_scale.py',
           ROOT/'experiments/mixtral_experiment.py']
    return dict(commit=commit,sources={str(p.relative_to(ROOT)):sha256_file(p) for p in paths})


def historical_exclusions(path):
    r=read_json(path); ids=set(); sources=set(); hashes=set()
    for role in ('train','development','selection','calibration'):
        item=r['identity'][role]
        ids.update(item['selected_document_ids'])
        sources.update(item['selected_document_ids'])
        for d in item['documents']:
            ids.add(d['id']); hashes.add(d['text_sha256'])
            sources.add(d['id'])
            if 'source_id' in d: sources.add(d['source_id'])
    return dict(excluded_document_ids=sorted(ids),excluded_source_ids=sorted(sources),
                excluded_content_hashes=sorted(hashes))


def frozen(c):
    p=Path(c['freeze_dir'])/'freeze.json'; index=read_json(p); _validate_index(index)
    if index['train_budgets'] != c['train_budgets'] or index['sequence_length'] != c['sequence_length']:
        raise ValueError('frozen data protocol differs from config')
    targets={'development_calibration':c['development_budget']//2,
             'development_selection':c['development_budget']-c['development_budget']//2,'test':c['test_budget']}
    for budget in c['train_budgets']:
        targets['adapter_train_'+str(budget)]=9*budget//10
        targets['gate_fit_'+str(budget)]=budget-9*budget//10
    if index['seed']!=c['split_seed'] or set(index['manifests'])!=set(targets) or any(
            index['manifests'][role]['target_positions']!=target for role,target in targets.items()):
        raise ValueError('frozen seed/role allocation differs from scientific protocol')
    if index.get('minimum_source_groups')!=c['minimum_source_groups'] or any(
            len(v['source_groups'])<c['minimum_source_groups'] for v in index['manifests'].values()):
        raise ValueError('insufficient independent source documents in frozen role')
    return p,index


def cache_for(c, freeze_path, role, code):
    rows=load_manifest(freeze_path,role=role)
    preflight=read_json(Path(c['run_root'])/'preflight.json')
    binding=dict(rows=rows.identity,code=code,model_runtime_sha256=digest(preflight['model']))
    cache=DiskDocumentCache(Path(c['run_root'])/'cache'/role,binding,
        max_shard_bytes=c['max_shard_bytes'],max_document_bytes=c['max_document_bytes'])
    return rows,cache


def require_complete_cache(cache, rows):
    if cache.index is None or cache.index['status'] != 'complete' or len(cache) != len(rows):
        raise ValueError('complete frozen document cache required before fit')


def gpu_adapter(c, index):
    import torch
    from experiments.mixtral_experiment import load_local_model, load_local_inputs
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise ValueError('exactly four visible GPUs required; CPU tests cannot launch formal execution')
    if any(c['require_gpu_name'] not in torch.cuda.get_device_name(i) for i in range(4)):
        raise ValueError('visible devices differ from frozen A6000 resource protocol')
    inputs=load_local_inputs(c)
    _,_,adapter,identity=load_local_model(c,local_inputs=inputs)
    for role in ('model','tokenizer'):
        actual=identity[role]['files']
        expected={k:v['sha256'] for k,v in index[role]['files'].items()
                  if Path(k).suffix in {'.json','.safetensors','.model','.txt','.tiktoken'} and '.cache' not in Path(k).parts}
        if actual != expected: raise ValueError('actual loaded '+role+' differs from prepared byte hashes')
        if index[role].get('path') and Path(identity[role]['directory']).resolve()!=Path(index[role]['path']).resolve():
            raise ValueError('prepared artifact directory changed')
    from experiments.mixtral_fit_experiment import runtime_versions
    import platform
    identity['runtime']={**runtime_versions(), 'python':platform.python_version(),
        'gpu_names':[torch.cuda.get_device_name(i) for i in range(4)],
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    return adapter, identity


class StageReceipt:
    """Retain every attempt, including unknown hard interrupts, across commands."""
    def __init__(self,c,code,stage,resume=False):
        self.path=Path(c['run_root'])/'stage_receipts.json'
        if self.path.parent.exists() and not self.path.exists() and any(self.path.parent.iterdir()):
            raise FileExistsError('run_root contains unrelated files; choose a new independent run_id')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        binding=dict(config_sha256=digest(c),code=code)
        if self.path.exists():
            self.value=read_json(self.path)
            if self.value['binding'] != binding: raise ValueError('run_root identity changed; use a new run_id')
        else: self.value=dict(binding=binding,created=time.time(),attempts=[])
        now=time.time()
        for attempt in self.value['attempts']:
            if attempt['status']=='running':
                attempt.update(status='unknown_interruption',wall_seconds=max(0.,now-attempt['started']),
                    cost_scope='upper bound includes downtime; not treated as zero or scientific completion')
        if now-self.value['created'] >= c['suite_wall_seconds']:
            _atomic_json(self.path,self.value); raise TimeoutError('suite immutable wall deadline exhausted')
        self.attempt=dict(stage=stage,status='running',started=now,resume=resume)
        self.value['attempts'].append(self.attempt); _atomic_json(self.path,self.value)
    def finish(self,status,**details):
        self.attempt.update(status=status,wall_seconds=max(0.,time.time()-self.attempt['started']),**details)
        _atomic_json(self.path,self.value)


def parser():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config',required=True)
    p.add_argument('--stage',required=True,choices=('prepare','validate','cache','preflight','precision-screen','fit','report','finalize','final-test'))
    p.add_argument('--expected-commit')
    p.add_argument('--budget',type=int,choices=(100000,300000,1000000),default=100000)
    p.add_argument('--cell',choices=tuple(CELLS),default='r2_reduced')
    p.add_argument('--seed',type=int,choices=(17,29,43),default=17)
    p.add_argument('--matched-exposure',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--lock',help='immutable finalization JSON, required only for final-test')
    return p


def cell_output(c,budget,cell,seed,matched=False):
    return Path(c['run_root'])/'cells'/str(budget)/(('matched_' if matched else '')+cell)/('seed'+str(seed))


def main(argv=None):
    args=parser().parse_args(argv); c=validate_config(read_json(args.config))
    if args.stage=='validate':
        result=dict(status='config_valid',data_status='not_prepared',gpu_status='not_executed')
        if (Path(c['freeze_dir'])/'freeze.json').exists():
            path,index=frozen(c)
            for role in index['manifests']:
                if role!='test': load_manifest(path,role=role)
            result.update(data_status='frozen',freeze_sha256=index['freeze_sha256'])
        print(json.dumps(result,ensure_ascii=False)); return result
    if args.expected_commit is None: raise ValueError('formal stages require --expected-commit full40SHA')
    code=source_identity(args.expected_commit,clean=True)
    receipt=StageReceipt(c,code,args.stage,resume=args.resume)
    receipt.attempt['arguments']=vars(args)
    _atomic_json(receipt.path,receipt.value)
    try:
        if args.stage=='prepare':
            from experiments.mixtral_experiment import load_local_inputs, local_input_identity, local_artifact_stats
            inputs=load_local_inputs(c)
            # Hash real weights before calling this dataset frozen; never infer
            # model bytes from the declared upstream revision alone.
            stats=local_artifact_stats(c)
            model=local_artifact_identity(c['model_path'],required_files=('config.json',))
            if not any(p.endswith('.safetensors') for p in model['files']): raise ValueError('local safetensors weights required')
            tokenizer=model if Path(c['model_path']).resolve()==Path(c['tokenizer_path']).resolve() else local_artifact_identity(c['tokenizer_path'])
            if local_artifact_stats(c)!=stats or local_input_identity(c)!=inputs['identity']:
                raise ValueError('model/tokenizer changed during preparation hashing')
            exclusions=historical_exclusions(c['historical_report'])
            result=prepare_manifests(c['source_files'],inputs['tokenizer'],c['freeze_dir'],
                train_budgets=c['train_budgets'],development_budget=c['development_budget'],test_budget=c['test_budget'],
                sequence_length=c['sequence_length'],seed=c['split_seed'],model_identity=model,tokenizer_identity=tokenizer,
                minimum_source_groups=c['minimum_source_groups'],**exclusions)
            _atomic_json(Path(c['run_root'])/'preparation.json',dict(code=code,config=c,freeze_sha256=result['freeze_sha256'],
                historical_report_sha256=sha256_file(c['historical_report']),historical_exclusions=exclusions))
        else:
            from experiments.mixtral_experiment import precision_scope
            with precision_scope(False):
                result=execute_prepared(args,c,code)
        receipt.finish('complete')
        print(json.dumps(dict(stage=args.stage,status='complete',run_root=c['run_root'],
            frozen_manifest=str(Path(c['freeze_dir'])/'freeze.json'),
            finalization_lock=str(Path(c['run_root'])/'final_selection.json') if args.stage=='finalize' else None)))
        return result
    except BaseException as error:
        receipt.finish('failed',error_type=type(error).__name__,error=str(error)); raise


def execute_prepared(args,c,code):
    from mixtral_lab import loop_scale as core
    path,index=frozen(c)
    prepared=read_json(Path(c['run_root'])/'preparation.json')
    if prepared['code'] != code or prepared['config'] != c or prepared['freeze_sha256'] != index['freeze_sha256']:
        raise ValueError('prepared source/config/data identity changed')
    if args.stage=='report':
        if (Path(c['run_root'])/'final-test'/'report.json').exists():
            lock=check_lock(Path(c['run_root'])/'final_selection.json',c,code,index)
            result=completed_final(c,lock,True)
            _atomic_json(Path(c['run_root'])/'cost_summary.json',cost_summary(c))
            return result
        _atomic_json(Path(c['run_root'])/'cost_summary.json',cost_summary(c))
        return screen(c,args.budget)
    if args.stage=='finalize': return finalize(c,args.budget,code,index)
    if args.stage=='fit': fit_guard(args,c)
    if args.stage=='final-test':
        lock=check_lock(args.lock,c,code,index)
        complete=completed_final(c,lock,args.resume)
        if complete is not None: return complete
        adapter,model=gpu_adapter(c,index)
        require_preflight(c,code,index,model)
        return final_test(args,c,code,path,index,adapter,model,lock)
    if args.stage=='preflight' and (Path(c['run_root'])/'preflight.json').exists():
        raise FileExistsError('preflight already complete; do not repeat completed formal stage')
    if args.stage=='preflight' and (Path(c['run_root'])/'preflight_progress.json').exists():
        raise ValueError('partial preflight retained; diagnose and use a new run_id rather than replay unknown work')
    if args.stage=='cache' and (Path(c['run_root'])/'cache_complete.json').exists() and not args.resume:
        raise FileExistsError('cache already complete; use --resume only for verification')
    if args.stage=='fit' and cell_output(c,args.budget,args.cell,args.seed,args.matched_exposure).exists() and not args.resume:
        raise FileExistsError('cell already exists; use --resume for incomplete matching state')
    if args.stage=='precision-screen':
        if c.get('r2_precision') not in ('fp32_experts','fp32_canonical32'):
            raise ValueError('precision-screen requires explicit FP32 candidate config')
        if any((Path(c['run_root'])/name).exists() for name in ('precision_screen.json','precision_screen_progress.json')):
            raise FileExistsError('precision screen already attempted; preserve evidence and use a new run_id')
    adapter,model=gpu_adapter(c,index)
    if args.stage=='precision-screen':
        return precision_screen(adapter,model,c,code,path,index)
    if args.stage=='preflight':
        from mixtral_lab.document_training import cache_documents
        import torch
        out=Path(c['run_root'])/'preflight.json'
        if out.exists(): raise FileExistsError('preflight exists; do not repeat completed formal stage')
        started=time.time(); results=[]; rows=load_manifest(path,role='adapter_train_100000')
        # Shared across all documents/amplitudes/parameter states, not per case.
        preflight_deadline=min(suite_deadline(c),started+c['preflight_wall_seconds'])
        expert_probe_budget=dict(remaining=2, deadline=time.monotonic()+max(0.,preflight_deadline-time.time()))
        if len(rows)<c['preflight_documents']:
            raise ValueError('insufficient documents for complete preflight coverage')
        for i in range(c['preflight_documents']):
            if time.time()-started>c['preflight_wall_seconds']:
                raise TimeoutError('throughput preflight hard cap exhausted')
            check_suite_deadline(c)
            _atomic_json(Path(c['run_root'])/'preflight_progress.json',dict(status='incomplete',
                code=code,freeze_sha256=index['freeze_sha256'],results=results,model=model,
                attempted_capture=dict(document_index=i,status='running_or_unknown_until_completed'),
                completed_cases=len(results)))
            cache=cache_documents(adapter,[rows[i]],max_cache_bytes=c['max_document_bytes'])[0]
            for amplitude in c['amplitudes']:
                torch.cuda.synchronize()
                if time.time()-started>c['preflight_wall_seconds']:
                    raise TimeoutError('throughput preflight hard cap exhausted')
                check_suite_deadline(c)
                attempt=dict(document_index=i,document_id=cache.document_id,amplitude=amplitude,
                    status='running_or_unknown_until_completed')
                _atomic_json(Path(c['run_root'])/'preflight_progress.json',dict(status='incomplete',
                    code=code,freeze_sha256=index['freeze_sha256'],results=results,model=model,
                    attempted_case=attempt,completed_cases=len(results)))
                t=time.perf_counter()
                try:
                    result=core.preflight(adapter,cache,replace(scale_config(c,17),absolute_deadline_utc=preflight_deadline),amplitude=amplitude,expert_probe_budget=expert_probe_budget)
                    torch.cuda.synchronize()
                except BaseException as error:
                    _atomic_json(Path(c['run_root'])/'preflight_progress.json',dict(status='failed',
                        code=code,freeze_sha256=index['freeze_sha256'],results=results,model=model,
                        attempted_case=attempt,completed_cases=len(results),
                        error_type=type(error).__name__,error=str(error),
                        diagnostic_evidence=getattr(error,'diagnostic_evidence',None)))
                    raise
                result.update(document_id=cache.document_id,seconds=time.perf_counter()-t)
                results.append(result)
                _atomic_json(Path(c['run_root'])/'preflight_progress.json',dict(status='incomplete',
                    code=code,freeze_sha256=index['freeze_sha256'],results=results,model=model))
                # A numerical failure is evidence, not permission to train. Collect
                # the remaining bounded cases so one failure cannot hide amplitude coverage.
                if not result['passed'] and (not result['backbone_gradients_absent'] or not result['backbone_parameter_versions_unchanged']):
                    raise ValueError('backbone integrity failed; unsafe to continue preflight')
                if time.time()-started>c['preflight_wall_seconds']: raise TimeoutError('throughput preflight hard cap exhausted')
                check_suite_deadline(c)
        if len(results)!=c['preflight_documents']*len(c['amplitudes']):
            raise ValueError('preflight coverage incomplete')
        if not all(x['passed'] for x in results):
            _atomic_json(Path(c['run_root'])/'preflight_progress.json',dict(status='failed',
                code=code,freeze_sha256=index['freeze_sha256'],results=results,model=model,
                expected_cases=c['preflight_documents']*len(c['amplitudes']),
                completed_cases=len(results),reason='strict parity/backward checks failed; formal stages remain blocked'))
            raise ValueError('actual backend/compaction/backward preflight failed')
        total=sum(x['seconds'] for x in results)
        result=dict(status='complete',code=code,freeze_sha256=index['freeze_sha256'],results=results,
            seconds=total,model=model,peak_memory=memory_peaks(),
            scope='bounded real backend parity and backward; not a full-run time guarantee',
            projection_scope='per-document samples include native parity and backward; do not extrapolate as measured suite cost')
        _atomic_json(out,result)
        _atomic_json(Path(c['run_root'])/'preflight_progress.json',result)
        return result
    require_preflight(c,code,index,model)
    if args.stage=='cache':
        from mixtral_lab.document_training import cache_documents
        roles=['adapter_train_1000000','gate_fit_1000000','development_calibration','development_selection']
        per_document=4*c['sequence_length']*adapter.hidden_size+25*c['sequence_length']+65536
        estimated=sum(index['manifests'][r]['unique_documents'] for r in roles)*per_document
        if estimated>c['max_cache_disk_bytes']: raise ValueError('conservative cache disk estimate exceeds hard cap')
        receipts=read_json(Path(c['run_root'])/'stage_receipts.json')
        started=min(x['started'] for x in receipts['attempts'] if x['stage']=='cache');work={}
        work_path=Path(c['run_root'])/f'cache_work_attempt{len(receipts["attempts"]):06d}.json'
        def capture(row):
            check_suite_deadline(c)
            if time.time()-started>c['cache_wall_seconds']: raise TimeoutError('cache command wall cap exhausted')
            actual=cache_disk_usage(c)
            if actual+c['max_shard_bytes']>c['max_cache_disk_bytes']:
                raise ValueError('actual cache disk cap includes preserved failed/orphan shards')
            try:
                result=cache_documents(adapter,[row],max_cache_bytes=c['max_document_bytes'],work=work)[0]
            finally:
                _atomic_json(work_path,dict(work=work,wall_seconds=time.time()-started,
                    scope='this command attempt only; keep all attempt files; capture journals preserve unknown subcall costs'))
            return result
        results={}
        for role in roles:
            rows,cache=cache_for(c,path,role,code)
            if args.resume and cache.index is not None and cache.index['status']=='complete':
                require_complete_cache(cache,rows)
            else: cache.build(rows,capture,resume=args.resume and cache.index is not None)
            results[role]=cache.identity
        _atomic_json(Path(c['run_root'])/'cache_complete.json',dict(status='complete',identities=results,peak_memory=memory_peaks()))
        return results
    if args.stage!='fit': raise ValueError('unknown prepared stage')
    train=prefix_cache(c,path,'adapter_train',args.budget,code)
    gate=prefix_cache(c,path,'gate_fit',args.budget,code)
    _,cal=cache_for(c,path,'development_calibration',code)
    calrows=load_manifest(path,role='development_calibration'); require_complete_cache(cal,calrows)
    devrows,dev=cache_for(c,path,'development_selection',code); require_complete_cache(dev,devrows)
    target=None;gate_target=None
    if args.matched_exposure:
        target=2*index['manifests']['adapter_train_1000000']['valid_positions']
        gate_target=index['manifests']['gate_fit_1000000']['valid_positions']
    config=scale_config(c,args.seed,target=target,gate_target=gate_target)
    kind,amplitude=CELLS[args.cell]
    identity=dict(code=code,freeze_sha256=index['freeze_sha256'],model=model,
        config_sha256=digest(c),budget=args.budget,cell=args.cell,matched_exposure=args.matched_exposure)
    output=cell_output(c,args.budget,args.cell,args.seed,args.matched_exposure)
    result=core.run_cell(adapter,train,gate,dev,config,identity,output,
        calibration_caches=cal,kind=kind,amplitude=amplitude,resume=args.resume)
    _atomic_json(output/'gpu_memory.json',memory_peaks())
    return result


def suite_deadline(c):
    return read_json(Path(c['run_root'])/'stage_receipts.json')['created']+c['suite_wall_seconds']


def check_suite_deadline(c):
    if time.time()>=suite_deadline(c): raise TimeoutError('suite immutable wall deadline exhausted')


def scale_config(c,seed,*,target=None,gate_target=None):
    from mixtral_lab.loop_scale import ScaleConfig
    return ScaleConfig(r2_precision=c.get('r2_precision','native_bf16'),rank=8,epochs=2,positions_per_update=256,seed=seed,
        max_updates=100000,max_wall_seconds=c['cell_wall_seconds'],target_exposures=target,
        gate_target_exposures=gate_target,absolute_deadline_utc=suite_deadline(c))


class PrefixCache(Sequence):
    """Nested datasets refer to the single largest committed prefix cache."""
    def __init__(self,cache,rows):
        self.cache=cache;self.rows=rows
        if cache.binding['rows']['freeze_sha256']!=rows.identity['freeze_sha256']:
            raise ValueError('nested row and cache freeze identities differ')
        if cache.binding['rows']['role']!=rows.metadata['family']+'_1000000':
            raise ValueError('nested rows do not belong to cache family')
        require_complete_cache(cache,type('FullRows',(),{'__len__':lambda _:cache.index['expected_documents']})())
        if len(rows)>len(cache): raise ValueError('nested cache length exceeds full cache')
        for i in range(len(rows)):
            if cache.index['entries'][i]['document_id']!=rows.metadata['document_ids'][i]:
                raise ValueError('cache is not the declared nested document prefix')
        self.identity=dict(rows=rows.identity,parent_cache=cache.identity)
    def __len__(self): return len(self.rows)
    def document_metadata(self,index):
        if index<0: index+=len(self)
        if not 0<=index<len(self): raise IndexError(index)
        return self.cache.document_metadata(index)
    def __getitem__(self,index):
        if isinstance(index,slice): raise ValueError('stream one cached document at a time')
        if index<0: index+=len(self)
        if not 0<=index<len(self): raise IndexError(index)
        return self.cache[index]


def prefix_cache(c,path,family,budget,code):
    _,cache=cache_for(c,path,family+'_1000000',code)
    return PrefixCache(cache,load_manifest(path,role=family+'_'+str(budget)))


def memory_peaks():
    import torch
    return {str(i):dict(allocated=torch.cuda.max_memory_allocated(i),reserved=torch.cuda.max_memory_reserved(i))
            for i in range(torch.cuda.device_count())}


def cache_disk_usage(c):
    return sum(p.stat().st_size for directory in ('cache','final-test-cache')
        for p in (Path(c['run_root'])/directory).rglob('*') if p.is_file())


def cost_summary(c):
    receipts=read_json(Path(c['run_root'])/'stage_receipts.json')
    ledgers={str(p.relative_to(Path(c['run_root']))):read_json(p) for p in
             sorted((Path(c['run_root'])/'cells').glob('**/ledger.json'))}
    if (Path(c['run_root'])/'final-test'/'ledger.json').exists():
        ledgers['final-test/ledger.json']=read_json(Path(c['run_root'])/'final-test'/'ledger.json')
    cache=Path(c['run_root'])/'cache'
    return dict(stage_attempts=receipts['attempts'],cell_ledgers=ledgers,
        cache_work_attempts={p.name:read_json(p) for p in sorted(Path(c['run_root']).glob('cache_work_attempt*.json'))},
        cache_capture_journals={str(p.relative_to(Path(c['run_root']))):p.read_text() for directory in ('cache','final-test-cache')
            for p in sorted((Path(c['run_root'])/directory).glob('**/attempts.jsonl'))},
        cache_bytes_including_orphans=cache_disk_usage(c),
        suite_elapsed_seconds=max(0.,time.time()-receipts['created']),
        gpu_hours_scope='stage and cell wall receipts overlap: do not sum them; elapsed includes hashing, I/O, and conservative downtime',
        runtime_cost_scope='all attempted/failed/unknown operations retained; CPU timestamps do not imply measured kernel FLOPs')


def require_preflight(c,code,index,model):
    preflight=read_json(Path(c['run_root'])/'preflight.json')
    actual_model=json.loads(json.dumps(model,allow_nan=False))
    if (preflight['status']!='complete' or preflight['code']!=code or
            preflight['freeze_sha256']!=index['freeze_sha256'] or preflight['model']!=actual_model):
        raise ValueError('successful source/model/data/runtime-bound preflight required')
    return preflight


def cell_report(c,budget,cell,seed=17,matched=False):
    from mixtral_lab import loop_scale as core
    path=cell_output(c,budget,cell,seed,matched)/'report.json';r=read_json(path)
    if r.get('status')!='complete' or r.get('sha256')!=core.digest({k:v for k,v in r.items() if k!='sha256'}):
        raise ValueError('complete untampered cell report required')
    if r['identity']['config_sha256']!=digest(c) or r['identity']['budget']!=budget or r['identity']['cell']!=cell:
        raise ValueError('cell report protocol mismatch')
    return r


def screen(c,budget):
    """Freeze one finite exploratory screen before confirmation seeds are run."""
    from mixtral_lab.loop_scale import paired_comparison
    reports={name:cell_report(c,budget,name) for name in CELLS}
    candidates=[]
    for name in ('r2_full','r2_reduced'):
        r=reports[name];results=r['selection_results']
        for mode in ('forced','gate'):
            comparison=paired_comparison(results[mode],results['native'])
            candidates.append(dict(cell=name,mode=mode,nll=results[mode]['nll'],comparison=comparison))
    winner=min((x for x in candidates if x['mode']=='gate'),key=lambda x:(x['nll'],x['cell']))
    forced_winner=min((x for x in candidates if x['mode']=='forced'),key=lambda x:(x['nll'],x['cell']))
    # The finite 100k->300k contrast tests whether data can repair a negative
    # branch. Requiring an already-positive native gain would censor that test.
    advancing=budget==100000
    trends=[]
    if budget==300000:
        for name in ('r2_full','r2_reduced'):
            old=cell_report(c,100000,name)
            for mode in ('forced','gate'):
                comparison=paired_comparison(reports[name]['selection_results'][mode],old['selection_results'][mode])
                trends.append(dict(cell=name,mode=mode,comparison=comparison,
                    question='same structure, 300k versus100k; native may still be better'))
        advancing=any(x['comparison']['gain_nats']>=c['minimum_gain'] and x['comparison']['ci95'][0]>0 for x in trends)
    value=dict(status='complete',budget=budget,candidates=candidates,winner=winner,forced_winner=forced_winner,
        allow_next_budget=advancing,data_scale_trends=trends,minimum_gain=c['minimum_gain'],
        escalation_rule=('preauthorized finite100k-to300k contrast after complete cohort, regardless of native gain'
            if budget==100000 else 'same-cell/mode300k-vs100k gain>=.001 and source-paired95%CI lower>0; no native-gain requirement'
            if budget==300000 else '1m is the final predeclared scale; no automatic further escalation'),
        reports={k:sha256_file(cell_output(c,budget,k,17)/'report.json') for k in CELLS},
        scope='exploratory development selection, document uncertainty excludes candidate-selection optimism')
    value['sha256']=digest(value);out=Path(c['run_root'])/f'screen-{budget}.json'
    if out.exists() and read_json(out)!=value: raise ValueError('screen already frozen; do not change candidate cohort')
    _atomic_json(out,value);return value


def fit_guard(args,c):
    if (Path(c['run_root'])/'final_selection.json').exists():
        raise ValueError('final cohort locked; no further fitting or retuning in this run')
    if args.matched_exposure:
        if args.budget!=100000 or args.seed!=17: raise ValueError('matched-exposure arm is predeclared100k seed17 only')
        for name in CELLS: cell_report(c,1000000,name)
    elif args.budget>100000:
        prior=100000 if args.budget==300000 else 300000
        prior_screen=screen(c,prior)
        if not prior_screen['allow_next_budget']: raise ValueError('predeclared data-escalation criterion failed; preserve negative result')
    if args.seed!=17:
        winner=screen(c,args.budget)['winner']
        if winner['comparison']['gain_nats']<c['minimum_gain'] or winner['comparison']['ci95'][0]<=0:
            raise ValueError('conditional screen has no confirmed positive signal; do not expand confirmation seeds')
        selected=winner['cell']
        if args.cell not in (selected,'parameter_control'):
            raise ValueError('confirmation seeds limited to frozen selected R2 and parameter control')


def finalize(c,budget,code,index):
    # Complete the finite primary data contrast before unsealing any test,
    # even if a smaller-data conditional candidate already looks promising.
    primary_contrast={str(n):screen(c,n) for n in (100000,300000)}
    s=screen(c,budget); selected=s['winner']['cell']
    q=s['winner']['comparison']
    if q['gain_nats']<c['minimum_gain'] or q['ci95'][0]<=0:
        raise ValueError('conditional screen failed; preserve negative result and leave final test sealed')
    if budget==1000000:
        for name in CELLS:
            large=cell_report(c,budget,name)
            small=cell_report(c,100000,name,matched=True)
            if (large['adapter_exposures'],large['adapter_updates'],large['gate_updates'])!=(small['adapter_exposures'],small['adapter_updates'],small['gate_updates']):
                raise ValueError('matched data-diversity arm has different exposure or update count')
    cells=[]
    for seed in c['seeds']:
        for cell in (selected,'parameter_control'):
            r=cell_report(c,budget,cell,seed);out=cell_output(c,budget,cell,seed)
            files=[out/'report.json',*sorted(out.glob('*-epoch*.pt'))]
            cells.append(dict(seed=seed,cell=cell,output=str(out.resolve()),
                report_sha256=sha256_file(out/'report.json'),artifacts={p.name:sha256_file(p) for p in files}))
    lock=dict(schema='mixtral-loop-scale-final-lock-v2',code=code,config_sha256=digest(c),
        freeze_sha256=index['freeze_sha256'],budget=budget,selected_cell=selected,screen=s,
        primary_data_contrast=primary_contrast,cells=cells,
        test_policy='one frozen cohort; all seeds and baselines reported, no retuning after final test')
    lock['sha256']=digest(lock);path=Path(c['run_root'])/'final_selection.json'
    if path.exists():
        if read_json(path)!=lock: raise ValueError('final cohort already locked; no replacement after seeing test')
        return lock
    _atomic_json(path,lock);return lock


def check_lock(path,c,code,index):
    expected=Path(c['run_root'])/'final_selection.json'
    if path is None or Path(path).resolve()!=expected.resolve(): raise ValueError('explicit canonical final lock required')
    lock=read_json(path);value={k:v for k,v in lock.items() if k!='sha256'}
    if lock.get('sha256')!=digest(value) or lock['code']!=code or lock['config_sha256']!=digest(c) or lock['freeze_sha256']!=index['freeze_sha256']:
        raise ValueError('final lock identity mismatch')
    for item in lock['cells']:
        for name,claimed in item['artifacts'].items():
            if Path(name).name!=name or sha256_file(Path(item['output'])/name)!=claimed:
                raise ValueError('final frozen checkpoint/report changed')
    return lock


def final_test(args,c,code,path,index,adapter,model,lock):
    from mixtral_lab import loop_scale as core
    from mixtral_lab.document_training import cache_documents
    rows=load_manifest(path,role='test',allow_test=True)
    binding=dict(rows=rows.identity,code=code,final_lock_sha256=lock['sha256'])
    cache=DiskDocumentCache(Path(c['run_root'])/'final-test-cache',binding,
        max_shard_bytes=c['max_shard_bytes'],max_document_bytes=c['max_document_bytes'])
    per_document=4*c['sequence_length']*adapter.hidden_size+25*c['sequence_length']+65536
    if cache_disk_usage(c)+max(0,len(rows)-len(cache))*per_document>c['max_cache_disk_bytes']:
        raise ValueError('sealed test cache estimate exceeds shared disk cap; leave existing evidence intact')
    root=Path(c['run_root'])/'final-test'
    ledger=core.RunLedger(root,core.digest(binding),c['cell_wall_seconds'],resume=args.resume,
        absolute_deadline_utc=suite_deadline(c))
    def capture(row):
        if cache_disk_usage(c)+c['max_shard_bytes']>c['max_cache_disk_bytes']:
            raise ValueError('shared cache disk cap includes sealed-test and preserved orphan shards')
        with ledger.operation('sealed-test-prefix:'+row['id']):
            return cache_documents(adapter,[row],max_cache_bytes=c['max_document_bytes'])[0]
    if cache.index is None or cache.index['status']!='complete':
        cache.build(rows,capture,resume=args.resume and cache.index is not None)
    results={}
    for item in lock['cells']:
        r=read_json(Path(item['output'])/'report.json');cfg=core.ScaleConfig(**r['config'])
        branch,gate,selection=core.load_selected(item['output'],cfg,r['identity'],adapter.device,which='gated')
        forced,_,_=core.load_selected(item['output'],cfg,r['identity'],adapter.device,which='forced')
        variants=[('native',None,None,'native',None),('forced',forced,None,'forced',None)]
        if item['cell']!='parameter_control':
            variants.append(('gate',branch,gate,'gate' if branch is not None else 'native',selection['threshold']))
            if branch is not None:
                variants.extend((mode,branch,gate,mode,selection['threshold']) for mode in ('matched_random','fixed_hash_25','fixed_hash_50','fixed_hash_100'))
        label=f"{item['cell']}-seed{item['seed']}";results[label]={}
        for name,b,g,mode,threshold in variants:
            results[label][name]=core.evaluate(adapter,cache,b,gate=g,threshold=threshold,mode=mode,
                role='final_test',config=cfg,output=root/label/name,ledger=ledger,allow_final_test=True)
    artifacts={str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob('*.json')) if p.name!='ledger.json'}
    comparisons=final_comparisons(results,lock['selected_cell'],c['seeds'])
    report=dict(status='complete',final_lock_sha256=lock['sha256'],results=results,comparisons=comparisons,model=model,cost=ledger.summary(),artifacts=artifacts,
        limitation='cached-prefix quality only; no online end-to-end speed measurement')
    report['sha256']=digest(report)
    _atomic_json(root/'report.json',report);return report


def final_comparisons(results,selected,seeds):
    from mixtral_lab.loop_scale import paired_comparison
    rows={};gains={}
    for seed in seeds:
        r=results[f'{selected}-seed{seed}'];control=results[f'parameter_control-seed{seed}']['forced']
        paired={}
        for mode in ('gate','forced'):
            for name,reference in [('native',r['native']),('parameter_control',control)]:
                key=mode+'_vs_'+name
                paired[key]=paired_comparison(r[mode],reference,seed=seed)
                gains.setdefault(key,[]).append(paired[key]['gain_nats'])
        if 'matched_random' in r:
            paired['gate_vs_matched_random']=paired_comparison(r['gate'],r['matched_random'],seed=seed)
            gains.setdefault('gate_vs_matched_random',[]).append(paired['gate_vs_matched_random']['gain_nats'])
        rows[str(seed)]=paired
    return dict(per_seed=rows,across_seeds={key:dict(gains=values,mean=sum(values)/len(values),
        minimum=min(values),maximum=max(values)) for key,values in gains.items()},
        scope='document-paired intervals within each seed; three-seed descriptive summary, no pooled independent-token claim',
        selection_scope='forced result uses independently selected forced epoch of the frozen primary conditional amplitude')


def completed_final(c,lock,resume):
    path=Path(c['run_root'])/'final-test'/'report.json'
    if not path.exists(): return None
    if not resume: raise FileExistsError('final test already complete; use --resume only to verify and return existing evidence')
    r=read_json(path)
    if r.get('sha256')!=digest({k:v for k,v in r.items() if k!='sha256'}) or r['final_lock_sha256']!=lock['sha256']:
        raise ValueError('completed final test evidence identity changed')
    for name,claimed in r['artifacts'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(path.parent/name)!=claimed:
            raise ValueError('completed final test artifact changed')
    return r





# Imported use only; main() is moved below these definitions.
def precision_screen(adapter, model, c, code, path, index):
    """Bounded paired numerical screen; NEVER writes a preflight certificate."""
    import torch
    from mixtral_lab import loop_scale as core
    from mixtral_lab.document_training import cache_documents
    root=Path(c['run_root']); started=time.time()
    deadline=min(suite_deadline(c),started+c['preflight_wall_seconds'])
    rows=load_manifest(path,role='adapter_train_100000')
    if len(rows)<c['preflight_documents']:
        raise ValueError('insufficient documents for precision screen')
    modes_all=('native_bf16','fp32_experts','fp32_canonical32') if c.get('r2_precision')=='fp32_canonical32' else ('native_bf16','fp32_experts')
    path_probe_budget={'remaining':12} if len(modes_all)==3 else None
    records=[];attempt=None
    def publish(status, **extra):
        value=dict(status=status,code=code,model=model,freeze_sha256=index['freeze_sha256'],
            config_sha256=digest(c),protocol='canonical32-r2-screen-v1' if len(modes_all)==3 else 'fp32-r2-screen-v1',absolute_deadline_utc=deadline,
            canonical_tile_rows=32 if len(modes_all)==3 else None,
            results=records,completed_cases=len(records),expected_cases=c['preflight_documents']*len(c['amplitudes'])*len(modes_all),
            attempted_case=attempt,scope='numerical candidate screen only; no preflight certificate or training authority',**extra)
        _atomic_json(root/'precision_screen_progress.json',value)
        return value
    def sync():
        for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
    def check():
        if time.time()>=deadline:raise TimeoutError('precision screen shared deadline exhausted')
    try:
        publish('incomplete')
        for i in range(c['preflight_documents']):
            check();attempt=dict(document_index=i,operation='capture')
            publish('incomplete')
            cache=cache_documents(adapter,[rows[i]],max_cache_bytes=c['max_document_bytes'])[0]
            for j,amplitude in enumerate(c['amplitudes']):
                shift=(i+j)%len(modes_all)
                modes=modes_all[shift:]+modes_all[:shift]
                for mode in modes:
                    check();sync()
                    for device in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(device)
                    baseline={str(d):dict(allocated=torch.cuda.memory_allocated(d),reserved=torch.cuda.memory_reserved(d))
                        for d in range(torch.cuda.device_count())}
                    attempt=dict(document_index=i,amplitude=amplitude,r2_precision=mode,operation='preflight_with_gradients')
                    publish('incomplete')
                    config=replace(scale_config(c,17),r2_precision=mode,absolute_deadline_utc=deadline)
                    t=time.perf_counter()
                    result=core.preflight(adapter,cache,config,amplitude=amplitude,gradient_checks=True,diagnostics=False,path_probe_budget=path_probe_budget)
                    sync()
                    elapsed=time.perf_counter()-t
                    diagnostic_seconds=result.get('work',{}).get('path_diagnostic_seconds',0.)
                    result.update(document_index=i,document_id=cache.document_id,
                        seconds=elapsed,path_diagnostic_seconds=diagnostic_seconds,
                        seconds_excluding_path_diagnostics=max(0.,elapsed-diagnostic_seconds),
                        peak_memory_includes_path_diagnostics=bool(result.get('work',{}).get('path_diagnostic_attempts',0)),
                        peak_memory=memory_peaks(),baseline_memory=baseline,
                        cost_scope='synchronized correctness+gradient screen; conditional diagnostic costs separated; NOT inference throughput')
                    records.append(result);publish('incomplete')
                    if not result['backbone_gradients_absent'] or not result['backbone_parameter_versions_unchanged']:
                        raise ValueError('backbone integrity failed during precision screen')
                    check()
        expected=c['preflight_documents']*len(c['amplitudes'])
        by_mode={mode:[r for r in records if r['r2_precision']==mode] for mode in modes_all}
        acceptance={mode:len(values)==expected and all(r['passed'] for r in values) for mode,values in by_mode.items()}
        value=publish('complete',numerical_checks_passed=acceptance,seconds=time.time()-started,
            conclusion='candidate still requires separate bound preflight and explicit operator decision before training')
        _atomic_json(root/'precision_screen.json',value)
        return value
    except BaseException as error:
        publish('failed',error_type=type(error).__name__,error=str(error),
            diagnostic_evidence=getattr(error,'diagnostic_evidence',None),peak_memory=memory_peaks())
        raise


if __name__=='__main__': main()
