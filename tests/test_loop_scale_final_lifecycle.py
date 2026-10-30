"""Actual tiny-HF six-cell final-test lifecycle, frozen disk data and evidence."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments import mixtral_loop_scale as cli
from mixtral_lab import loop_scale as core
from mixtral_lab.adapter import MixtralAdapter
from mixtral_lab.document_training import cache_documents
from mixtral_lab.loop_scale_data import (prepare_manifests,load_manifest,DiskDocumentCache,
                                        local_artifact_identity,digest,sha256_file)
from tests.test_mixtral_adapter import tiny


class TinyTokenizer:
    pad_token_id=0
    def __call__(self,text,**kwargs):
        i=int(text.split()[0])
        return {'input_ids':[1+i%30,1+(i//30)%30,3,4,5,6],
                'attention_mask':[1]*6}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class ActualFinalLifecycle(unittest.TestCase):
    def test_real_six_cells_partial_resume_final_statistics_and_locked_evidence(self):
        torch.manual_seed(204);adapter=MixtralAdapter(tiny())
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);runroot=root/'run';runroot.mkdir()
            code={'commit':'a'*40,'sources':{'tiny':'f'*64}}
            artifact=root/'tiny-model';artifact.mkdir();(artifact/'config.json').write_text('{}')
            identity=local_artifact_identity(artifact)
            source=root/'source.jsonl'
            source.write_text(''.join(json.dumps(dict(id=f'doc{i}',source_id=f'source{i}',
                text=f'{i} distinct local source text'))+'\n' for i in range(160)))
            freeze=root/'freeze'
            index=prepare_manifests([source],TinyTokenizer(),freeze,train_budgets=(20,),
                development_budget=20,test_budget=20,sequence_length=6,seed=17,
                tokenizer_identity=identity,model_identity=identity)
            c=dict(run_root=str(runroot),cell_wall_seconds=300,suite_wall_seconds=300,
                max_shard_bytes=1024**2,max_document_bytes=1024**2,seeds=[17,29,43],
                sequence_length=6,max_cache_disk_bytes=64*1024**2)
            write(runroot/'stage_receipts.json',dict(created=time.time(),attempts=[]))
            def cached(role):
                rows=load_manifest(freeze,role=role)
                cache=DiskDocumentCache(root/'cache'/role,dict(rows=rows.identity,code=code),
                    max_shard_bytes=1024**2,max_document_bytes=1024**2)
                cache.build(rows,lambda row:cache_documents(adapter,[row])[0])
                return cache
            train=cached('adapter_train_20');gate=cached('gate_fit_20')
            cal=cached('development_calibration');dev=cached('development_selection')
            cells=[]
            for seed in c['seeds']:
                cfg=core.ScaleConfig(rank=2,epochs=1,positions_per_update=4,max_updates=100,
                    max_wall_seconds=300,seed=seed,thresholds=(0.,.01))
                for cell in ('r2_reduced','parameter_control'):
                    out=cli.cell_output(c,20,cell,seed)
                    ids=dict(code=code,freeze_sha256=index['freeze_sha256'],
                        config_sha256=digest(c),budget=20,cell=cell)
                    kind,amplitude=cli.CELLS[cell]
                    report=core.run_cell(adapter,train,gate,dev,cfg,ids,out,
                        calibration_caches=cal,kind=kind,amplitude=amplitude)
                    self.assertEqual(report['status'],'complete')
                    artifacts=[out/'report.json',*out.glob('*-epoch*.pt')]
                    cells.append(dict(seed=seed,cell=cell,output=str(out.resolve()),
                        artifacts={p.name:sha256_file(p) for p in artifacts}))
            lock=dict(schema='mixtral-loop-scale-final-lock-v2',code=code,config_sha256=digest(c),
                freeze_sha256=index['freeze_sha256'],budget=20,selected_cell='r2_reduced',cells=cells)
            lock['sha256']=digest(lock);lockpath=runroot/'final_selection.json';write(lockpath,lock)
            self.assertEqual(cli.check_lock(lockpath,c,code,index),lock)
            original=core.RunLedger.operation;interrupted=[]
            @contextmanager
            def after_first_final_document(ledger,key,**kwargs):
                with original(ledger,key,**kwargs) as event:
                    yield event
                if key.startswith('evaluate:') and 'final-test' in key and not interrupted:
                    interrupted.append(key)
                    raise InterruptedError('controlled final document interruption after evidence commit')
            args=SimpleNamespace(resume=False)
            with patch.object(core.RunLedger,'operation',after_first_final_document):
                with self.assertRaises(InterruptedError):
                    cli.final_test(args,c,code,freeze/'freeze.json',index,adapter,{'tiny':'actual CPU'},lock)
            first=(runroot/'final-test/r2_reduced-seed17/native/document-000000.json')
            self.assertTrue(first.exists());firstbytes=first.read_bytes()
            self.assertFalse((runroot/'final-test/report.json').exists())
            args.resume=True
            # A completed test-prefix cache must be reused; the actual model
            # expert/head paths remain real for unfinished quality variants.
            with patch('mixtral_lab.document_training.cache_documents',side_effect=AssertionError('completed prefix repeated')):
                final=cli.final_test(args,c,code,freeze/'freeze.json',index,adapter,{'tiny':'actual CPU'},lock)
            self.assertEqual(first.read_bytes(),firstbytes)
            events=json.loads((runroot/'final-test/ledger.json').read_text())['events']
            self.assertEqual(sum(e['key']==interrupted[0] for e in events),1)
            self.assertEqual(final['status'],'complete')
            self.assertEqual(set(final['comparisons']['per_seed']),{'17','29','43'})
            for seed in c['seeds']:
                r=final['results'][f'r2_reduced-seed{seed}']
                baseline=final['results'][f'parameter_control-seed{seed}']['forced']
                for mode in ('forced','gate'):
                    expected=core.paired_comparison(r[mode],baseline,seed=seed)
                    actual=final['comparisons']['per_seed'][str(seed)][mode+'_vs_parameter_control']
                    self.assertEqual(actual,expected)
                ev=next((runroot/f'final-test/r2_reduced-seed{seed}/forced').glob('document-*.json'))
                raw=json.loads(ev.read_text())['evidence']
                self.assertAlmostEqual(sum(raw['candidate_ce'])/raw['tokens'],raw['candidate_nll'])
            # Completed execution returns frozen evidence without loading models
            # or calling training/scoring on the already consumed final test.
            with patch.object(core,'evaluate',side_effect=AssertionError('final scoring repeated')), \
                 patch.object(cli,'gpu_adapter',side_effect=AssertionError('model loaded after final completion')):
                completed=cli.completed_final(c,lock,resume=True)
            self.assertEqual(completed,final)
            with self.assertRaises(FileExistsError):cli.completed_final(c,lock,resume=False)
            checkpoint=next((Path(cells[0]['output'])).glob('candidate-epoch1.pt'))
            checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint/report changed'):
                cli.check_lock(lockpath,c,code,index)


if __name__=='__main__':unittest.main()
