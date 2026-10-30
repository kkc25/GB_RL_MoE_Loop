"""CPU-only frozen-data and disk-shard lifecycle tests; never downloads artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from mixtral_lab.loop_scale_data import (prepare_manifests, load_manifest,
    local_artifact_identity, sha256_file, digest, DiskDocumentCache)

try:
    import torch
except ImportError:
    torch = None


class Tokenizer:
    def __call__(self, text, **kwargs):
        n = min(len(text.split()), kwargs['max_length'])
        import hashlib
        ids=[int(hashlib.sha256(word.encode()).hexdigest()[:6],16)+1 for word in text.split()[:n]]
        return dict(input_ids=ids+[0]*(kwargs['max_length']-n),
                    attention_mask=[1]*n+[0]*(kwargs['max_length']-n))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        artifact = self.root/'artifacts'; artifact.mkdir(); (artifact/'config.json').write_text('{}')
        self.artifact = local_artifact_identity(artifact, required_files=('config.json',))
        self.source = self.root/'source.jsonl'
        self.rows = [dict(id=f'doc-{i}',source_id=f'source-{i}',text=f'{i} unique source words a b c d e f') for i in range(150)]
        self.write()
    def tearDown(self): self.temp.cleanup()
    def write(self): self.source.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
    def prepare(self, name='frozen', **kwargs):
        defaults = dict(train_budgets=(40,80,120), development_budget=20,test_budget=20,
            sequence_length=10,seed=17,tokenizer_identity=self.artifact,model_identity=self.artifact)
        defaults.update(kwargs)
        return prepare_manifests([self.source],Tokenizer(),self.root/name,**defaults)
    def test_actual_local_file_bytes_bind_identity(self):
        self.assertEqual(self.artifact['files']['config.json']['sha256'],sha256_file(self.root/'artifacts/config.json'))
        with self.assertRaises(ValueError): local_artifact_identity(self.root/'artifacts',required_files=('absent',))
        (self.root/'artifacts/config.json').write_text('{"changed":true}')
        self.assertNotEqual(local_artifact_identity(self.root/'artifacts')['files_sha256'],self.artifact['files_sha256'])
    def test_deterministic_freeze_and_nested_source_disjoint_counts(self):
        a = self.prepare(); b = self.prepare('again')
        self.assertEqual(a['freeze_sha256'],b['freeze_sha256'])
        self.assertEqual(a['manifests'],b['manifests'])
        maximum=[]
        for family in ('adapter_train','gate_fit'):
            previous=[]
            for budget in (40,80,120):
                entry=a['manifests'][f'{family}_{budget}']
                self.assertEqual(entry['document_ids'][:len(previous)],previous)
                self.assertGreaterEqual(entry['valid_positions'],entry['target_positions'])
                self.assertLess(entry['overshoot_positions'],9)
                previous=entry['document_ids']
            maximum.append(a['manifests'][f'{family}_120'])
        maximum += [a['manifests'][k] for k in ('development_calibration','development_selection','test')]
        for i,x in enumerate(maximum):
            for y in maximum[i+1:]:
                self.assertFalse(set(x['source_groups']) & set(y['source_groups']))
                self.assertFalse(set(x['content_hashes']) & set(y['content_hashes']))
    def test_dedup_normalization_and_source_union(self):
        self.rows += [dict(id='alias',source_id='alias-source',text=self.rows[0]['text'].replace(' ','  '))]
        self.rows += [dict(id='alias-other',source_id='alias-source',text='other alias group a b c d e f')]
        self.write(); index=self.prepare()
        self.assertEqual(index['counts']['duplicate_rows_removed'],1)
        self.assertEqual(index['counts']['unique_normalized_content'],151)
        families={}
        for item in index['manifests'].values():
            for doc in item['document_ids']: families.setdefault(doc,set()).add(item['family'])
        if 'doc-0' in families and 'alias-other' in families:
            self.assertEqual(families['doc-0'],families['alias-other'])
        self.assertNotIn('alias',families)
    def test_exclusions_remove_connected_source_group(self):
        self.rows += [dict(id='same-source',source_id='source-0',text='another same source words a b c d e')]
        self.write()
        normalized=self.rows[0]['text']
        import hashlib
        index=self.prepare(excluded_content_hashes=[hashlib.sha256(normalized.encode()).hexdigest()])
        self.assertEqual(index['counts']['excluded_documents'],2)
        all_ids={d for v in index['manifests'].values() for d in v['document_ids']}
        self.assertNotIn('doc-0',all_ids); self.assertNotIn('same-source',all_ids)
        self.assertEqual(index['exclusions_sha256'],digest(index['exclusions']))
    def test_explicit_document_and_source_exclusions(self):
        index=self.prepare(excluded_document_ids=['doc-0'],excluded_source_ids=['source-1'])
        self.assertEqual(index['counts']['excluded_documents'],2)
        all_ids={d for v in index['manifests'].values() for d in v['document_ids']}
        self.assertFalse({'doc-0','doc-1'} & all_ids)
    def test_shortage_fails_without_claiming_frozen_dataset(self):
        self.rows=self.rows[:3]; self.write()
        with self.assertRaisesRegex(ValueError,'insufficient source-disjoint'): self.prepare()
        self.assertFalse((self.root/'frozen/freeze.json').exists())
    def test_legacy_raw_content_excludes_whole_source_group(self):
        import hashlib
        self.rows[0]['text']='  legacy   whitespace document a b c d e f  '
        self.rows.append(dict(id='legacy-related',source_id='source-0',text='different content from old source a b c d'))
        self.write()
        raw_hash=hashlib.sha256(self.rows[0]['text'].encode()).hexdigest()
        index=self.prepare(excluded_content_hashes=[raw_hash])
        self.assertEqual(index['counts']['excluded_documents'],2)
    def test_incomplete_preparation_never_loads_as_frozen(self):
        self.prepare();(self.root/'frozen/PREPARING.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'preparation is incomplete'):
            load_manifest(self.root/'frozen','adapter_train_40')
    def test_effective_truncated_token_dedup_before_split(self):
        self.rows.append(dict(id='effective-alias',source_id='alias-source',text=self.rows[0]['text']+' discarded suffix'))
        self.rows.append(dict(id='alias-group-extra',source_id='alias-source',text='source sharing the same group but other context a b c d'))
        self.write();index=self.prepare()
        self.assertEqual(index['counts']['duplicate_effective_sequences_removed'],1)
        ids={d for v in index['manifests'].values() for d in v['document_ids']}
        self.assertNotIn('effective-alias',ids)
    def test_effective_duplicate_exclusion_propagates_to_source_group(self):
        self.rows.append(dict(id='effective-alias',source_id='alias-source',text=self.rows[0]['text']+' discarded suffix'))
        self.rows.append(dict(id='alias-group-extra',source_id='alias-source',text='source sharing other context a b c d e f g'))
        self.write();index=self.prepare(excluded_document_ids=['doc-0'])
        ids={d for v in index['manifests'].values() for d in v['document_ids']}
        self.assertFalse({'doc-0','effective-alias','alias-group-extra'} & ids)
    def test_hard_source_preparation_bounds(self):
        for kw in (dict(max_source_bytes=1),dict(max_row_bytes=10),dict(max_source_rows=1)):
            with self.assertRaisesRegex(ValueError,'bound'):self.prepare(**kw)
        self.assertFalse((self.root/'frozen').exists())
    @unittest.skipIf(torch is None,'torch required for token tensors')
    def test_no_pad_token_uses_eos_without_mutating_tokenizer(self):
        class NoPad(Tokenizer):
            pad_token_id=None;eos_token_id=0
            def __call__(self,text,**kwargs):
                base=super().__call__(text,**kwargs)
                return {k:v[:3] for k,v in base.items()}
        index=prepare_manifests([self.source],NoPad(),self.root/'nopad',train_budgets=(40,),development_budget=20,
            test_budget=20,sequence_length=10,tokenizer_identity=self.artifact,model_identity=self.artifact)
        rows=load_manifest(self.root/'nopad','adapter_train_40')
        self.assertEqual(int(rows[0]['attention_mask'].sum()),3)
        self.assertIsNone(NoPad.pad_token_id)
    def test_minimum_independent_sources_applies_to_each_nested_role(self):
        with self.assertRaisesRegex(ValueError,'minimum independent source groups'):
            self.prepare(minimum_source_groups=2)
        self.assertFalse((self.root/'frozen/freeze.json').exists())
        self.assertTrue((self.root/'frozen/PREPARING.json').exists())
        index=self.prepare('valid',minimum_source_groups=1)
        self.assertEqual(index['minimum_source_groups'],1)
        self.assertTrue(all(v['unique_source_groups']>=1 for v in index['manifests'].values()))
    def test_resealed_below_minimum_source_count_refused(self):
        self.prepare();path=self.root/'frozen/freeze.json';index=json.loads(path.read_text())
        index['minimum_source_groups']=30;index.pop('freeze_sha256');index['freeze_sha256']=digest(index)
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,'independent source groups'):
            load_manifest(path,'adapter_train_40')
    def test_existing_output_never_overwritten(self):
        self.prepare(); before=(self.root/'frozen/freeze.json').read_bytes()
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertEqual((self.root/'frozen/freeze.json').read_bytes(),before)
    def test_missing_source_id_and_conflicting_document_rejected(self):
        del self.rows[0]['source_id']; self.write()
        with self.assertRaisesRegex(ValueError,'source_id'): self.prepare()
        self.rows[0]['source_id']='source-0'
        self.rows.append(dict(id='doc-0',source_id='other',text='new source content'));self.write()
        with self.assertRaisesRegex(ValueError,'conflicting'): self.prepare()
    def test_revision_without_file_hashes_is_not_actual_identity(self):
        with self.assertRaisesRegex(ValueError,'actual file bytes'): self.prepare(model_identity={'revision':'a'*40})
    def test_invalid_targets_and_tokenizer_shapes_rejected(self):
        for kw in (dict(train_budgets=(80,40)),dict(train_budgets=(0,)),dict(development_budget=0),dict(sequence_length=1)):
            with self.assertRaises(ValueError): self.prepare(**kw)
        class Broken:
            def __call__(self,*args,**kwargs): return dict(input_ids=[1]*257,attention_mask=[1]*257)
        with self.assertRaisesRegex(ValueError,'fixed-length'):
            prepare_manifests([self.source],Broken(),self.root/'bad',train_budgets=(40,),development_budget=20,
                test_budget=20,tokenizer_identity=self.artifact,model_identity=self.artifact)
    def test_sealed_test_refused_before_manifest_materialization(self):
        index=self.prepare(); (self.root/'frozen/test.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'sealed final test'): load_manifest(self.root/'frozen','test')
        with self.assertRaises(ValueError): load_manifest(self.root/'frozen')
        self.assertEqual(index['status'],'frozen')
    def test_index_tamper_rejected(self):
        self.prepare(); path=self.root/'frozen/freeze.json'; index=json.loads(path.read_text())
        index['seed']=99;path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,'index identity'): load_manifest(path,'adapter_train_40')
    def test_resealed_leakage_rejected(self):
        self.prepare(); path=self.root/'frozen/freeze.json';index=json.loads(path.read_text())
        index['manifests']['test']['source_groups']=index['manifests']['development_selection']['source_groups']
        index['manifests']['test']['unique_source_groups']=len(index['manifests']['test']['source_groups'])
        index.pop('freeze_sha256');index['freeze_sha256']=digest(index);path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,'leakage'): load_manifest(path,'adapter_train_40')
    @unittest.skipIf(torch is None,'torch required for token tensors')
    def test_lazy_rows_effective_position_count_and_tamper(self):
        index=self.prepare(); rows=load_manifest(self.root/'frozen','adapter_train_120')
        self.assertEqual(len(rows),index['manifests']['adapter_train_120']['unique_documents'])
        row=rows[0];self.assertEqual(row['input_ids'].shape,(1,10))
        self.assertEqual(sum(int(((r['attention_mask'][0,:-1]!=0)&(r['attention_mask'][0,1:]!=0)).sum()) for r in rows),rows.metadata['valid_positions'])
        path=rows.path; value=path.read_text();path.write_text(value.replace('source-', 'mutant-',1))
        with self.assertRaisesRegex(ValueError,'hash mismatch'): _=rows[0]
        with self.assertRaisesRegex(ValueError,'bytes/hash'): load_manifest(self.root/'frozen','adapter_train_120')


@unittest.skipIf(torch is None,'torch required for cache tests')
class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows=[dict(id=f'd-{i}',input_ids=torch.tensor([[1,2,3,0]]),attention_mask=torch.tensor([[1,1,1,0]])) for i in range(5)]
        self.binding=dict(freeze_sha256='f'*64,model='actual-files',tokenizer='actual-files',source='code-sha')
    def tearDown(self): self.temp.cleanup()
    def capture(self,row):
        from mixtral_lab.document_training import DocumentCache,_row
        ids,mask,targets,valid=_row(row)
        return DocumentCache(row['id'],torch.zeros(1,4,8),torch.ones(1,4,8),targets,valid,ids,mask)
    def test_roundtrip_and_one_document_materialization(self):
        cache=DiskDocumentCache(self.root/'cache',self.binding).build(self.rows,self.capture)
        self.assertEqual(len(cache),5);self.assertEqual(cache.identity['status'],'complete')
        doc=cache[0];self.assertEqual(doc.document_id,'d-0');self.assertTrue(torch.equal(doc.native_h1,torch.ones(1,4,8)))
        self.assertEqual(len(list((self.root/'cache').glob('*.pt'))),5)
        reopened=DiskDocumentCache(self.root/'cache',self.binding)
        self.assertEqual(reopened.identity,cache.identity)
        self.assertEqual(reopened.load_to(0,'cpu').h0.device.type,'cpu')
        with self.assertRaisesRegex(ValueError,'slicing forbidden'): _=cache[:2]
    def test_resume_skips_committed_captures_and_keeps_failed_attempts(self):
        seen=[]
        def interrupted(row):
            seen.append(row['id'])
            if row['id']=='d-2': raise RuntimeError('interrupted')
            return self.capture(row)
        cache=DiskDocumentCache(self.root/'cache',self.binding)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):cache.build(self.rows,interrupted)
        self.assertEqual(len(cache),2)
        resumed=[]
        cache.build(self.rows,lambda row:resumed.append(row['id']) or self.capture(row),resume=True)
        self.assertEqual(resumed,['d-2','d-3','d-4'])
        journal=[json.loads(s) for s in (self.root/'cache/attempts.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['event']=='capture_failed' for e in journal),1)
        self.assertEqual(sum(e['event']=='capture_committed' for e in journal),5)
    def test_unknown_orphan_preserved_and_never_reused(self):
        cache=DiskDocumentCache(self.root/'cache',self.binding)
        def fail(row): raise RuntimeError('before shard')
        with self.assertRaises(RuntimeError):cache.build(self.rows,fail)
        orphan=self.root/'cache/doc-00000000-0001.pt';orphan.write_bytes(b'unknown')
        cache.build(self.rows,self.capture,resume=True)
        self.assertEqual(orphan.read_bytes(),b'unknown')
        self.assertEqual(cache.index['entries'][0]['path'],'doc-00000000-0002.pt')
    def test_identity_tamper_and_input_order_changes_refused(self):
        cache=DiskDocumentCache(self.root/'cache',self.binding).build(self.rows,self.capture)
        with self.assertRaisesRegex(ValueError,'identity/bounds'):
            DiskDocumentCache(self.root/'cache',dict(self.binding,source='changed'))
        with self.assertRaisesRegex(ValueError,'identity/order'):
            cache.build(list(reversed(self.rows)),self.capture,resume=True)
        shard=self.root/'cache'/cache.index['entries'][0]['path'];shard.write_bytes(shard.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'changed'): _=cache[0]
        with self.assertRaisesRegex(ValueError,'byte/hash'): DiskDocumentCache(self.root/'cache',self.binding)
    def test_allocation_bound_and_wrong_capture_rejected(self):
        cache=DiskDocumentCache(self.root/'small',self.binding,max_document_bytes=16,max_shard_bytes=64)
        with self.assertRaisesRegex(ValueError,'tensor bytes exceed'): cache.build(self.rows,self.capture)
        self.assertEqual(len(cache),0)
        self.assertEqual(list((self.root/'small').glob('*.pt')),[])
        def wrong(row):
            c=self.capture(row);c.document_id='other';return c
        with self.assertRaisesRegex(ValueError,'wrong DocumentCache'):
            DiskDocumentCache(self.root/'wrong',self.binding).build(self.rows,wrong)
    def test_serialized_byte_bound_stops_write_and_preserves_failure(self):
        cache=DiskDocumentCache(self.root/'limited',self.binding,max_document_bytes=1024,max_shard_bytes=1024)
        with self.assertRaises((ValueError,RuntimeError)):
            cache.build(self.rows,self.capture)
        self.assertEqual(len(cache),0)
        self.assertTrue(all(p.stat().st_size<=1024 for p in (self.root/'limited').glob('*.pt')))
        journal=(self.root/'limited/attempts.jsonl').read_text()
        self.assertIn('capture_failed',journal)
    def test_capture_targets_and_nonfinite_hidden_refused(self):
        def bad_target(row):
            c=self.capture(row);c.targets[0]=999;return c
        with self.assertRaisesRegex(ValueError,'target/mask'):
            DiskDocumentCache(self.root/'targets',self.binding).build(self.rows,bad_target)
        def nonfinite(row):
            c=self.capture(row);c.native_h1.fill_(float('nan'));return c
        with self.assertRaisesRegex(ValueError,'hidden cache'):
            DiskDocumentCache(self.root/'nan',self.binding).build(self.rows,nonfinite)
    def test_no_implicit_overwrite_or_unknown_resume(self):
        cache=DiskDocumentCache(self.root/'cache',self.binding).build(self.rows,self.capture)
        with self.assertRaises(FileExistsError):cache.build(self.rows,self.capture)
        root=self.root/'unknown';root.mkdir();(root/'unknown.pt').write_bytes(b'unknown')
        with self.assertRaisesRegex(ValueError,'committed cache index'):
            DiskDocumentCache(root,self.binding).build(self.rows,self.capture,resume=True)
    def test_duplicate_source_document_refused(self):
        with self.assertRaisesRegex(ValueError,'duplicate cache source'):
            DiskDocumentCache(self.root/'cache',self.binding).build([self.rows[0],self.rows[0]],self.capture)
    def test_source_metadata_roundtrip_and_resume_source_change_refused(self):
        rows=copy.deepcopy(self.rows)
        for row in rows:
            row.update(source_id='shared-source',source_group_sha256='a'*64)
        cache=DiskDocumentCache(self.root/'cache',self.binding).build(rows,self.capture)
        self.assertEqual(cache.document_metadata(0),dict(document_id='d-0',source_id='shared-source',source_group_sha256='a'*64))
        changed=copy.deepcopy(rows);changed[0]['source_id']='changed-source'
        with self.assertRaisesRegex(ValueError,'source document/group metadata changed'):
            cache.build(changed,self.capture,resume=True)
        changed=copy.deepcopy(rows);changed[0]['source_group_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'source document/group metadata changed'):
            cache.build(changed,self.capture,resume=True)
        loaded=DiskDocumentCache(self.root/'cache',self.binding)
        self.assertEqual(loaded.document_metadata(-1)['source_group_sha256'],'a'*64)
        with self.assertRaises(IndexError):loaded.document_metadata(len(loaded))
    def test_cache_input_token_mutation_refused(self):
        def changed(row):
            c=self.capture(row);c.input_ids[0,0]=99;return c
        with self.assertRaisesRegex(ValueError,'token input changed'):
            DiskDocumentCache(self.root/'cache',self.binding).build(self.rows,changed)

if __name__=='__main__':unittest.main()
