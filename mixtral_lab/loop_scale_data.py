"""Frozen source-disjoint v2 datasets and bounded, restartable disk cache.

Preparation reads only explicitly supplied local JSONL. Token manifests, rather
than an unfrozen source path or a repeated epoch, define unique supervision.
Hidden tensors are committed one document per bounded disk shard and are never
collected into a corpus-sized Python list or accelerator allocation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import unicodedata

FORMAT = 'mixtral-loop-scale-manifests-v2'
CACHE_FORMAT = 'mixtral-loop-scale-cache-v2'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def local_artifact_identity(directory, *, required_files=()):
    """Hash actual local files, including weights; no revision-only attestation.

    Symlinks are resolved for reading. No download, mutation, or model loading.
    A caller must freeze model and tokenizer directories independently.
    """
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('local artifact directory required')
    def snapshot():
        result = {}
        for path in sorted(root.rglob('*')):
            if path.is_file() and '.git' not in path.relative_to(root).parts:
                st = path.stat()
                result[str(path.relative_to(root))] = (st.st_dev, st.st_ino, st.st_size,
                                                       st.st_mtime_ns, st.st_ctime_ns)
        return result
    before = snapshot()
    entries = {rel: dict(bytes=signature[2], sha256=sha256_file(root/rel))
               for rel, signature in before.items()}
    if snapshot() != before:
        raise ValueError('local artifact files changed during actual byte hashing')
    if not entries or any(name not in entries for name in required_files):
        raise ValueError('artifact missing required actual files')
    return dict(path=str(root), files=entries, files_sha256=digest(entries))


def _artifact(identity, label):
    if not isinstance(identity, dict) or not isinstance(identity.get('files'), dict) or not identity['files']:
        raise ValueError(label + ' actual file bytes identities required')
    for name, item in identity['files'].items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise ValueError('invalid artifact identity')
        if type(item.get('bytes')) is not int or item['bytes'] < 0:
            raise ValueError('invalid artifact byte count')
        if not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']):
            raise ValueError('invalid artifact SHA256')
    if identity.get('files_sha256') != digest(identity['files']):
        raise ValueError(label + ' actual file identity hash mismatch')


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be positive integer')


def _normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def _tokenize(tokenizer, text, length):
    encoded = tokenizer(text, add_special_tokens=True, truncation=True,
                        max_length=length, padding=False, return_attention_mask=True)
    ids, mask = encoded['input_ids'], encoded['attention_mask']
    if hasattr(ids, 'tolist'): ids = ids.tolist()
    if hasattr(mask, 'tolist'): mask = mask.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    if mask and isinstance(mask[0], list): mask = mask[0]
    if not 0 <= len(ids) <= length or len(mask) != len(ids):
        raise ValueError('tokenizer must return one bounded fixed-length document')
    if len(ids) < length:
        pad = getattr(tokenizer, 'pad_token_id', None)
        if pad is None: pad = getattr(tokenizer, 'eos_token_id', None)
        if type(pad) is not int or pad < 0: raise ValueError('tokenizer needs local pad or eos token for manual padding')
        mask = mask + [0] * (length-len(ids))
        ids = ids + [pad] * (length-len(ids))
    if any(type(i) is not int or i < 0 for i in ids) or any(i not in (0, 1) for i in mask):
        raise ValueError('invalid tokenizer ids/mask')
    positions = sum(bool(a and b) for a, b in zip(mask[:-1], mask[1:]))
    return ids, mask, positions


def prepare_manifests(source_files, tokenizer, output_dir, *,
                      train_budgets=(100000, 300000, 1000000),
                      development_budget=100000, test_budget=100000,
                      sequence_length=256, seed=17,
                      tokenizer_identity=None, model_identity=None,
                      excluded_source_ids=(), excluded_content_hashes=(), excluded_document_ids=(),
                      max_source_bytes=512*1024**2, max_row_bytes=1024**2, max_source_rows=100000,
                      minimum_source_groups=1):
    """Freeze 90% adapter / 10% source-held-out gate supervision per total budget.

    Each JSONL row requires explicit id, source_id, text. Duplicate normalized
    contents connect their source groups before splitting. Whole documents form
    each nested prefix; actual positions may exceed a target by at most S-2.
    Unused documents from an assigned source group are discarded, never moved
    to another role to fill a shortage. Insufficient source-disjoint data fail.
    """
    budgets = tuple(train_budgets)
    if not budgets or list(budgets) != sorted(set(budgets)):
        raise ValueError('strictly increasing unique train budgets required')
    for n in budgets: _positive(n, 'train budget')
    for name, n in [('development budget', development_budget), ('test budget', test_budget)]:
        _positive(n, name)
    if type(sequence_length) is not int or not 2 <= sequence_length <= 4096:
        raise ValueError('sequence_length must be in [2,4096]')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('invalid seed')
    _artifact(tokenizer_identity, 'tokenizer'); _artifact(model_identity, 'model')
    _positive(minimum_source_groups,'minimum_source_groups')
    for value,name in ((max_source_bytes,'max_source_bytes'),(max_row_bytes,'max_row_bytes'),(max_source_rows,'max_source_rows')):
        _positive(value,name)
    paths = [Path(p).resolve(strict=True) for p in source_files]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('distinct explicit local source files required')
    if sum(p.stat().st_size for p in paths) > max_source_bytes:
        raise ValueError('aggregate explicit source bytes exceed bounded preparation budget')
    output = Path(output_dir).resolve()
    # Reject output inside immutable inputs/artifact directories before writing.
    for identity in (tokenizer_identity, model_identity):
        if identity.get('path') and output.is_relative_to(Path(identity['path']).resolve()):
            raise ValueError('output cannot modify local model/tokenizer artifacts')
    if output.exists(): raise FileExistsError(output)
    parent = {}; raw = []; by_id = {}; by_content = {}; duplicates = 0; input_rows = 0
    exclusions = dict(source_ids=sorted(set(excluded_source_ids)),
                      content_hashes=sorted(set(excluded_content_hashes)),
                      document_ids=sorted(set(excluded_document_ids)))
    if any(not isinstance(x, str) or not x for values in exclusions.values() for x in values):
        raise ValueError('nonempty string exclusion identities required')
    excluded_group_sources = set(exclusions['source_ids'])
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    sources = []
    for path in paths:
        if not path.is_file(): raise ValueError('local JSONL file required')
        sources.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path)))
        with path.open('rb') as f:
            number = 0
            while True:
                line = f.readline(max_row_bytes+1)
                if not line: break
                number += 1
                if len(line) > max_row_bytes: raise ValueError('source JSONL row exceeds bounded preparation bytes')
                if not line.strip(): continue
                if input_rows >= max_source_rows: raise ValueError('source rows exceed bounded preparation count')
                row = json.loads(line); input_rows += 1
                if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('id', 'source_id', 'text')):
                    raise ValueError(f'{path}:{number}: explicit nonempty id/source_id/text required')
                text = _normalize(row['text'])
                if not text: raise ValueError('normalized empty content')
                content = hashlib.sha256(text.encode()).hexdigest()
                raw_content = hashlib.sha256(row['text'].encode()).hexdigest()
                legacy_content = hashlib.sha256(' '.join(row['text'].split()).encode()).hexdigest()
                source = row['source_id']; find(source)
                signature = (source, content)
                if row['id'] in exclusions['document_ids'] or content in exclusions['content_hashes'] or raw_content in exclusions['content_hashes'] or legacy_content in exclusions['content_hashes']:
                    excluded_group_sources.add(source)
                if row['id'] in by_id and by_id[row['id']] != signature:
                    raise ValueError('same document id has conflicting source/content')
                by_id[row['id']] = signature
                if content in by_content:
                    union(source, by_content[content]['source_id']); duplicates += 1
                else:
                    kept = dict(id=row['id'], source_id=source, content_sha256=content,
                                raw_content_sha256=raw_content, text=row['text'])
                    by_content[content] = kept; raw.append(kept)
    groups = {}; unusable = 0; excluded_documents = 0; effective_duplicates = 0
    tokenized = []; by_effective = {}
    # Identical actual truncated contexts are also duplicates, even when the raw
    # suffix differs. Union their source groups before exclusion and split.
    for row in raw:
        ids, mask, positions = _tokenize(tokenizer, row.pop('text'), sequence_length)
        if not positions: unusable += 1; continue
        effective = digest(dict(input_ids=ids, attention_mask=mask))
        if effective in by_effective:
            union(row['source_id'], by_effective[effective]['source_id'])
            effective_duplicates += 1; continue
        row.update(input_ids=ids, attention_mask=mask, valid_positions=positions,
                   effective_tokens_sha256=effective)
        by_effective[effective] = row; tokenized.append(row)
    excluded_roots = {find(source) for source in excluded_group_sources}
    for row in tokenized:
        if find(row['source_id']) in excluded_roots:
            excluded_documents += 1; continue
        row['source_group_sha256'] = hashlib.sha256(find(row['source_id']).encode()).hexdigest()
        row['record_sha256'] = digest(row)
        groups.setdefault(find(row['source_id']), []).append(row)
    for source in sources:
        if Path(source['path']).stat().st_size != source['bytes'] or sha256_file(source['path']) != source['sha256']:
            raise ValueError('source changed during manifest preparation')
    order = sorted(groups, key=lambda s: hashlib.sha256(f'{seed}:source:{s}'.encode()).hexdigest())
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256(f"{seed}:document:{r['id']}".encode()).hexdigest())
    targets = dict(development_calibration=development_budget//2,
                   development_selection=development_budget-development_budget//2, test=test_budget,
                   gate_fit=budgets[-1] - (9 * budgets[-1] // 10),
                   adapter_train=9 * budgets[-1] // 10)
    if min(targets.values()) < 1: raise ValueError('training budgets must allocate adapter and gate positions')
    selected = {}; cursor = 0; discarded = 0
    for role, target in targets.items():
        rows = []; count = 0
        while count < target and cursor < len(order):
            group = groups[order[cursor]]; cursor += 1
            for i, row in enumerate(group):
                if count >= target:
                    discarded += len(group) - i; break
                rows.append(row); count += row['valid_positions']
        if count < target:
            raise ValueError(f'insufficient source-disjoint unique positions for {role}: {count} < {target}')
        selected[role] = rows
    output.mkdir(parents=True)
    _atomic_json(output / 'PREPARING.json', dict(format=FORMAT, status='incomplete', sources=sources))
    manifests = {}
    def write(role, rows, target, family):
        selected_groups = {r['source_group_sha256'] for r in rows}
        if len(selected_groups) < minimum_source_groups:
            raise ValueError(f'{role} has fewer than minimum independent source groups: {len(selected_groups)} < {minimum_source_groups}')
        path = output / (role + '.jsonl')
        with path.open('x', encoding='utf8') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n')
            f.flush(); os.fsync(f.fileno())
        count = sum(r['valid_positions'] for r in rows)
        manifests[role] = dict(path=path.name, sha256=sha256_file(path), bytes=path.stat().st_size,
            role=role, family=family, target_positions=target, valid_positions=count,
            overshoot_positions=count-target, unique_documents=len(rows),
            unique_source_documents=len({r['source_id'] for r in rows}),
            unique_source_groups=len({r['source_group_sha256'] for r in rows}),
            source_groups=sorted({r['source_group_sha256'] for r in rows}),
            effective_token_hashes=[r['effective_tokens_sha256'] for r in rows],
            document_ids=[r['id'] for r in rows], content_hashes=[r['content_sha256'] for r in rows],
            record_hashes=[r['record_sha256'] for r in rows])
    for role in ('development_calibration', 'development_selection', 'test'):
        write(role, selected[role], targets[role], role)
    for budget in budgets:
        for family, target in [('adapter_train', 9 * budget // 10), ('gate_fit', budget - 9 * budget // 10)]:
            rows = []; count = 0
            for row in selected[family]:
                if count >= target: break
                rows.append(row); count += row['valid_positions']
            write(f'{family}_{budget}', rows, target, family)
    index = dict(format=FORMAT, status='frozen', seed=seed, sequence_length=sequence_length,
        minimum_source_groups=minimum_source_groups,
        train_budgets=list(budgets), allocation='total training target: 90% adapter, 10% source-disjoint gate fit',
        preparation_bounds=dict(max_source_bytes=max_source_bytes,max_row_bytes=max_row_bytes,max_source_rows=max_source_rows),
        sources=sources, exclusions=exclusions, exclusions_sha256=digest(exclusions), tokenizer=tokenizer_identity, model=model_identity, manifests=manifests,
        counts=dict(input_rows=input_rows, duplicate_rows_removed=duplicates, excluded_documents=excluded_documents,
                    duplicate_effective_sequences_removed=effective_duplicates,
                    unique_normalized_content=len(raw), unusable_documents=unusable,
                    unused_documents_in_assigned_groups=discarded,
                    unassigned_source_groups=len(order)-cursor),
        test_policy='sealed; default readers refuse test; no model/checkpoint/threshold selection on test')
    index['freeze_sha256'] = digest(index)
    _validate_index(index)
    _atomic_json(output / 'freeze.json', index)
    (output / 'PREPARING.json').unlink()
    return index


def _validate_index(index):
    value = dict(index); claimed = value.pop('freeze_sha256', None)
    if claimed != digest(value) or value.get('format') != FORMAT or value.get('status') != 'frozen':
        raise ValueError('manifest index identity/status mismatch')
    _artifact(index['model'], 'model'); _artifact(index['tokenizer'], 'tokenizer')
    if digest(index.get('exclusions')) != index.get('exclusions_sha256'):
        raise ValueError('exclusions identity mismatch')
    _positive(index.get('minimum_source_groups'),'minimum_source_groups')
    families = {}
    for role, item in index['manifests'].items():
        if item['role'] != role or Path(item['path']).name != item['path']:
            raise ValueError('unsafe manifest role/path')
        n = item['unique_documents']
        if (item.get('unique_source_groups') != len(set(item['source_groups']))
                or len(item['source_groups']) != len(set(item['source_groups']))
                or item['unique_source_groups'] < index['minimum_source_groups']):
            raise ValueError('manifest has insufficient or inconsistent independent source groups')
        if any(len(item[k]) != n for k in ('document_ids', 'content_hashes', 'record_hashes', 'effective_token_hashes')):
            raise ValueError('manifest count metadata mismatch')
        if len(set(item['document_ids'])) != n or len(set(item['content_hashes'])) != n or len(set(item['effective_token_hashes'])) != n:
            raise ValueError('manifest duplicate documents/content')
        if item['valid_positions'] < item['target_positions'] or not 0 <= item['overshoot_positions'] < index['sequence_length']-1:
            raise ValueError('invalid document-bounded token overshoot')
        prior = families.get(item['family'])
        if prior is None or prior['target_positions'] < item['target_positions']:
            families[item['family']] = item
    expected = {'adapter_train', 'gate_fit', 'development_calibration', 'development_selection', 'test'}
    if set(families) != expected: raise ValueError('missing isolated data role')
    for family in ('adapter_train', 'gate_fit'):
        ordered = sorted((v for v in index['manifests'].values() if v['family'] == family), key=lambda v:v['target_positions'])
        for a, b in zip(ordered, ordered[1:]):
            for k in ('document_ids', 'content_hashes', 'record_hashes', 'effective_token_hashes'):
                if b[k][:len(a[k])] != a[k]: raise ValueError('training manifests are not nested prefixes')
    keys = list(families)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            for name in ('document_ids', 'content_hashes', 'source_groups', 'effective_token_hashes'):
                if set(families[a][name]) & set(families[b][name]):
                    raise ValueError('source/document/content leakage across data roles')


class ManifestRows(Sequence):
    """Token-only lazy rows; a single access materializes one small document."""
    def __init__(self, path, metadata, freeze_sha256):
        self.path = Path(path); self.metadata = metadata
        self.identity = dict(freeze_sha256=freeze_sha256, role=metadata['role'],
                             manifest_sha256=metadata['sha256'],
                             valid_positions=metadata['valid_positions'],
                             unique_documents=metadata['unique_documents'],
                             unique_source_documents=metadata['unique_source_documents'],
                             unique_source_groups=metadata['unique_source_groups'])
        if self.path.stat().st_size != metadata['bytes'] or sha256_file(self.path) != metadata['sha256']:
            raise ValueError('token manifest bytes/hash mismatch')
        self._offsets = []
        with self.path.open('rb') as f:
            while True:
                offset = f.tell(); line = f.readline()
                if not line: break
                self._offsets.append(offset)
        if len(self._offsets) != metadata['unique_documents']:
            raise ValueError('token manifest document count mismatch')
    def __len__(self): return len(self._offsets)
    def __getitem__(self, index):
        import torch
        if isinstance(index, slice): return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        with self.path.open('rb') as f:
            f.seek(self._offsets[index]); row = json.loads(f.readline())
        check = dict(row); claimed = check.pop('record_sha256', None)
        if claimed != self.metadata['record_hashes'][index] or digest(check) != claimed:
            raise ValueError('token row hash mismatch')
        if row['id'] != self.metadata['document_ids'][index] or row['content_sha256'] != self.metadata['content_hashes'][index]:
            raise ValueError('token row identity mismatch')
        ids = torch.tensor([row['input_ids']], dtype=torch.long)
        mask = torch.tensor([row['attention_mask']], dtype=torch.long)
        positions = int(((mask[0,:-1] != 0) & (mask[0,1:] != 0)).sum())
        if positions != row['valid_positions']: raise ValueError('supervision position count mismatch')
        return dict(id=row['id'], source_id=row['source_id'], source_group_sha256=row['source_group_sha256'],
                    content_sha256=row['content_sha256'], input_ids=ids, attention_mask=mask)


def load_manifest(path, role=None, *, allow_test=False):
    """Read a frozen index and one explicitly requested role; test refuses by default."""
    path = Path(path)
    if path.is_dir(): path = path / 'freeze.json'
    if (path.parent / 'PREPARING.json').exists():
        raise ValueError('manifest preparation is incomplete; do not treat as frozen')
    with path.open(encoding='utf8') as f: index = json.load(f)
    _validate_index(index)
    if role is None: raise ValueError('explicit manifest role required')
    if role not in index['manifests']: raise ValueError('unknown frozen manifest role')
    item = index['manifests'][role]
    if item['family'] == 'test' and not allow_test:
        raise ValueError('sealed final test forbidden in preparation/training/development')
    return ManifestRows(path.parent / item['path'], item, index['freeze_sha256'])


class DiskDocumentCache(Sequence):
    """One document per hashed bounded shard, with an atomic committed index.

    Unknown/partial shards are never accepted as completed. The durable attempt
    journal retains pending or failed captures. Resume skips committed documents
    and verifies exact input/config/model/tokenizer identity before any capture.
    """
    def __init__(self, root, binding, *, max_shard_bytes=64*1024**2, max_document_bytes=16*1024**2):
        self.root = Path(root); self.binding = binding
        for value, name in [(max_shard_bytes, 'max_shard_bytes'), (max_document_bytes, 'max_document_bytes')]:
            _positive(value, name)
        if max_document_bytes > max_shard_bytes: raise ValueError('document bound exceeds shard bound')
        self.max_shard_bytes = max_shard_bytes; self.max_document_bytes = max_document_bytes
        self.index = None
        if (self.root / 'index.json').exists(): self._read_index()
    def _read_index(self):
        with (self.root/'index.json').open() as f: index = json.load(f)
        original = dict(index); claimed = original.pop('index_sha256', None)
        if claimed != digest(original) or index.get('format') != CACHE_FORMAT:
            raise ValueError('cache index digest/format mismatch')
        if index.get('binding') != self.binding or index.get('max_document_bytes') != self.max_document_bytes or index.get('max_shard_bytes') != self.max_shard_bytes:
            raise ValueError('cache identity/bounds mismatch')
        if len({e['document_id'] for e in index['entries']}) != len(index['entries']):
            raise ValueError('duplicate cache document')
        for e in index['entries']:
            if _source_metadata(dict(id=e['document_id'], source_id=e.get('source_id'),
                                     source_group_sha256=e.get('source_group_sha256'))) != {
                    k:e.get(k) for k in ('document_id','source_id','source_group_sha256')}:
                raise ValueError('invalid cache source metadata')
            if Path(e['path']).name != e['path'] or not 0 < e['bytes'] <= self.max_shard_bytes:
                raise ValueError('invalid bounded cache shard')
            path = self.root/e['path']
            if not path.is_file() or path.stat().st_size != e['bytes'] or sha256_file(path) != e['sha256']:
                raise ValueError('cache shard byte/hash mismatch')
        self.index = index
    def _commit(self):
        self.index.pop('index_sha256', None)
        self.index['index_sha256'] = digest(self.index)
        _atomic_json(self.root/'index.json', self.index)
    def _journal(self, event):
        with (self.root/'attempts.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    def build(self, rows, capture_fn, *, resume=False):
        import torch
        from .document_training import DocumentCache, _row
        if not isinstance(rows, Sequence) or not len(rows): raise ValueError('nonempty lazy document sequence required')
        if self.root.exists() and not resume: raise FileExistsError(self.root)
        if resume and not (self.root/'index.json').exists():
            raise ValueError('resume requires a committed cache index; retain unknown directory for diagnosis')
        self.root.mkdir(parents=True, exist_ok=resume)
        if resume: self._read_index()
        else:
            self.index = dict(format=CACHE_FORMAT, binding=self.binding, status='building',
                max_shard_bytes=self.max_shard_bytes, max_document_bytes=self.max_document_bytes,
                entries=[], expected_documents=len(rows), attempt_counter=0)
            self._commit()
        if self.index['expected_documents'] != len(rows): raise ValueError('cache input count changed')
        seen = set()
        for i, row in enumerate(rows):
            if row['id'] in seen: raise ValueError('duplicate cache source document id')
            seen.add(row['id'])
            input_identity = _row_identity(row)
            source_metadata = _source_metadata(row)
            if i < len(self.index['entries']):
                if self.index['entries'][i]['input_sha256'] != input_identity or self.index['entries'][i]['document_id'] != row['id']:
                    raise ValueError('resume row identity/order changed')
                if any(self.index['entries'][i].get(k) != v for k,v in source_metadata.items()):
                    raise ValueError('resume source document/group metadata changed')
                continue
            # Each attempt writes a new shard name. A crash orphan is preserved,
            # never overwritten or counted as a completed prefix capture.
            attempt = self.index['attempt_counter'] + 1
            self.index['attempt_counter'] = attempt
            self._commit()  # durable reservation survives a capture interrupted before file creation
            name = f'doc-{i:08d}-{attempt:04d}.pt'; path = self.root/name
            self._journal(dict(event='capture_reserved', index=i, document_id=row['id'], shard=name))
            try:
                cache = capture_fn(row)
                if not isinstance(cache, DocumentCache) or cache.document_id != row['id']:
                    raise ValueError('capture returned wrong DocumentCache')
                payload = {'document_id':cache.document_id}
                total_bytes = 0
                _, _, expected_targets, expected_valid = _row(row)
                if not torch.equal(cache.targets.cpu(), expected_targets) or not torch.equal(cache.valid_mask.cpu(), expected_valid):
                    raise ValueError('capture target/mask differs from source tokens')
                expected_shape = (*row['input_ids'].shape, cache.native_h1.shape[-1])
                for hidden_name in ('h0', 'native_h1'):
                    tensor = getattr(cache, hidden_name)
                    if tensor.shape != expected_shape or tensor.dtype not in (torch.float32, torch.bfloat16) or not bool(torch.isfinite(tensor).all()):
                        raise ValueError('invalid full-document hidden cache')
                for field in fields(DocumentCache):
                    if field.name == 'document_id': continue
                    tensor = getattr(cache, field.name)
                    if not isinstance(tensor, torch.Tensor): raise ValueError('all document cache tensors required')
                    total_bytes += tensor.numel()*tensor.element_size()
                    if total_bytes > self.max_document_bytes: raise ValueError('document tensor bytes exceed bound')
                    payload[field.name] = tensor.detach().cpu().clone()
                del cache
                if _row_identity(dict(id=payload['document_id'],input_ids=payload['input_ids'],attention_mask=payload['attention_mask'])) != input_identity:
                    raise ValueError('cache token input changed during capture')
                with path.open('xb') as f:
                    torch.save(payload, _LimitedWriter(f, self.max_shard_bytes))
                    f.flush(); os.fsync(f.fileno())
                if path.stat().st_size > self.max_shard_bytes: raise ValueError('serialized shard exceeds bound')
                self.index['entries'].append(dict(document_id=row['id'], input_sha256=input_identity,
                    source_id=source_metadata['source_id'], source_group_sha256=source_metadata['source_group_sha256'],
                    path=name, bytes=path.stat().st_size, sha256=sha256_file(path), tensor_bytes=total_bytes))
                self._commit()
                self._journal(dict(event='capture_committed', index=i, document_id=row['id'], shard=name))
            except BaseException as error:
                self._journal(dict(event='capture_failed', index=i, document_id=row['id'], shard=name,
                                   error_type=type(error).__name__, error=str(error)))
                raise
        self.index['status'] = 'complete'; self._commit()
        return self
    @property
    def identity(self):
        if self.index is None: raise ValueError('cache not built')
        return dict(index_sha256=self.index['index_sha256'], binding=self.binding,
                    documents=len(self.index['entries']), status=self.index['status'])
    def __len__(self): return len(self.index['entries']) if self.index else 0
    def __getitem__(self, index):
        import torch
        from .document_training import DocumentCache
        if isinstance(index, slice): raise ValueError('cache slicing forbidden: stream one document at a time')
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        entry = self.index['entries'][index]; path = self.root/entry['path']
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError('cache shard changed since index validation')
        with path.open('rb') as f: payload = torch.load(f, map_location='cpu', weights_only=True)
        if _row_identity(dict(id=payload['document_id'],input_ids=payload['input_ids'],attention_mask=payload['attention_mask'])) != entry['input_sha256']:
            raise ValueError('cache shard input identity mismatch')
        return DocumentCache(**payload)
    def document_metadata(self, index):
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        entry = self.index['entries'][index]
        return {key:entry[key] for key in ('document_id','source_id','source_group_sha256')}
    def load_to(self, index, device):
        cache = self[index]
        cache.h0 = cache.h0.to(device); cache.native_h1 = cache.native_h1.to(device)
        return cache


def _row_identity(row):
    ids = row['input_ids'].detach().cpu().tolist()
    mask = row['attention_mask'].detach().cpu().tolist()
    return digest(dict(id=row['id'], input_ids=ids, attention_mask=mask))


class _LimitedWriter:
    def __init__(self, file, maximum):
        self.file = file; self.maximum = maximum
    def write(self, data):
        if self.file.tell() + len(data) > self.maximum:
            raise ValueError('serialized shard exceeds bound')
        return self.file.write(data)
    def flush(self): return self.file.flush()


def _source_metadata(row):
    document_id = row['id']
    source_id = row.get('source_id', document_id)
    if not isinstance(document_id, str) or not document_id or not isinstance(source_id, str) or not source_id:
        raise ValueError('invalid source document identity')
    group = row.get('source_group_sha256', hashlib.sha256(source_id.encode()).hexdigest())
    if not isinstance(group, str) or len(group) != 64 or any(c not in '0123456789abcdef' for c in group):
        raise ValueError('invalid source group digest')
    return dict(document_id=document_id, source_id=source_id, source_group_sha256=group)
