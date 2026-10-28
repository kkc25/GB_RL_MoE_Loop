"""Offline inputs and bounded, disjoint development protocol for Mixtral."""
from __future__ import annotations
import hashlib
import io
import json
import math
import re
from pathlib import Path
import torch
import torch.nn.functional as F

SCHEMA = 'mixtral-frozen-routing-v1'
ROLES = ('train', 'selection', 'calibration')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'invalid {name}: expected integer {minimum}..{maximum}')
    return value


def load_config(path, *, stage=None, actor_checkpoint=None):
    config = json.loads(Path(path).read_text())
    validate_config(config, stage=stage, actor_checkpoint=actor_checkpoint)
    return config


def validate_config(c, *, stage=None, actor_checkpoint=None):
    # Standalone validation remains strict; CLI stages only require caches they use.
    if stage not in (None, 'validate', 'profile', 'train', 'oracle', 'adapter'):
        raise ValueError('unknown Mixtral experiment stage')
    needs_train_contexts = (stage is None or stage in ('train', 'adapter')
                            or (stage == 'profile' and bool(actor_checkpoint)))
    if c.get('schema') != SCHEMA:
        raise ValueError('unknown Mixtral experiment schema')
    for field in ('model_path', 'tokenizer_path', 'output_dir'):
        if not isinstance(c.get(field), str) or not c[field].strip():
            raise ValueError(f'{field} must be a nonempty local path')
    revision = c.get('model_revision', '')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('model_revision must declare the pinned 40-character upstream commit; local bytes are independently hashed')
    if c.get('dtype') != 'bfloat16':
        raise ValueError('this experiment requires BF16; quantization is a separate protocol')
    if c.get('experts_implementation') not in ('eager', 'grouped_mm'):
        raise ValueError('only eager or local PyTorch grouped_mm expert backends are supported')
    if c.get('device_map') not in ('auto', 'balanced'):
        raise ValueError('device_map must be auto or balanced')
    memory = c.get('max_memory')
    if not isinstance(memory, dict) or not memory:
        raise ValueError('max_memory must explicitly bound GPU placement')
    for device, amount in memory.items():
        if not str(device).isdecimal() or not isinstance(amount, str) or not re.fullmatch(r'[1-9][0-9]*(GiB|GB)', amount):
            raise ValueError('max_memory accepts explicit GPU indices and positive GiB/GB strings only; no CPU/disk offload')
    for key, lower, upper in (('sequence_length', 2, 4096), ('max_documents_per_split', 1, 4096),
                              ('max_text_characters', 1, 1_000_000), ('max_train_contexts', 1, 4096),
                              ('warmup', 0, 100), ('repeats', 1, 20), ('measurements', 1, 1000)):
        integer(c.get(key), key, lower, upper)
    counts = c.get('sequences')
    if not isinstance(counts, dict) or set(counts) != set(ROLES):
        raise ValueError('sequences must specify exactly train, selection, calibration')
    for role in ROLES:
        integer(counts[role], f'sequences.{role}', 1, c['max_documents_per_split'])
    if needs_train_contexts and c['max_train_contexts'] > counts['train'] * (c['sequence_length'] - 1):
        raise ValueError('max_train_contexts exceeds the configured train token upper bound')
    files = c.get('data_files')
    if not isinstance(files, dict) or set(files) != set(ROLES):
        raise ValueError('data_files must specify exactly development roles; newtest is not accepted')
    resolved = []
    for role, name in files.items():
        if not isinstance(name, str) or not name.strip() or 'newtest' in name.casefold():
            raise ValueError('invalid development data path or forbidden newtest path')
        resolved.append(str(Path(name).resolve()))
    if len(set(resolved)) != len(resolved):
        raise ValueError('development roles must use different files')
    for key in ('nll_tolerance', 'ce_mean_abs_tolerance', 'logit_max_abs_tolerance'):
        value = c.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not isinstance(c.get('allow_tf32'), bool):
        raise ValueError('allow_tf32 must be boolean')
    integer(c.get('seed'), 'seed', 0, 2**32 - 1)
    from mixtral_lab.training import RouterTrainingConfig
    training = RouterTrainingConfig(**c.get('training', {}))
    training.validate()
    if training.max_train_contexts != c['max_train_contexts']:
        raise ValueError('training and runner max_train_contexts must match')
    sampler = c.setdefault('context_sampling', {'version': 1, 'strategy': 'document_balanced_uniform_positions'})
    if sampler != {'version': 1, 'strategy': 'document_balanced_uniform_positions'} or type(sampler.get('version')) is not int:
        raise ValueError('unsupported context_sampling contract; legacy first-N checkpoint reuse is forbidden')
    from mixtral_lab.oracle import OracleConfig
    oracle = c.setdefault('oracle', {'role': 'calibration', 'max_contexts': 8, 'max_score_calls': 464})
    if not isinstance(oracle, dict):
        raise ValueError('oracle must be an object')
    settings = dict(oracle)
    role = settings.pop('role', 'calibration')
    if role not in ('calibration', 'selection'):
        raise ValueError('oracle role must be calibration or selection, never held-out')
    oc = OracleConfig(**settings); oc.validate()
    integer(oc.max_contexts, 'oracle.max_contexts', 1, 4096)
    integer(oc.max_score_calls, 'oracle.max_score_calls', 1, 237568)
    if oc.max_score_calls < 58 * oc.max_contexts:
        raise ValueError('oracle score budget must cover 58 score calls per context')
    second = c.setdefault('second_round', {'enabled': False})
    if not isinstance(second, dict) or not isinstance(second.get('enabled'), bool):
        raise ValueError('second_round.enabled must be explicitly boolean')
    from mixtral_lab.second_round import SecondRoundConfig
    settings = dict(second); settings.pop('enabled')
    sc = SecondRoundConfig(**settings); sc.validate()
    if second['enabled'] and sc.max_train_contexts != c['max_train_contexts']:
        raise ValueError('second_round and runner max_train_contexts must match')
    return c


def directory_identity(directory):
    """Hash actual local artifacts, including weights. No network or remote-code load."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('model/tokenizer must already exist in a local directory')
    allowed = {'.json', '.safetensors', '.model', '.txt', '.tiktoken'}
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in allowed and '.cache' not in p.relative_to(root).parts}
    if not files:
        raise ValueError('local artifact directory is empty')
    return {'directory': str(root.resolve()), 'files': files, 'content_sha256': canonical_hash(files)}


def read_documents(config):
    """Validate all bounded source files before selecting or tokenizing any rows."""
    documents, identities, ids, content = {}, {}, {}, {}
    for role in ROLES:
        path = Path(config['data_files'][role])
        rows = []
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError('development JSONL file exceeds 128 MiB safety bound')
        source_bytes = path.read_bytes()
        if len(source_bytes) > 128 * 1024 * 1024:
            raise ValueError('development JSONL file exceeds safety bound')
        with io.StringIO(source_bytes.decode('utf-8')) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if len(rows) >= config['max_documents_per_split']:
                    raise ValueError('document file exceeds declared safety limit; create an explicit bounded input manifest')
                row = json.loads(line)
                ident, text = row.get('id'), row.get('text')
                if not isinstance(ident, str) or not ident.strip() or not isinstance(text, str) or not text.strip():
                    raise ValueError(f'{role}:{line_number} requires nonempty string id and text')
                if len(text) > config['max_text_characters']:
                    raise ValueError('document exceeds character limit')
                digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
                if ident in ids or digest in content:
                    raise ValueError('duplicate document id or normalized content within/across development roles')
                ids[ident], content[digest] = role, role
                rows.append({'id': ident, 'text': text, 'text_sha256': digest})
        if len(rows) < config['sequences'][role]:
            raise ValueError(f'{role} has fewer documents than requested sequences')
        identities[role] = {'path': str(path.resolve()), 'file_sha256': hashlib.sha256(source_bytes).hexdigest(),
                            'documents': [{'id': row['id'], 'text_sha256': row['text_sha256']} for row in rows],
                            'selected_document_ids': [row['id'] for row in rows[:config['sequences'][role]]],
                            'declared_role': role}
        documents[role] = rows[:config['sequences'][role]]
    return documents, identities


def prepare_data(config, tokenizer):
    documents, identities = read_documents(config)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError('tokenizer needs pad or eos token')
    result, effective_sequences = {}, {}
    for role, rows in documents.items():
        prepared = []
        within_role_sequences = {}
        for row in rows:
            ids = tokenizer.encode(row['text'], add_special_tokens=True,
                                   truncation=True, max_length=config['sequence_length'])
            if len(ids) < 2:
                raise ValueError('document has fewer than two tokens after tokenization')
            # Compare actual non-padding model inputs, not IDs or raw text tails
            # discarded by truncation. Retain raw-document checks above as well.
            effective = tuple(ids)
            previous = effective_sequences.get(effective)
            if previous is not None and previous[0] != role:
                raise ValueError('duplicate effective token sequence across development roles: '
                                 f'{previous[0]}:{previous[1]} and {role}:{row["id"]}')
            effective_sequences[effective] = (role, row['id'])
            within_role_sequences.setdefault(effective, []).append(row['id'])
            valid = len(ids)
            ids = ids + [pad] * (config['sequence_length'] - valid)
            prepared.append({'id': row['id'], 'input_ids': torch.tensor(ids, dtype=torch.long),
                             'attention_mask': torch.tensor([1]*valid+[0]*(len(ids)-valid), dtype=torch.long)})
        token_identity = [{'id': row['id'], 'input_ids': row['input_ids'].tolist(),
                           'attention_mask': row['attention_mask'].tolist()} for row in prepared]
        identities[role]['tokenized_sha256'] = canonical_hash(token_identity)
        identities[role]['supervised_tokens'] = sum(int((row['attention_mask'][:-1] * row['attention_mask'][1:]).sum()) for row in prepared)
        duplicate_groups = [
            {'effective_tokens_sha256': canonical_hash(list(tokens)),
             'document_ids': doc_ids, 'effective_token_count': len(tokens)}
            for tokens, doc_ids in within_role_sequences.items() if len(doc_ids) > 1
        ]
        unique_count = len(within_role_sequences)
        identities[role]['effective_input_statistics'] = {
            'documents': len(prepared), 'unique_effective_sequences': unique_count,
            'redundant_documents': len(prepared) - unique_count,
            'redundant_document_fraction': (len(prepared) - unique_count) / len(prepared),
            'documents_in_duplicate_groups': sum(len(group['document_ids']) for group in duplicate_groups),
            'duplicate_groups': duplicate_groups,
            'scope': 'exact effective token sequences after truncation, excluding padding and document IDs; not semantic or partial-prefix deduplication',
            'sampling_changed': False,
        }
        result[role] = prepared
    return result, identities


def batch_row(row, device):
    return {name: row[name].unsqueeze(0).to(device) for name in ('input_ids', 'attention_mask')}


@torch.no_grad()
def token_ce(logits, input_ids, attention_mask):
    if logits.shape[:2] != input_ids.shape or input_ids.shape != attention_mask.shape:
        raise ValueError('logits/token/mask shape mismatch')
    valid = (attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()).reshape(-1)
    targets = input_ids[:, 1:].reshape(-1).to(logits.device)
    flat = logits[:, :-1].reshape(-1, logits.shape[-1])
    pieces = []
    for start in range(0, len(flat), 256):
        pieces.append(F.cross_entropy(flat[start:start+256].float(), targets[start:start+256], reduction='none').cpu())
    values = torch.cat(pieces)[valid.cpu()]
    if not len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError('no valid finite supervised losses')
    return values


def fresh_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('output already exists; choose a fresh experiment directory')
    path.mkdir(parents=True)
    return path
