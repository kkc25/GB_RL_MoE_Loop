"""Offline bounded protocol and end-to-end CLI contracts; never load real weights."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
from contextlib import contextmanager

@contextmanager
def raises(exc, match=None):
    with unittest.TestCase().assertRaises(exc) as caught:
        yield
    if match is not None:
        import re
        assert re.search(match, str(caught.exception))

import torch
from mixtral_lab.protocol import validate_config, read_documents, prepare_data, token_ce, fresh_output, directory_identity, sha256_file
from experiments import mixtral_experiment as runner
from tests.test_mixtral_adapter import tiny
from mixtral_lab.adapter import MixtralAdapter


def config(tmp):
    c = json.loads(Path('configs/mixtral/frozen_router_smoke.json').read_text())
    c.update(output_dir=str(tmp/'output'), max_train_contexts=4, sequence_length=6)
    c['training'].update(updates=1,batch_contexts=2,group_size=2,eval_interval=1,max_train_contexts=4)
    c['sequences']={k:1 for k in c['sequences']}
    for i,k in enumerate(c['data_files']):
        p=tmp/(k+'.jsonl');p.write_text(json.dumps({'id':k,'text':f'document {i} unique content'})+'\n')
        c['data_files'][k]=str(p)
    return c


class Tokenizer:
    pad_token_id=0
    def encode(self,text,**kwargs):
        # Keep development roles distinct at the effective-token level.
        return [1,2,3,4,5 + int(text.split()[1])]


def test_config_limits(tmp_path):
    c=config(tmp_path); validate_config(c)
    for key,value in [('sequence_length',True),('measurements',0),('nll_tolerance',float('nan')),('experts_implementation','batched_mm')]:
        bad=copy.deepcopy(c);bad[key]=value
        with raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(c);bad['data_files']['test']='a'
    with raises(ValueError):validate_config(bad)


def test_duplicates_and_tokenization(tmp_path):
    c=config(tmp_path)
    data,identity=prepare_data(c,Tokenizer())
    assert data['train'][0]['input_ids'].tolist()==[1,2,3,4,5,0]
    assert identity['train']['supervised_tokens']==4
    Path(c['data_files']['selection']).write_text(json.dumps({'id':'different','text':'document   0 unique content'}))
    with raises(ValueError,match='duplicate|overlap'):read_documents(c)


def test_cached_hf_snapshot_symlinks(tmp_path):
    hub = tmp_path/'.cache'/'huggingface'/'hub'/'models--local--tiny'
    blobs = hub/'blobs'; blobs.mkdir(parents=True)
    snapshot = hub/'snapshots'/('a'*40); snapshot.mkdir(parents=True)
    blob = blobs/'weight-blob'; blob.write_bytes(b'actual weight bytes')
    (snapshot/'model.safetensors').symlink_to(blob)
    (snapshot/'config.json').write_text('{"model_type":"mixtral"}')
    internal = snapshot/'.cache'; internal.mkdir()
    (internal/'temporary.json').write_text('{}')
    identity = directory_identity(snapshot)
    assert set(identity['files']) == {'model.safetensors', 'config.json'}
    assert identity['files']['model.safetensors'] == sha256_file(blob)
    before = identity['content_sha256']
    blob.write_bytes(b'changed actual weight bytes')
    assert directory_identity(snapshot)['content_sha256'] != before


def test_effective_token_overlap_real_fast_tokenizer(tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    backend = BackendTokenizer(WordLevel({'[UNK]':0, '[PAD]':1, 'shared':2,
        'token':3, 'train_tail':4, 'selection_tail':5, 'calibration':6}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
    c = config(tmp_path)
    for role, text in {'train':'shared ' + 'token '*12 + 'train_tail',
                       'selection':'shared ' + 'token '*12 + 'selection_tail',
                       'calibration':'calibration token'}.items():
        Path(c['data_files'][role]).write_text(json.dumps({'id':role, 'text':text}))
    # The raw documents genuinely differ; only their consumed sequences collide.
    read_documents(c)
    with raises(ValueError, match='effective token sequence across'):
        prepare_data(c, tokenizer)
    # Also reject tokenization-normalized duplicates even when padding is needed.
    # Unknown spellings normalize to the same UNK token, independent of IDs.
    Path(c['data_files']['train']).write_text(json.dumps({'id':'train', 'text':'shared unknown_a'}))
    Path(c['data_files']['selection']).write_text(json.dumps({'id':'selection', 'text':'shared unknown_b'}))
    with raises(ValueError, match='effective token sequence across'):
        prepare_data(c, tokenizer)


def test_backend_configuration_is_not_dispatch_proof():
    evidence = runner.expert_backend_evidence('grouped_mm', ['grouped_mm']*2)
    assert evidence['configuration_matches_request']
    assert evidence['actual_dispatch'] is None
    assert evidence['dispatch_verification_status'] == 'not_instrumented'



def test_dispatch_observer_restores_and_fails_closed():
    from transformers.integrations import moe
    original = moe._can_use_grouped_mm
    tensor = torch.zeros(2, 2); weights = torch.zeros(1, 2, 2); offsets = torch.tensor([2], dtype=torch.int32)
    with patch.object(runner, 'version', return_value='5.15.1'):
        with runner.observe_native_expert_dispatch() as evidence:
            moe._can_use_grouped_mm(tensor, weights, offsets)
        assert evidence['status'] == 'dispatch_branch_observed'
        assert evidence['observations'][0]['calls'] == 1
        assert evidence['observations'][0]['device'] == 'cpu'
        assert moe._can_use_grouped_mm is original
        with raises(RuntimeError):
            with runner.observe_native_expert_dispatch():
                raise RuntimeError('failed calibration')
        assert moe._can_use_grouped_mm is original
    with patch.object(runner, 'version', return_value='unknown'):
        with runner.observe_native_expert_dispatch() as evidence:
            assert moe._can_use_grouped_mm is original
        assert evidence['status'] == 'unverified'
        assert not evidence['observations']



def test_real_grouped_calibration_dispatch_observed(tmp_path):
    from transformers.integrations import moe
    original = moe._can_use_grouped_mm
    c = config(tmp_path); c['experts_implementation'] = 'grouped_mm'
    data, _ = prepare_data(c, Tokenizer())
    result = runner.validate_native(MixtralAdapter(tiny(backend='grouped_mm')), data['calibration'], c)
    evidence = result['groups'][0]['native_expert_dispatch']
    assert evidence['status'] == 'dispatch_branch_observed'
    assert sum(row['calls'] for row in evidence['observations']) > 0
    assert all(row['device'] == 'cpu' for row in evidence['observations'])
    assert moe._can_use_grouped_mm is original


def test_causal_ce_and_padding():
    logits=torch.zeros(1,4,8);ids=torch.tensor([[1,2,3,0]]);mask=torch.tensor([[1,1,1,0]])
    logits[0,0,2]=4;logits[0,1,3]=4
    losses=token_ce(logits,ids,mask)
    assert losses.shape==(2,)
    expected=torch.nn.functional.cross_entropy(logits[0,:2],ids[0,1:3],reduction='none')
    torch.testing.assert_close(losses,expected)


def test_output_and_precision_guards(tmp_path):
    fresh_output(tmp_path/'out')
    with raises(FileExistsError):fresh_output(tmp_path/'out')
    before=torch.backends.cuda.matmul.allow_tf32
    with raises(RuntimeError):
        with runner.precision_scope(not before):raise RuntimeError('test')
    assert torch.backends.cuda.matmul.allow_tf32==before


def test_tiny_native_screen(tmp_path):
    c=config(tmp_path);data,_=prepare_data(c,Tokenizer())
    result=runner.validate_native(MixtralAdapter(tiny()),data['calibration'],c)
    assert result['eligible']
    assert result['groups'][0]['exact_logits']


def test_train_then_profile_new_output(tmp_path):
    c=config(tmp_path);path=tmp_path/'config.json'
    model=tiny(); adapter=MixtralAdapter(model)
    info={'model':{'hash':'same'},'tokenizer':{'hash':'same'},'cuda_devices':[]}
    with patch.object(torch.cuda,'is_available',return_value=True), patch.object(torch.cuda,'is_current_stream_capturing',return_value=False), patch.object(runner,'load_local_model',return_value=(model,Tokenizer(),adapter,info)):
        path.write_text(json.dumps(c))
        runner.main(['--config',str(path),'--stage','train'])
        report=json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert report['status']=='complete'
        checkpoint=Path(c['output_dir'])/'router'/'checkpoint_best.pt'
        assert checkpoint.exists()
        c['output_dir']=str(tmp_path/'profile'); path.write_text(json.dumps(c))
        runner.main(['--config',str(path),'--stage','profile','--actor-checkpoint',str(checkpoint)])
        profile=json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert profile['status']=='complete'
        assert 'learned_route_r1' in profile['timing']['variants']
        provenance = profile['actor_checkpoint']
        assert provenance['path'] == str(checkpoint.resolve())
        assert provenance['checkpoint_sha256'] == sha256_file(checkpoint)
        payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert provenance['binding_sha256'] == payload['binding']['sha256']
        assert provenance['update'] == payload['update']
        assert provenance['epsilon'] == payload['epsilon']
        assert provenance['training_config'] == payload['binding']['config']
        assert len(provenance['actor_state_sha256']) == 64
        # A second valid checkpoint under the same experiment binding is traceable.
        payload['update'] = 1 - provenance['update']
        first = next(iter(payload['actor_state_dict']))
        payload['actor_state_dict'][first] = payload['actor_state_dict'][first] + .001
        second = tmp_path/'second_actor.pt'; torch.save(payload, second)
        c['output_dir'] = str(tmp_path/'profile_second'); path.write_text(json.dumps(c))
        runner.main(['--config',str(path),'--stage','profile','--actor-checkpoint',str(second)])
        other = json.loads((Path(c['output_dir'])/'report.json').read_text())['actor_checkpoint']
        assert other['checkpoint_sha256'] == sha256_file(second)
        assert other['checkpoint_sha256'] != provenance['checkpoint_sha256']
        assert other['actor_state_sha256'] != provenance['actor_state_sha256']
        assert other['binding_sha256'] == provenance['binding_sha256']
        assert other['update'] != provenance['update']


def test_complete_checkpoint_guard(tmp_path):
    # Real installed HF serialization/loading API, tiny random model only, no network.
    from transformers import AutoModelForCausalLM
    model = tiny()
    model.save_pretrained(tmp_path/'tiny')
    loaded, info = AutoModelForCausalLM.from_pretrained(str(tmp_path/'tiny'), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    assert isinstance(info, dict)
    assert runner.require_complete_checkpoint(info)['missing_keys'] == []
    for field in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        bad = dict(info); bad[field] = {'bad'}
        with raises(ValueError, match='incomplete'):runner.require_complete_checkpoint(bad)
    with raises(ValueError):runner.require_complete_checkpoint({})
    # A genuinely missing tensor is tolerated by HF but refused by our guard.
    from safetensors.torch import load_file, save_file
    path = tmp_path/'tiny'/'model.safetensors'
    tensors = load_file(str(path)); tensors.pop('lm_head.weight')
    save_file(tensors, str(path), metadata={'format':'pt'})
    _, incomplete = AutoModelForCausalLM.from_pretrained(str(tmp_path/'tiny'), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    with raises(ValueError, match='incomplete'):runner.require_complete_checkpoint(incomplete)


class ProtocolTests(unittest.TestCase):
    pass


def _wrap(function):
    def wrapped(self):
        with tempfile.TemporaryDirectory() as directory:
            if function.__code__.co_argcount:
                function(Path(directory))
            else:
                function()
    return wrapped


for _name, _function in list(globals().items()):
    if _name.startswith('test_'):
        setattr(ProtocolTests, _name, _wrap(_function))
        del globals()[_name]
