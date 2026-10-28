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
from mixtral_lab.protocol import validate_config, read_documents, prepare_data, token_ce, fresh_output
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
        return [1,2,3,4,5]


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
