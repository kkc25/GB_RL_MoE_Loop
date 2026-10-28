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



@contextmanager
def mocked_local_model(model, tokenizer, adapter, info):
    # Keep cheap tokenizer preflight independent of the full-weight seam.
    local_inputs = {'metadata': model.config, 'tokenizer': tokenizer,
                    'identity': {'model': info['model'], 'tokenizer': info['tokenizer']}}
    with patch.object(runner, 'load_local_inputs', return_value=local_inputs), \
         patch.object(runner, 'load_local_model', return_value=(model, tokenizer, adapter, info)) as loader:
        yield loader


def pinned_local_inputs(tmp_path, c):
    """Real offline config/tokenizer artifacts; no pretrained weights or download."""
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import MixtralConfig, PreTrainedTokenizerFast
    model_path, tokenizer_path = tmp_path/'model', tmp_path/'tokenizer'
    model_path.mkdir(); tokenizer_path.mkdir()
    MixtralConfig(hidden_size=4096, num_hidden_layers=32, num_local_experts=8,
                  num_experts_per_tok=2).save_pretrained(model_path)
    words = ['[UNK]', '[PAD]', 'train', 'selection', 'calibration', 'token', 'word', 'other', 'extra']
    backend = BackendTokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
    tokenizer.save_pretrained(tokenizer_path)
    c.update(model_path=str(model_path), tokenizer_path=str(tokenizer_path), max_memory={'0':'1GiB'})
    for role, text in {'train':'train token', 'selection':'selection token word other',
                       'calibration':'calibration token word other extra'}.items():
        Path(c['data_files'][role]).write_text(json.dumps({'id':role, 'text':text})+'\n')
    return tokenizer


def test_config_limits(tmp_path):
    c=config(tmp_path); validate_config(c)
    for key,value in [('sequence_length',True),('measurements',0),('nll_tolerance',float('nan')),('experts_implementation','batched_mm')]:
        bad=copy.deepcopy(c);bad[key]=value
        with raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(c);bad['data_files']['test']='a'
    with raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(c);bad['max_train_contexts']=bad['training']['max_train_contexts']=8
    for stage, actor in [(None, None), ('train', None), ('adapter', None), ('profile', 'bound.pt')]:
        with raises(ValueError, match='train token upper bound'):
            validate_config(copy.deepcopy(bad), stage=stage, actor_checkpoint=actor)
    for stage in ('validate', 'oracle', 'profile'):
        validate_config(copy.deepcopy(bad), stage=stage)
    with raises(ValueError, match='unknown Mixtral experiment stage'):
        validate_config(c, stage='unknown')


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
    with patch.object(torch.cuda,'is_available',return_value=True), patch.object(torch.cuda,'is_current_stream_capturing',return_value=False), mocked_local_model(model,Tokenizer(),adapter,info):
        path.write_text(json.dumps(c))
        runner.main(['--config',str(path),'--stage','train'])
        report=json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert report['status']=='complete'
        assert 'routing' in report['latest_development']
        assert 0 <= report['latest_development']['routing']['greedy_native_agreement'] <= 1
        assert 'routing' in report['selected_best_development']
        checkpoint=Path(c['output_dir'])/'router'/'checkpoint_best.pt'
        assert checkpoint.exists()
        c['output_dir']=str(tmp_path/'profile'); path.write_text(json.dumps(c))
        runner.main(['--config',str(path),'--stage','profile','--actor-checkpoint',str(checkpoint)])
        profile=json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert profile['status']=='complete'
        assert profile['identity'] == report['identity']
        assert profile['identity']['context_sampling'] == report['identity']['context_sampling']
        assert profile['identity']['context_sampling'] == json.loads((Path(c['output_dir'])/'train_contexts.json').read_text())
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
        # A changed but still sufficient training corpus must not bypass binding.
        Path(c['data_files']['train']).write_text(json.dumps({'id':'train', 'text':'document 9 changed content'})+'\n')
        c['output_dir'] = str(tmp_path/'profile_changed_train'); path.write_text(json.dumps(c))
        with raises(ValueError, match='checkpoint experiment identities mismatch'):
            runner.main(['--config',str(path),'--stage','profile','--actor-checkpoint',str(checkpoint)])
        failed = json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert failed['status'] == 'failed' and failed['failure_phase'] == 'execution'
        assert failed['identity']['context_sampling'] != report['identity']['context_sampling']
        assert failed['identity']['context_sampling']['actual_count'] == 4


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


def test_learnability_config_contract(tmp_path):
    c=config(tmp_path); validate_config(c)
    assert c['context_sampling']['version'] == 1
    assert c['oracle']['role'] == 'calibration'
    assert c['second_round']['enabled'] is False
    for changes in ({'context_sampling': {'version': 0}},
                    {'oracle': {'role': 'train'}},
                    {'oracle': {'max_contexts': 2, 'max_score_calls': 1}},
                    {'second_round': {'enabled': 'yes'}},
                    {'second_round': {'enabled': True, 'max_train_contexts': 6}}):
        bad=copy.deepcopy(c); bad.update(changes)
        with raises(ValueError): validate_config(bad)


def test_oracle_and_adapter_cli_tiny(tmp_path):
    c=config(tmp_path); path=tmp_path/'config.json'
    model=tiny(); adapter=MixtralAdapter(model)
    before={k:v.clone() for k,v in model.state_dict().items()}
    info={'model':{'hash':'same'},'tokenizer':{'hash':'same'},'cuda_devices':[]}
    c['oracle']={'role':'calibration','max_contexts':2,'max_score_calls':116}
    with patch.object(torch.cuda,'is_available',return_value=True), patch.object(torch.cuda,'is_current_stream_capturing',return_value=False), mocked_local_model(model,Tokenizer(),adapter,info):
        path.write_text(json.dumps(c))
        runner.main(['--config',str(path),'--stage','oracle'])
        report=json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert report['status']=='complete'
        assert report['oracle_contexts']['actual_count']==2
        assert len(report['oracle']['contexts'])==2
        assert 'native_selection' not in report
        assert 'controlled_original_selection' not in report
        assert 'context_sampling' not in report['identity']
        assert report['identity']['oracle_context_sampling'] == report['oracle_contexts']
        assert not (Path(c['output_dir'])/'train_contexts.json').exists()
        c['output_dir']=str(tmp_path/'disabled'); path.write_text(json.dumps(c))
        with raises(ValueError,match='default off'):
            runner.main(['--config',str(path),'--stage','adapter'])
        assert not Path(c['output_dir']).exists()
        c['output_dir']=str(tmp_path/'adapter')
        c['second_round']={'enabled':True,'rank':2,'updates':1,'batch_contexts':2,
                           'max_train_contexts':4,'eval_interval':1}
        path.write_text(json.dumps(c))
        runner.main(['--config',str(path),'--stage','adapter'])
        report=json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert report['status']=='complete'
        assert 'fixed_native_r1_stop' in report['adapter_controls']
        assert 'untrained_native_r2_repeat' in report['adapter_controls']
        assert (Path(c['output_dir'])/'second_round'/'checkpoint_best.pt').exists()
        assert report['selected_adapter_development']['terminal_expert_calls']==1
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])


def test_validate_cli_backward_compatible(tmp_path):
    c=config(tmp_path); path=tmp_path/'config.json'; path.write_text(json.dumps(c))
    model=tiny(); adapter=MixtralAdapter(model)
    info={'model':{'hash':'same'},'tokenizer':{'hash':'same'},'cuda_devices':[]}
    with patch.object(torch.cuda,'is_available',return_value=True), mocked_local_model(model,Tokenizer(),adapter,info):
        runner.main(['--config',str(path)])
    report=json.loads((Path(c['output_dir'])/'report.json').read_text())
    assert report['stage']=='validate' and report['status']=='complete'
    assert report['config']['context_sampling']['strategy']=='document_balanced_uniform_positions'


def test_route_diagnostics_preserves_full_padded_execution_shape(tmp_path):
    from mixtral_lab.adapter import pair_to_action
    c=config(tmp_path); data,_=prepare_data(c,Tokenizer())
    adapter=MixtralAdapter(tiny())
    row=data['selection'][0]
    gate_shapes=[]; actor_shapes=[]; actual=[]
    class ShapeSensitiveActor:
        def log_probs(self, features, native_pairs):
            actor_shapes.append(len(features))
            scores=torch.zeros(len(features),28,device=features.device)
            native=pair_to_action(native_pairs)-1
            # Correct full-shaped execution selects native on even positions;
            # an erroneously filtered four-row call selects no native actions.
            chosen=(native+1)%28
            if len(features)==6:
                chosen[::2]=native[::2]
            scores.scatter_(1,chosen[:,None],2.)
            return scores.log_softmax(-1)
        def __call__(self, obs):
            result=self.log_probs(obs.features,obs.native_pairs).argmax(-1)+1
            actual.append((result.clone(),pair_to_action(obs.native_pairs)))
            return result
    actor=ShapeSensitiveActor()
    hook=adapter.moe.gate.register_forward_pre_hook(lambda module,args:gate_shapes.append(len(args[0])))
    try:
        batch=runner.batch_row(row,runner.input_device(adapter.model))
        result=adapter.forward(**batch,policy=actor,max_calls=1)
        diagnostic=runner.route_diagnostics(adapter,[row],actor)
    finally:
        hook.remove()
    assert actor_shapes==[6,6]
    assert gate_shapes and all(shape==6 for shape in gate_shapes)
    assert diagnostic['supervised_tokens']==4
    assert diagnostic['greedy_native_agreement']==.5
    assert torch.equal(result.actions.reshape(-1),actual[0][0])
    assert diagnostic['greedy_native_agreement']==float((actual[0][0][:4]==actual[0][1][:4]).float().mean())


def test_stage_preflight_uses_only_required_quota_real_tokenizer(tmp_path):
    c = config(tmp_path); pinned_local_inputs(tmp_path, c)
    c['max_train_contexts'] = c['training']['max_train_contexts'] = 8
    model = tiny(); adapter = MixtralAdapter(model)
    c['oracle'] = {'role':'calibration', 'max_contexts':2, 'max_score_calls':116}
    path = tmp_path/'config.json'
    # Train has one valid position, and even its theoretical bound is below 8.
    for stage, role in [('validate', None), ('profile', None),
                        ('oracle', 'calibration'), ('oracle', 'selection')]:
        c['output_dir'] = str(tmp_path/f'{stage}_{role}')
        if role:
            c['oracle']['role'] = role
        path.write_text(json.dumps(c))
        def load_weights(config, *, local_inputs):
            report = json.loads((Path(config['output_dir'])/'report.json').read_text())
            assert report['preflight']['status'] == 'complete'
            assert report['preflight']['data']['train']['supervised_tokens'] == 1
            assert 'train' not in report['preflight']['context_requirements']
            if stage == 'oracle':
                assert json.loads((Path(config['output_dir'])/'oracle_contexts.json').read_text())['role'] == role
            tokenizer = local_inputs['tokenizer']
            info = {**local_inputs['identity'], 'cuda_devices':[]}
            return model, tokenizer, adapter, info
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(runner, 'load_local_model', side_effect=load_weights) as loader:
            runner.main(['--config', str(path), '--stage', stage])
        loader.assert_called_once()
        report = json.loads((Path(c['output_dir'])/'report.json').read_text())
        assert report['status'] == 'complete'
        assert 'context_sampling' not in report['identity']
        assert not (Path(c['output_dir'])/'train_contexts.json').exists()
        if stage == 'oracle':
            assert report['oracle_contexts']['actual_count'] == 2
            assert report['oracle_contexts']['role'] == role
            assert report['identity']['oracle_context_sampling'] == report['oracle_contexts']


def test_required_quota_failure_is_reported_before_weights_or_cuda(tmp_path):
    c = config(tmp_path); pinned_local_inputs(tmp_path, c)
    c['second_round'] = {'enabled':True, 'rank':2, 'updates':1,
                         'batch_contexts':2, 'max_train_contexts':4, 'eval_interval':1}
    path = tmp_path/'config.json'
    for stage, role, count in [('train', 'train', 4), ('adapter', 'train', 4),
                                ('profile', 'train', 4), ('oracle', 'calibration', 5),
                                ('oracle', 'selection', 4)]:
        c['output_dir'] = str(tmp_path/f'failed_{stage}_{role}')
        if stage == 'oracle':
            c['oracle'] = {'role':role, 'max_contexts':count, 'max_score_calls':58*count}
        path.write_text(json.dumps(c))
        argv = ['--config', str(path), '--stage', stage]
        if stage == 'profile':
            argv += ['--actor-checkpoint', str(tmp_path/'bound_actor.pt')]
        with patch.object(runner, 'load_local_model') as loader, \
             patch.object(torch.cuda, 'is_available') as cuda_check:
            with raises(ValueError, match=f'{role} context preflight failed'):
                runner.main(argv)
        loader.assert_not_called(); cuda_check.assert_not_called()
        report_path = Path(c['output_dir'])/'report.json'
        before = report_path.read_bytes()
        report = json.loads(before)
        assert report['status'] == 'failed' and report['failure_phase'] == 'preflight'
        assert report['preflight']['status'] == 'failed'
        assert report['model_load_started'] is False
        required = report['preflight']['context_requirements']['oracle' if stage == 'oracle' else 'train']
        assert required['role'] == role and required['requested_count'] == count
        assert required['available_positions'] < count
        with raises(FileExistsError):
            runner.main(argv)
        assert report_path.read_bytes() == before



def test_data_leakage_preflight_failure_preserves_report_before_weights(tmp_path):
    c = config(tmp_path); pinned_local_inputs(tmp_path, c)
    c['oracle'] = {'role':'calibration', 'max_contexts':2, 'max_score_calls':116}
    # Distinct raw documents truncate to the same consumed tokens in two roles.
    Path(c['data_files']['train']).write_text(json.dumps({'id':'train', 'text':'token '*8+'train'}))
    Path(c['data_files']['selection']).write_text(json.dumps({'id':'selection', 'text':'token '*8+'selection'}))
    path = tmp_path/'config.json'; path.write_text(json.dumps(c))
    with patch.object(runner, 'load_local_model') as loader, \
         patch.object(torch.cuda, 'is_available') as cuda_check:
        with raises(ValueError, match='effective token sequence across'):
            runner.main(['--config', str(path), '--stage', 'oracle'])
    loader.assert_not_called(); cuda_check.assert_not_called()
    report = json.loads((Path(c['output_dir'])/'report.json').read_text())
    assert report['status'] == 'failed' and report['failure_phase'] == 'preflight'
    assert report['model_load_started'] is False
    assert 'effective token sequence across' in report['error']


def test_local_inputs_are_offline_cheap_and_change_guarded(tmp_path):
    from transformers import AutoConfig, AutoTokenizer
    c = config(tmp_path); pinned_local_inputs(tmp_path, c)
    weight_path = Path(c['model_path'])/'model.safetensors'
    weight_path.write_bytes(b'not loaded in input preflight')
    original_hash = runner.sha256_file
    def cheap_hash(path):
        assert Path(path).suffix != '.safetensors'
        return original_hash(path)
    with patch.object(runner, 'sha256_file', side_effect=cheap_hash), \
         patch.object(AutoConfig, 'from_pretrained', wraps=AutoConfig.from_pretrained) as metadata_loader, \
         patch.object(AutoTokenizer, 'from_pretrained', wraps=AutoTokenizer.from_pretrained) as tokenizer_loader:
        inputs = runner.load_local_inputs(c)
    for loader in (metadata_loader, tokenizer_loader):
        assert loader.call_args.kwargs['local_files_only'] is True
        assert loader.call_args.kwargs['trust_remote_code'] is False
    assert inputs['tokenizer'].encode('train token') == [2, 5]
    assert 'model.safetensors' not in inputs['identity']['model']['files']
    # Changes between preflight and weights cannot be rebound silently.
    tokenizer_config = Path(c['tokenizer_path'])/'tokenizer_config.json'
    tokenizer_config.write_text(tokenizer_config.read_text()+'\n')
    from transformers import AutoModelForCausalLM
    with patch.object(AutoModelForCausalLM, 'from_pretrained') as loader:
        with raises(ValueError, match='changed after preflight'):
            runner.load_local_model(c, local_inputs=inputs)
    loader.assert_not_called()
    # A mutation while tokenizer loading is in progress is also rejected.
    original_loader = AutoTokenizer.from_pretrained
    def changed_tokenizer(*args, **kwargs):
        tokenizer = original_loader(*args, **kwargs)
        tokenizer_config.write_text(tokenizer_config.read_text()+'\n')
        return tokenizer
    with patch.object(AutoTokenizer, 'from_pretrained', side_effect=changed_tokenizer):
        with raises(ValueError, match='changed during preflight loading'):
            runner.load_local_inputs(c)


def test_weight_identity_change_during_loading_fails_closed(tmp_path):
    from transformers import AutoModelForCausalLM
    c = config(tmp_path); pinned_local_inputs(tmp_path, c)
    weight_path = Path(c['model_path'])/'model.safetensors'
    weight_path.write_bytes(b'original pinned weight bytes')
    inputs = runner.load_local_inputs(c)
    model = tiny()
    def changing_weights(*args, **kwargs):
        assert kwargs['config'] is inputs['metadata']
        assert kwargs['local_files_only'] is True and kwargs['trust_remote_code'] is False
        assert kwargs['use_safetensors'] is True and kwargs['output_loading_info'] is True
        weight_path.write_bytes(b'changed while loading')
        return model, {key:[] for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
    with patch.object(torch.cuda, 'device_count', return_value=1), \
         patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=changing_weights):
        with raises(ValueError, match='local artifacts changed during weight loading'):
            runner.load_local_model(c, local_inputs=inputs)
        # Public one-argument API still performs cheap inputs before weights.
        with patch.object(runner, 'load_local_inputs', return_value=inputs) as preflight:
            with raises(ValueError, match='local artifacts changed during weight loading'):
                runner.load_local_model(c)
        preflight.assert_called_once_with(c)



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
