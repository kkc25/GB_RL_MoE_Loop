"""Real tiny HF models: no Hub access or downloaded pretrained weights."""
import unittest
from unittest.mock import patch
from importlib.util import find_spec
import torch
import torch.nn.functional as F

if find_spec('transformers') is not None:
    from transformers import MixtralConfig, MixtralForCausalLM
from mixtral_lab.adapter import MixtralAdapter, PAIR_TABLE, action_to_pair, pair_to_action


def tiny(dtype=torch.float32, backend='eager'):
    torch.manual_seed(19)
    config = MixtralConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=8, num_experts_per_tok=2, max_position_embeddings=64)
    config._attn_implementation = 'eager'
    config._experts_implementation = backend
    return MixtralForCausalLM(config).to(dtype=dtype).eval()


def check_native_r1_exact_parity(dtype, backend):
    model = tiny(dtype, backend)
    adapter = MixtralAdapter(model, head_chunk=3)
    ids = torch.tensor([[0, 0, 5, 6, 7], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    native = model(ids, attention_mask=mask, use_cache=False).logits
    controlled = adapter.forward(ids, mask)
    torch.testing.assert_close(controlled.logits, native, rtol=0, atol=0)
    assert model.config._experts_implementation == backend
    assert torch.all(controlled.rounds == 1)
    assert not any(p.requires_grad for p in model.parameters())
    assert not controlled.logits.requires_grad
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)
    # Readout uses the genuine untied pretrained head, not the embeddings.
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    h0 = adapter.capture(ids, mask).reshape(-1, 16)
    native_pairs = adapter.native_pair(h0)
    torch.testing.assert_close(adapter.step_pairs(h0, native_pairs.flip(-1)),
                               adapter.step_pairs(h0, native_pairs), rtol=0, atol=0)
    labels = ids.flatten()
    losses = adapter.score_pairs(h0, labels, native_pairs)
    expected = F.cross_entropy(native.reshape(-1, 32).float(), labels, reduction='none')
    torch.testing.assert_close(losses, expected, rtol=1e-6, atol=1e-6)


def test_action_bijection_and_validation():
    actions = torch.arange(1, 29)
    assert action_to_pair(actions).tolist() == [list(pair) for pair in PAIR_TABLE]
    assert torch.equal(pair_to_action(action_to_pair(actions)), actions)
    assert torch.equal(pair_to_action(action_to_pair(actions).flip(-1)), actions)
    for invalid in (torch.tensor([0]), torch.tensor([29]), torch.tensor([1.])):
        with unittest.TestCase().assertRaises(ValueError):
            action_to_pair(invalid)
    for invalid in (torch.tensor([[2, 2]]), torch.tensor([[-1, 3]]), torch.tensor([[3, 8]])):
        with unittest.TestCase().assertRaises(ValueError):
            pair_to_action(invalid)


def test_stop_and_second_call_contract():
    adapter = MixtralAdapter(tiny())
    ids = torch.tensor([[2, 4, 6]])
    native = adapter.forward(ids)
    def first_native_then_stop(obs):
        return torch.where(obs.rounds == 0, pair_to_action(obs.native_pairs), 0)
    stopped = adapter.forward(ids, policy=first_native_then_stop, max_calls=2)
    torch.testing.assert_close(stopped.logits, native.logits, rtol=0, atol=0)
    assert torch.all(stopped.rounds == 1)
    assert torch.all(stopped.actions[..., 1] == 0)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'STOP'):
        adapter.forward(ids, policy=lambda obs: torch.zeros_like(obs.rounds), max_calls=2)
    twice = adapter.forward(ids, max_calls=2)
    h0 = adapter.capture(ids)
    h1 = adapter.step_pairs(h0, adapter.native_pair(h0))
    h2 = adapter.step_pairs(h1, adapter.native_pair(h1))
    torch.testing.assert_close(twice.final_hidden, h2, rtol=0, atol=0)
    assert torch.all(twice.rounds == 2)
    assert adapter.features(h0).shape == (3, 52)


def test_capture_hook_cleanup_on_native_error():
    model = tiny()
    adapter = MixtralAdapter(model)
    module = adapter.layer.post_attention_layernorm
    initial = len(module._forward_pre_hooks)
    with unittest.TestCase().assertRaises((IndexError, RuntimeError)):
        adapter.capture(torch.tensor([[10000]]))
    assert len(module._forward_pre_hooks) == initial
    assert not adapter._capturing
    adapter.capture(torch.tensor([[1, 2]]))
    assert len(module._forward_pre_hooks) == initial


def test_reject_offload_quantization_and_wrong_contract():
    model = tiny()
    model.hf_device_map = {'model': 'cpu'}
    with unittest.TestCase().assertRaisesRegex(ValueError, 'offload'):
        MixtralAdapter(model)
    del model.hf_device_map
    model.is_quantized = True
    with unittest.TestCase().assertRaisesRegex(ValueError, 'quantized'):
        MixtralAdapter(model)
    model.is_quantized = False
    model.config.num_experts_per_tok = 1
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Top-2'):
        MixtralAdapter(model)


@unittest.skipUnless(find_spec('transformers') is not None, 'optional Mixtral dependencies absent')
class TestMixtralAdapter(unittest.TestCase):
    def test_native_parity_matrix(self):
        for dtype in (torch.float32, torch.bfloat16):
            for backend in ('eager', 'grouped_mm'):
                with self.subTest(dtype=dtype, backend=backend):
                    check_native_r1_exact_parity(dtype, backend)

    def test_r1_reference_parity_all_actions(self):
        # Cover all 28 unordered pairs, plus native routing, on real HF kernels.
        ids = torch.arange(28).reshape(4, 7) % 31
        for dtype in (torch.float32, torch.bfloat16):
            for backend in ('eager', 'grouped_mm'):
                with self.subTest(dtype=dtype, backend=backend):
                    adapter = MixtralAdapter(tiny(dtype, backend))
                    before = {k: v.clone() for k, v in adapter.model.state_dict().items()}
                    seen = []
                    def all_pairs(obs):
                        seen.append(obs.features.clone())
                        return torch.arange(1, 29, device=obs.h.device)
                    for policy in (None, all_pairs, lambda obs: pair_to_action(obs.native_pairs)):
                        fast = adapter.forward(ids, policy=policy)
                        reference = adapter.forward(ids, policy=policy, reference=True)
                        for field in ('logits', 'final_hidden', 'actions', 'rounds'):
                            torch.testing.assert_close(getattr(fast, field), getattr(reference, field),
                                                       rtol=0, atol=0)
                    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
                    for k, v in adapter.model.state_dict().items():
                        self.assertTrue(torch.equal(before[k], v), k)
                    self.assertFalse(any(p.requires_grad or p.grad is not None
                                         for p in adapter.model.parameters()))

    def test_r1_norm_gate_counts_and_no_unused_observation(self):
        adapter = MixtralAdapter(tiny())
        ids = torch.tensor([[2, 3, 4]])
        counts = dict(norm=0, gate=0)
        def increment(key):
            def hook(*args):
                counts[key] += 1
            return hook
        handles = [adapter.layer.post_attention_layernorm.register_forward_hook(increment('norm')),
                   adapter.moe.gate.register_forward_hook(increment('gate'))]
        try:
            for policy in (None, lambda obs: pair_to_action(obs.native_pairs)):
                counts.update(norm=0, gate=0)
                adapter.forward(ids, policy=policy)
                self.assertEqual(counts, dict(norm=1, gate=1))
            counts.update(norm=0, gate=0)
            adapter.forward(ids, reference=True)
            self.assertEqual(counts, dict(norm=3, gate=2))
            with patch.object(adapter, '_observation', side_effect=AssertionError('unused features')):
                adapter.forward(ids)
        finally:
            for handle in handles:
                handle.remove()

    def test_r1_policy_validation(self):
        adapter = MixtralAdapter(tiny())
        ids = torch.tensor([[2, 3, 4]])
        for invalid in (torch.zeros(3, dtype=torch.long), torch.full((3,), 29),
                        torch.full((3,), -1), torch.ones(3), torch.ones(3, 1, dtype=torch.long), None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    adapter.forward(ids, policy=lambda obs: invalid)
        for calls in (True, 0, 3):
            with self.assertRaises(ValueError):
                adapter.forward(ids, max_calls=calls)

    def test_mixed_stop_compaction(self):
        adapter = MixtralAdapter(tiny())
        ids = torch.tensor([[2, 3, 4, 5]])
        def policy(obs):
            chosen = pair_to_action(obs.native_pairs)
            if bool((obs.rounds == 1).all()):
                chosen[::2] = 0
            return chosen
        result = adapter.forward(ids, policy=policy, max_calls=2)
        self.assertEqual(result.rounds.tolist(), [[1, 2, 1, 2]])
        native = adapter.forward(ids)
        torch.testing.assert_close(result.final_hidden[:, ::2], native.final_hidden[:, ::2], rtol=0, atol=0)

    def test_nonnative_extreme_logits_still_finite(self):
        adapter = MixtralAdapter(tiny())
        h = torch.ones(1, 16)
        with torch.no_grad():
            adapter.moe.gate.weight[0].fill_(1000.)
            adapter.moe.gate.weight[1:].fill_(-1000.)
        result = adapter.step_pairs(h, torch.tensor([[2, 3]]))
        self.assertTrue(torch.isfinite(result).all())

    def test_nested_offload_rejected(self):
        from types import SimpleNamespace
        model = tiny()
        model._hf_hook = SimpleNamespace(hooks=[SimpleNamespace(offload=True)])
        with self.assertRaisesRegex(ValueError, 'offload'):
            MixtralAdapter(model)

    def _gpu_tiny(self, backend):
        # Aligned dimensions exercise CUDA grouped-MM without pretrained weights.
        config = tiny().config
        config.hidden_size = 64
        config.intermediate_size = 64
        config._experts_implementation = backend
        return MixtralForCausalLM(config).to(dtype=torch.bfloat16).eval()

    def _assert_cuda_parity(self, model):
        adapter = MixtralAdapter(model)
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        terminal_norm = adapter.layer.post_attention_layernorm
        hooks_before = len(terminal_norm._forward_pre_hooks)
        device = model.get_input_embeddings().weight.device
        ids = torch.tensor([[0, 0, 5, 6, 7], [1, 2, 3, 4, 5]], device=device)
        mask = ids.ne(0).long()
        with torch.no_grad():
            native = model(ids, attention_mask=mask, use_cache=False).logits.float().cpu()
            fast = adapter.forward(ids, mask)
            controlled = fast.logits.float().cpu()
            for policy in (None, lambda obs: pair_to_action(obs.native_pairs),
                           lambda obs: torch.arange(len(obs.h), device=obs.h.device) % 28 + 1):
                candidate = adapter.forward(ids, mask, policy=policy)
                reference = adapter.forward(ids, mask, policy=policy, reference=True)
                torch.testing.assert_close(candidate.actions, reference.actions, rtol=0, atol=0)
                torch.testing.assert_close(candidate.rounds, reference.rounds, rtol=0, atol=0)
                torch.testing.assert_close(candidate.final_hidden, reference.final_hidden, rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(candidate.logits, reference.logits, rtol=2e-2, atol=2e-3)
        maximum_error = (native - controlled).abs().max().item()
        # Same backend and precision, not a cross-kernel equivalence assertion.
        # BF16 GPU reductions may differ slightly with changed tensor layout.
        torch.testing.assert_close(controlled, native, rtol=2e-2, atol=2e-3,
            msg=f'CUDA native/controlled max_abs={maximum_error:.9g}')
        self.assertEqual(len(terminal_norm._forward_pre_hooks), hooks_before)
        self.assertFalse(adapter._capturing)
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in model.parameters()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value.detach().cpu()), key)
        captured = adapter.capture(ids, mask)
        self.assertEqual(captured.device, adapter.device)
        self.assertEqual(len(terminal_norm._forward_pre_hooks), hooks_before)
        return adapter

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_bf16_native_controlled_backends(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest('CUDA device lacks BF16 support')
        for backend in ('eager', 'grouped_mm'):
            with self.subTest(backend=backend):
                model = self._gpu_tiny(backend).to('cuda:0')
                self._assert_cuda_parity(model)
                del model

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and find_spec('accelerate') is not None,
                         'requires two CUDA devices and accelerate')
    def test_two_gpu_dispatched_prefix_and_readout(self):
        from accelerate import dispatch_model
        for device in (0, 1):
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    self.skipTest(f'CUDA device {device} lacks BF16 support')
        placement = {'model.embed_tokens': 'cuda:0', 'model.layers.0': 'cuda:0',
                     'model.layers.1': 'cuda:1', 'model.norm': 'cuda:1',
                     'model.rotary_emb': 'cuda:0', 'lm_head': 'cuda:1'}
        for backend in ('eager', 'grouped_mm'):
            with self.subTest(backend=backend):
                model = dispatch_model(self._gpu_tiny(backend), device_map=placement)
                adapter = self._assert_cuda_parity(model)
                self.assertEqual(adapter.device, torch.device('cuda:1'))
                self.assertEqual(model.get_input_embeddings().weight.device, torch.device('cuda:0'))
                self.assertEqual(model.lm_head.weight.device, torch.device('cuda:1'))
                del adapter, model

    test_actions = staticmethod(test_action_bijection_and_validation)
    test_stop = staticmethod(test_stop_and_second_call_contract)
    test_capture_cleanup = staticmethod(test_capture_hook_cleanup_on_native_error)
    test_rejections = staticmethod(test_reject_offload_quantization_and_wrong_contract)
