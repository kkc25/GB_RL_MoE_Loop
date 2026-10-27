"""CPU/mock protocol regressions, not GPU performance or numerical evidence."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

import torch

from experiments import rl_lm_profile as profile
from model.moe import MoE
from model.top2 import Top2MoE, set_top2_dispatch_backend, top2_dispatch_identity


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        for block in self.blocks:
            block.executor = Top2MoE(MoE(4, 6, 8))
        self.config = SimpleNamespace(adaptive_layers=(1,), hidden_size=4)
        self.calls = []
        self.requires_grad_(False)

    def forward(self, ids, labels, **kwargs):
        self.calls.append((ids, kwargs, top2_dispatch_identity(self)))


def fake_measure(fn, *, measurements, before_call=None, **kwargs):
    for _ in range(measurements):
        if before_call:
            before_call()
        fn()
    return dict(raw=[dict(event_ms=[1.] * measurements, wall_ms=[2.] * measurements)],
                allocated_bytes=0, reserved_bytes=0, peak_allocated_bytes=0,
                peak_reserved_bytes=0)


class FactorialProfileTests(unittest.TestCase):
    def builders(self, model, factorial=True):
        ids = [torch.tensor([[i]]) for i in range(2)]
        segments = [torch.zeros_like(item) for item in ids]
        actor = object()
        args = SimpleNamespace(sequences=2, factorial_end_to_end=factorial,
                               prefix_backend='merged', deployment_backend='sparse_active_merged_v3')
        variants = profile.build_end_to_end_variants(
            1, model=model, dense_model=FakeModel(), all_input_ids=ids,
            all_segment_ids=segments, args=args, actors={'learned_r2': (actor, 2)},
            device=torch.device('cpu'), compute_dtype='float32', chunk=64, head_chunk=32)
        return variants, ids, segments, actor

    def test_factorial_cells_and_selected_original_control(self):
        for factorial, expected_online in [(True, 12), (False, 3)]:
            model = FakeModel()
            variants, ids, segments, actor = self.builders(model, factorial)
            self.assertEqual(len(variants), expected_online + 3)
            self.assertEqual(len({name for name, _ in variants}), len(variants))
            with patch.object(profile, 'forward_with_last_ffn_policy') as online:
                for name, variant in reversed(variants):
                    if isinstance(variant, profile.RuntimeVariant):
                        before = {k: v.clone() for k, v in model.state_dict().items()}
                        identity = variant.prepare()
                        variant(1)
                        self.assertIn('__prefix_' + identity['prefix_backend'], name)
                        self.assertIn('__terminal_' + identity['terminal_backend'], name)
                        if identity['policy'] == 'original_moe':
                            self.assertTrue(all(e['backend'] == 'legacy' for e in model.calls[-1][2]['executors']))
                        else:
                            call = online.call_args
                            self.assertIs(call.args[0], model)
                            self.assertIs(call.args[2], ids[1])
                            self.assertIs(call.args[3], segments[1])
                            self.assertEqual(call.kwargs['backend'], identity['terminal_backend'])
                            self.assertEqual(call.kwargs['max_calls'], identity['max_calls'])
                            if identity['policy'] == 'learned_r2':
                                self.assertIs(call.args[1], actor)
                        for key, value in model.state_dict().items():
                            self.assertTrue(torch.equal(before[key], value))

    def test_all_cells_interleaved_same_input_indices_and_verified_each_call(self):
        model = FakeModel()
        calls, windows = [], []
        variants = []
        for prefix, terminal in profile.end_to_end_cells(factorial=True, prefix_backend='legacy', deployment_backend='unused'):
            name = prefix + '/' + terminal
            def run(index, name=name, prefix=prefix):
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(model.blocks[0].executor.dispatch_backend, prefix)
                self.assertEqual(model.blocks[1].executor.dispatch_backend, 'legacy')
                calls.append((name, index))
            variants.append((name, profile.RuntimeVariant(model, run, prefix, terminal, 'fixed', 2)))
        def measure(fn, **kwargs):
            start = len(calls)
            result = fake_measure(fn, **kwargs)
            windows.append(calls[start:])
            return result
        with patch.object(profile, 'configure_prefix', wraps=profile.configure_prefix) as configure:
            result = profile.measure_variants(variants, groups=2, warmup=2, repeats=4,
                                              measurements=3, tokens=2, measure_fn=measure)
        self.assertEqual(configure.call_count, 4 * (2 + 4 * 3))
        expected = [name for order in profile.variant_schedule([name for name, _ in variants], 4) for name in order]
        self.assertEqual([window[0][0] for window in windows], expected)
        self.assertTrue(all([i for _, i in window] == [0, 1, 0] for window in windows))
        for name, variant in variants:
            self.assertEqual(result[name]['runtime_identity']['prefix_backend'], variant.prefix_backend)
            self.assertEqual(result[name]['runtime_identity']['terminal_backend'], variant.terminal_backend)

    def test_setter_before_event_and_wall_timer(self):
        events = []
        class Event:
            def __init__(self, **kwargs):
                pass
            def record(self):
                events.append('event')
            def elapsed_time(self, other):
                return 1.
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda, 'Event', Event))
            for method in ('synchronize', 'reset_peak_memory_stats', 'memory_allocated',
                           'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda, method, return_value=0))
            stack.enter_context(patch.object(profile.time, 'perf_counter', side_effect=lambda: events.append('wall') or len(events)))
            profile.measure(lambda: events.append('call'), warmup=1, repeats=1,
                            measurements=2, tokens=1, before_call=lambda: events.append('prepare'))
        self.assertEqual(events, ['prepare', 'call'] + ['prepare', 'event', 'wall', 'call', 'event', 'wall'] * 2)

    def test_component_fixture_restores_legacy_without_weight_changes(self):
        model = FakeModel()
        ids = [torch.ones(1, 2, dtype=torch.long)]
        def cache(actual, *args):
            self.assertIs(actual, model)
            self.assertTrue(all(e['backend'] == 'legacy' for e in top2_dispatch_identity(model)['executors']))
            return torch.ones(1, 2, 4)
        with patch.object(profile, 'cache_last_ffn_input', side_effect=cache):
            set_top2_dispatch_backend(model, 'merged')
            first = profile.build_component_hidden_batches(model, ids, ids, 1, torch.device('cpu'), 'float32')
            set_top2_dispatch_backend(model, 'merged')
            second = profile.build_component_hidden_batches(model, ids, ids, 1, torch.device('cpu'), 'float32')
        self.assertTrue(torch.equal(first[0], second[0]))

    def test_measurement_hook_cannot_be_skipped_after_warmup(self):
        variant = profile.RuntimeVariant(FakeModel(), lambda i: None, 'legacy',
                                         'sparse_merged_v1', 'fixed', 1)
        def broken_measure(fn, **kwargs):
            kwargs.pop('before_call')
            return fake_measure(fn, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'before every call'):
            profile.measure_variants([('control', variant)], groups=1, warmup=1,
                                     repeats=1, measurements=1, tokens=1,
                                     measure_fn=broken_measure)

    def test_incorrect_effective_configuration_fails_closed(self):
        model = FakeModel()
        with patch.object(profile, 'set_top2_dispatch_backend'):
            with self.assertRaisesRegex(RuntimeError, 'effective Top2'):
                profile.configure_prefix(model, 'merged')


if __name__ == '__main__':
    unittest.main()
