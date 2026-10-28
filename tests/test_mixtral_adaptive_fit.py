"""Pure CPU scalar policy tests: no model download, torch, or GPU required."""
import copy
import inspect
import json
import random
import subprocess
import sys
import unittest

from mixtral_lab.adaptive_fit import AdaptiveFitConfig, AdaptiveFitController


def controller(**overrides):
    initial_lr = overrides.pop('initial_lr', .01)
    settings = {'enabled': True, **overrides}
    return AdaptiveFitController(settings, initial_lr=initial_lr)


class AdaptiveFitTests(unittest.TestCase):
    def test_defaults_and_none_preserve_fixed_learning_rate(self):
        cfg = AdaptiveFitConfig.from_dict(None)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.eval_interval, 16)
        policy = AdaptiveFitController(cfg, initial_lr=.2)
        for update in range(10):
            result = policy.observe(update, 2., True, ineffective_update=True)
            self.assertEqual(result['action'], 'hold')
            self.assertEqual(result['lr'], .2)
            self.assertFalse(result['stop'])
        self.assertEqual(policy.snapshot()['growths'], 0)
        self.assertEqual(policy.snapshot()['reductions'], 0)
        self.assertEqual(policy.snapshot()['best_train_loss'], 2.)

    def test_baseline_cannot_change_lr_or_stop(self):
        policy = controller(initial_lr=.01, min_lr=.01, min_updates=0,
                            patience=1, stop_patience=1)
        result = policy.observe(64, 2., True, ineffective_update=True)
        self.assertEqual(result['action'], 'hold')
        self.assertEqual(result['reason'], 'baseline')
        self.assertFalse(result['stop'])

    def test_plateau_patience_counts_observations_not_updates(self):
        policy = controller()
        policy.observe(0, 2., False)
        for update in (16, 32):
            self.assertEqual(policy.observe(update, 2., True)['action'], 'hold')
        decision = policy.observe(48, 2., True)
        self.assertEqual(decision['action'], 'reduce_lr')
        self.assertEqual(decision['lr'], .005)
        self.assertFalse(decision['stop'])
        self.assertEqual(decision['state']['plateau_observations'], 0)

    def test_meaningful_progress_resets_plateau_and_updates_best(self):
        policy = controller(patience=3, min_delta=.25)
        policy.observe(0, 2., False)
        policy.observe(1, 1.875, False)
        self.assertEqual(policy.snapshot()['best_train_loss'], 1.875)
        self.assertEqual(policy.snapshot()['progress_anchor_loss'], 2.)
        # Progress accumulates against the last meaningful-progress anchor,
        # rather than being lost when each small improvement updates raw best.
        result = policy.observe(2, 1.5, True)
        self.assertEqual(result['reason'], 'train_loss_improved')
        self.assertEqual(result['state']['plateau_observations'], 0)
        self.assertEqual(result['state']['progress_anchor_loss'], 1.5)
        self.assertEqual(policy.observe(3, 1.5, True)['action'], 'hold')

    def test_zero_min_delta_still_requires_strict_progress(self):
        policy = controller(patience=1, min_delta=0)
        policy.observe(0, 1., True)
        self.assertEqual(policy.observe(1, 1., True)['action'], 'reduce_lr')

    def test_reductions_clamp_to_floor_and_stop_needs_new_floor_observations(self):
        policy = controller(patience=1, min_lr=.006, stop_patience=2, min_updates=0)
        policy.observe(0, 2., True)
        first = policy.observe(1, 2., True)
        self.assertEqual(first['action'], 'reduce_lr')
        self.assertEqual(first['lr'], .006)
        self.assertEqual(first['state']['floor_plateau_observations'], 0)
        self.assertFalse(policy.observe(2, 2., True)['stop'])
        last = policy.observe(3, 2., True)
        self.assertTrue(last['stop'])
        self.assertEqual(last['reason'], 'train_plateau_at_min_lr')

    def test_stop_requires_coverage_and_min_updates_and_consecutive_eligible_checks(self):
        policy = controller(initial_lr=.01, min_lr=.01, max_growths=0,
                            min_updates=4, stop_patience=2)
        policy.observe(0, 2., False)
        for update in (1, 2, 3):
            result = policy.observe(update, 2., True)
            self.assertFalse(result['stop'])
            self.assertEqual(result['state']['floor_plateau_observations'], 0)
        self.assertFalse(policy.observe(4, 2., True)['stop'])
        missing = policy.observe(5, 2., False)
        self.assertFalse(missing['stop'])
        self.assertEqual(missing['reason'], 'training_coverage_incomplete')
        self.assertEqual(missing['state']['floor_plateau_observations'], 0)
        self.assertFalse(policy.observe(6, 2., True)['stop'])
        self.assertTrue(policy.observe(7, 2., True)['stop'])

    def test_improvement_at_floor_restarts_stop_patience(self):
        policy = controller(initial_lr=.01, min_lr=.01, min_updates=0,
                            stop_patience=2, max_growths=0)
        policy.observe(0, 2., True)
        policy.observe(1, 2., True)
        self.assertFalse(policy.observe(2, 1., True)['stop'])
        self.assertFalse(policy.observe(3, 1., True)['stop'])
        self.assertTrue(policy.observe(4, 1., True)['stop'])

    def test_short_smoke_budget_is_not_stopped_for_flat_loss(self):
        policy = controller(initial_lr=.01, min_lr=.01, patience=1, stop_patience=1)
        for update in (0, 16, 32, 48):
            self.assertFalse(policy.observe(update, 1., True)['stop'])

    def test_ineffective_growth_needs_patience_is_capped_and_finite(self):
        policy = controller(initial_lr=.001, max_lr=.003, patience=2,
                            min_updates=100, max_growths=2)
        policy.observe(0, 2., False, True)
        self.assertEqual(policy.observe(1, 2., False, True)['action'], 'hold')
        first = policy.observe(2, 2., False, True)
        self.assertEqual(first['action'], 'grow_lr')
        self.assertEqual(first['lr'], .002)
        policy.observe(3, 2., False, True)
        second = policy.observe(4, 2., False, True)
        self.assertEqual(second['action'], 'grow_lr')
        self.assertEqual(second['lr'], .003)
        policy.observe(5, 2., False, True)
        after_limit = policy.observe(6, 2., False, True)
        self.assertEqual(after_limit['action'], 'reduce_lr')
        self.assertEqual(after_limit['state']['growths'], 2)

    def test_growth_never_occurs_on_worsening_or_stably_degraded_loss(self):
        policy = controller(patience=2)
        policy.observe(0, 1., True)
        for update, loss in enumerate((1.1, 1.1, 1.2, 1.2, 1.3), start=1):
            result = policy.observe(update, loss, True, True)
            self.assertNotEqual(result['action'], 'grow_lr')
        self.assertEqual(policy.snapshot()['growths'], 0)

    def test_missing_ineffectiveness_witness_resets_growth_patience(self):
        policy = controller(patience=3)
        policy.observe(0, 1., True)
        policy.observe(1, 1., True, True)
        policy.observe(2, 1., True, False)
        result = policy.observe(3, 1., True, True)
        self.assertEqual(result['action'], 'reduce_lr')
        self.assertEqual(result['state']['growths'], 0)

    def test_progress_does_not_trigger_ineffectiveness_growth(self):
        policy = controller(patience=1)
        policy.observe(0, 2., True)
        result = policy.observe(1, 1., True, True)
        self.assertEqual(result['reason'], 'train_loss_improved')
        self.assertEqual(result['state']['growths'], 0)

    def test_growth_zero_cap_and_equal_lr_bounds(self):
        for overrides in ({'max_growths': 0}, {'max_lr': .01}):
            with self.subTest(overrides=overrides):
                policy = controller(patience=1, **overrides)
                policy.observe(0, 1., True)
                self.assertNotEqual(policy.observe(1, 1., True, True)['action'], 'grow_lr')
        equal = controller(min_lr=.01, max_lr=.01, min_updates=0, stop_patience=1)
        equal.observe(0, 1., True)
        self.assertTrue(equal.observe(1, 1., True, True)['stop'])

    def test_short_stop_patience_does_not_preempt_available_bf16_recovery(self):
        policy = controller(min_lr=.01, min_updates=0, patience=3, stop_patience=1)
        policy.observe(0, 1., True)
        for update in (1, 2):
            result = policy.observe(update, 1., True, True)
            self.assertFalse(result['stop'])
            self.assertEqual(result['reason'], 'awaiting_native_bf16_recovery_patience')
        self.assertEqual(policy.observe(3, 1., True, True)['action'], 'grow_lr')

    def test_lost_recovery_witness_does_not_indefinitely_defer_floor_stop(self):
        policy = controller(min_lr=.01, min_updates=0, patience=3, stop_patience=1)
        policy.observe(0, 1., True)
        self.assertFalse(policy.observe(1, 1., True, True)['stop'])
        self.assertTrue(policy.observe(2, 1., True, False)['stop'])

    def test_nonfinite_feedback_is_rejected_without_any_mutation(self):
        for loss in (float('nan'), float('inf'), -float('inf'), 10**1000,
                     True, None, '1.0'):
            with self.subTest(loss=str(loss)[:20]):
                policy = controller(patience=1)
                policy.observe(0, 1., True)
                before = policy.snapshot()
                with self.assertRaisesRegex(ValueError, 'finite'):
                    policy.observe(1, loss, True, True)
                self.assertEqual(before, policy.snapshot())
                json.dumps(policy.snapshot(), allow_nan=False)

    def test_inputs_validate_before_state_mutation(self):
        policy = controller()
        policy.observe(0, 1., True)
        for args in [(0, 1., True), (-1, 1., True), (10001, 1., True),
                     (1.0, 1., True), (True, 1., True), (1, 1., 1),
                     (1, 1., True, 1)]:
            with self.subTest(args=args):
                before = policy.snapshot()
                with self.assertRaises(ValueError):
                    policy.observe(*args)
                self.assertEqual(before, policy.snapshot())

    def test_stopped_controller_rejects_further_observations(self):
        policy = controller(min_lr=.01, min_updates=0, stop_patience=1)
        policy.observe(0, 1., True)
        policy.observe(1, 1., True)
        before = policy.snapshot()
        with self.assertRaisesRegex(ValueError, 'already stopped'):
            policy.observe(2, 1., True)
        self.assertEqual(before, policy.snapshot())

    def test_config_rejects_unknown_malformed_and_unbounded_settings(self):
        invalid = [[], True, {'unknown': 1}, {'enabled': 1},
                   {'eval_interval': 0}, {'patience': True}, {'patience': 10001},
                   {'stop_patience': 0}, {'min_updates': -1}, {'min_updates': 10001},
                   {'max_growths': -1}, {'max_growths': 101}, {'min_delta': -1},
                   {'min_delta': 101}, {'min_delta': float('nan')},
                   {'lr_factor': 0}, {'lr_factor': 1}, {'min_lr': 0},
                   {'min_lr': .03}, {'max_lr': 1.1}, {'growth_factor': 1},
                   {'growth_factor': 101}, {'growth_factor': float('inf')}]
        for settings in invalid:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    AdaptiveFitConfig.from_dict(settings)

    def test_initial_lr_bounds_are_enabled_only_but_always_finite(self):
        for lr in (1e-6, .2):
            with self.subTest(lr=lr):
                with self.assertRaisesRegex(ValueError, 'within adaptive'):
                    controller(initial_lr=lr)
                self.assertEqual(AdaptiveFitController(None, initial_lr=lr).lr, lr)
        for lr in (0, -1, 1.1, float('inf'), float('nan'), True):
            with self.subTest(lr=lr):
                with self.assertRaises(ValueError):
                    AdaptiveFitController(None, initial_lr=lr)

    def test_direct_dataclass_configuration_is_validated(self):
        with self.assertRaises(ValueError):
            AdaptiveFitController(AdaptiveFitConfig(patience=0), initial_lr=.01)

    def test_state_is_deterministic_detached_json_and_replayable(self):
        trace = [(0, 2., False, False), (16, 2., True, True),
                 (32, 2., True, True), (48, 2., True, True),
                 (64, 1.5, True, False), (80, 1.5, True, False)]
        first, second = controller(), controller()
        for args in trace:
            self.assertEqual(first.observe(*args), second.observe(*args))
        snapshot = json.loads(json.dumps(first.snapshot(), allow_nan=False))
        self.assertEqual(snapshot, second.snapshot())
        restored = AdaptiveFitController.from_snapshot(snapshot)
        self.assertEqual(restored.observe(96, 1.5, True), first.observe(96, 1.5, True))
        snapshot['decisions'][0]['lr'] = .9
        snapshot['config']['patience'] = 1
        self.assertNotEqual(snapshot, second.snapshot())
        self.assertNotIn('decisions', first.observe(112, 1.5, True)['state'])

    def test_snapshot_replay_rejects_counter_history_and_nonfinite_tampering(self):
        policy = controller()
        policy.observe(0, 1., True)
        valid = policy.snapshot()
        for key, value in [('growths', 2), ('observation_count', True),
                           ('current_lr', float('nan')), ('schema', 'legacy')]:
            with self.subTest(key=key):
                bad = copy.deepcopy(valid)
                bad[key] = value
                with self.assertRaises(ValueError):
                    AdaptiveFitController.from_snapshot(bad)
        bad = copy.deepcopy(valid)
        del bad['decisions'][0]['train_loss']
        with self.assertRaises(ValueError):
            AdaptiveFitController.from_snapshot(bad)

    def test_maximum_update_and_event_count_are_bounded(self):
        policy = AdaptiveFitController(None, initial_lr=.01)
        for update in range(10001):
            policy.observe(update, 1., True)
        self.assertEqual(len(policy.snapshot()['decisions']), 10001)
        with self.assertRaises(ValueError):
            policy.observe(10001, 1., True)

    def test_no_rng_change_or_torch_dependency_or_selection_input(self):
        before = random.getstate()
        policy = controller(patience=1)
        for update in range(10):
            policy.observe(update, 2., True, True)
        self.assertEqual(before, random.getstate())
        self.assertEqual(list(inspect.signature(policy.observe).parameters),
                         ['update', 'train_loss', 'coverage_complete', 'ineffective_update'])
        code = ("import sys; sys.modules['torch'] = None; sys.modules['numpy'] = None; "
                "from mixtral_lab.adaptive_fit import AdaptiveFitController; "
                "p = AdaptiveFitController({'enabled': True}, initial_lr=.01); "
                "assert p.observe(0, 1., False)['lr'] == .01")
        completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
