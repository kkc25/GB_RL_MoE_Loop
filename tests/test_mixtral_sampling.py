"""Document/position sampling tests, including a real randomly initialized HF model."""
import random
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import torch

from mixtral_lab.sampling import collect_stratified_contexts, plan_stratified_contexts


def row(name, length, mask=None):
    return {'id': name, 'input_ids': torch.arange(length, dtype=torch.long) % 30,
            'attention_mask': torch.ones(length, dtype=torch.long) if mask is None else torch.tensor(mask)}


class SamplingTests(unittest.TestCase):
    def test_imbalanced_documents_and_unique_positions(self):
        rows = [row('short', 3), row('long', 100), row('medium', 10)]
        plan = plan_stratified_contexts(rows, 20, 9, 'train')
        counts = {x['document_id']: x['selected_count'] for x in plan['documents']}
        self.assertEqual(counts, {'short': 2, 'long': 9, 'medium': 9})
        self.assertEqual(plan['requested_count'], plan['actual_count'])
        self.assertEqual(plan['document_coverage'], 1.)
        keys = {(x['document_id'], x['token_position']) for x in plan['contexts']}
        self.assertEqual(len(keys), 20)
        long_positions = {p for d, p in keys if d == 'long'}
        self.assertNotEqual(long_positions, set(range(9)))

    def test_seed_local_repeatable_and_content_bound(self):
        rows = [row('a', 30), row('b', 30)]
        python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        first = plan_stratified_contexts(rows, 12, 91, 'train')
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(first, plan_stratified_contexts(rows, 12, 91, 'train'))
        self.assertNotEqual(first['contexts'], plan_stratified_contexts(rows, 12, 92, 'train')['contexts'])
        rows[0]['input_ids'][0] = 29
        self.assertNotEqual(first['sampling_sha256'], plan_stratified_contexts(rows, 12, 91, 'train')['sampling_sha256'])
        self.assertNotEqual(first['sampling_sha256'], plan_stratified_contexts(rows, 12, 91, 'selection')['sampling_sha256'])

    def test_masks_shift_empty_documents_and_small_budget(self):
        rows = [row('masked', 8, [0, 1, 1, 0, 1, 1, 1, 0]), row('empty', 1)]
        plan = plan_stratified_contexts(rows, 3, 8, 'calibration')
        self.assertEqual({x['token_position'] for x in plan['contexts']}, {1, 4, 5})
        self.assertTrue(all(x['target'] == x['token_position'] + 1 for x in plan['contexts']))
        self.assertEqual(plan['eligible_document_count'], 1)
        self.assertEqual(plan['documents'][1]['selected_count'], 0)
        many = [row(str(i), 5) for i in range(20)]
        small = plan_stratified_contexts(many, 3, 8, 'train')
        self.assertEqual(small['covered_document_count'], 3)
        self.assertEqual(small['document_coverage'], 3 / 20)
        self.assertNotEqual({x['document_id'] for x in small['contexts']}, {'0', '1', '2'})

    def test_invalid_and_shortage_fail_before_capture(self):
        for rows, count, role in [([], 1, 'train'), ([row('a', 1)], 1, 'train'),
                                  ([row('a', 3)], 3, 'train'), ([row('a', 3)], 0, 'train'),
                                  ([row('a', 3)], 1, 'test'), ([row('a', 3)], True, 'train'),
                                  ([row('a', 3), row('a', 3)], 1, 'train'),
                                  ([row('a', 3, [1, 2, 1])], 1, 'train')]:
            with self.subTest(count=count, role=role), self.assertRaises(ValueError):
                collect_stratified_contexts(None, rows, count, 7, role)

    @unittest.skipUnless(find_spec('transformers'), 'requires transformers')
    def test_real_hf_capture_once_shift_and_contiguous(self):
        from tests.test_mixtral_adapter import tiny
        from mixtral_lab.adapter import MixtralAdapter
        adapter = MixtralAdapter(tiny())
        rows = [row('a', 8, [0, 1, 1, 1, 1, 1, 0, 0]), row('b', 11)]
        state = torch.random.get_rng_state().clone()
        with patch.object(adapter, 'capture', wraps=adapter.capture) as capture:
            h0, targets, plan = collect_stratified_contexts(adapter, rows, 8, 3, 'train')
            self.assertEqual(capture.call_count, 2)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(h0.shape, (8, 16))
        self.assertEqual(targets.device, h0.device)
        self.assertTrue(h0.is_contiguous() and targets.is_contiguous())
        self.assertFalse(h0.requires_grad or targets.requires_grad)
        expected = {r['id']: adapter.capture(r['input_ids'][None], r['attention_mask'][None])[0] for r in rows}
        for index, context in enumerate(plan['contexts']):
            torch.testing.assert_close(h0[index], expected[context['document_id']][context['token_position']], rtol=0, atol=0)
            self.assertEqual(targets[index].item(), context['target'])
        again = collect_stratified_contexts(adapter, rows, 8, 3, 'train')
        self.assertTrue(torch.equal(again[0], h0) and torch.equal(again[1], targets))
        self.assertEqual(again[2], plan)


if __name__ == '__main__':
    unittest.main()
