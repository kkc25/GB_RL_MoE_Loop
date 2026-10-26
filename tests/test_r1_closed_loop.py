import unittest

import torch

from experiments.r1_closed_loop import position_cross_entropy, tensor_sha256


class ClosedLoopHelperTests(unittest.TestCase):
    def test_position_ce_preserves_explicit_batch_dimensions(self):
        logits = torch.randn(3, 32)
        targets = torch.tensor([1, 2, 3])
        expected = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        for index in range(3):
            actual = position_cross_entropy(logits[index : index + 1], targets[index : index + 1])
            self.assertEqual(actual.shape, (1,))
            self.assertAlmostEqual(float(actual[0]), float(expected[index]), places=6)

    def test_position_ce_rejects_ambiguous_shapes(self):
        with self.assertRaises(ValueError):
            position_cross_entropy(torch.randn(32), torch.tensor([1]))
        with self.assertRaises(ValueError):
            position_cross_entropy(torch.randn(2, 32), torch.tensor([1]))

    def test_tensor_hash_is_content_sensitive(self):
        self.assertEqual(tensor_sha256(torch.ones(2)), tensor_sha256(torch.ones(2)))
        self.assertNotEqual(tensor_sha256(torch.ones(2)), tensor_sha256(torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
