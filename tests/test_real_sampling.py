"""Regression checks for the selected full-pool real-reference protocol."""
import unittest

from driftaudio.real_sampling import real_indices


class RealSamplingTests(unittest.TestCase):
    def test_step_indexed_without_replacement(self):
        a = real_indices(1, population=100, count=20)
        b = real_indices(1, population=100, count=20)
        c = real_indices(2, population=100, count=20)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(len(set(a.tolist())), 20)
        self.assertTrue(all(0 <= i < 100 for i in a.tolist()))
        self.assertNotEqual(a.tolist(), c.tolist())

    def test_invalid_requests(self):
        for args in [(0, 100, 20), (1, 10, 11), (1, 10, 0)]:
            with self.assertRaises(ValueError):
                real_indices(*args)


if __name__ == '__main__':
    unittest.main()
