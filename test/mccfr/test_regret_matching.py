"""Unit tests for regret matching and linear-weight/rescale arithmetic."""
import unittest

import numpy as np

from MCCFR.solver import regret_matching


class TestRegretMatching(unittest.TestCase):

    def test_all_negative_gives_uniform_over_legal(self):
        r = np.array([-5.0, -1.0, -3.0])
        mask = np.array([True, True, True])
        np.testing.assert_allclose(regret_matching(r, mask), [1/3] * 3)

    def test_respects_legal_mask(self):
        r = np.array([10.0, -1.0, 5.0])
        mask = np.array([False, True, True])  # e.g. nothing to call: no FOLD
        sigma = regret_matching(r, mask)
        self.assertEqual(sigma[0], 0.0)
        np.testing.assert_allclose(sigma, [0.0, 0.0, 1.0])

    def test_all_negative_with_mask(self):
        r = np.array([-1.0, -2.0, -3.0])
        mask = np.array([False, True, True])
        np.testing.assert_allclose(regret_matching(r, mask), [0.0, 0.5, 0.5])

    def test_mixed_normalization(self):
        r = np.array([3.0, 1.0, -7.0])
        mask = np.array([True, True, True])
        np.testing.assert_allclose(regret_matching(r, mask), [0.75, 0.25, 0.0])

    def test_scale_invariance(self):
        # rescaling regrets by any positive constant leaves the policy fixed --
        # the property the solver's periodic table rescale relies on
        r = np.array([3.0, 1.0, -7.0])
        mask = np.array([True, True, True])
        np.testing.assert_allclose(regret_matching(r, mask),
                                   regret_matching(r * 1e-6, mask))


if __name__ == '__main__':
    unittest.main()
