"""
Tests for MCCFR.abstraction.preflop_iso and the M1 equity machinery.

- the class map is invariant under all 24 suit permutations and has exactly
  16,432 classes (Burnside over S4)
- our lean idx<->hole-cards LUTs match _LutGetterPLO's ordering
- rollout equity orders premium vs junk hands sensibly
"""
import unittest

import numpy as np

from MCCFR.abstraction.equity import RolloutFeatures
from MCCFR.abstraction.preflop_iso import (build_idx_to_hole_cards,
                                           build_hole_cards_to_idx,
                                           build_preflop_class_map,
                                           range_idx_of, N_PREFLOP_CLASSES,
                                           RANGE_SIZE)


def _card(rank, suit):
    return rank * 4 + suit


class TestPreflopIso(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.idx_to_hc = build_idx_to_hole_cards()
        cls.hc_to_idx = build_hole_cards_to_idx(cls.idx_to_hc)
        cls.class_map = build_preflop_class_map(cls.idx_to_hc)

    def test_class_count(self):
        self.assertEqual(len(np.unique(self.class_map)), N_PREFLOP_CLASSES)
        self.assertEqual(len(self.class_map), RANGE_SIZE)

    def test_suit_permutation_invariance(self):
        import itertools
        rng = np.random.default_rng(3)
        for _ in range(200):
            cards = rng.choice(52, size=4, replace=False)
            base = self.class_map[range_idx_of(cards, self.hc_to_idx)]
            ranks, suits = cards // 4, cards % 4
            for perm in itertools.permutations(range(4)):
                p = np.array(perm)
                permuted = ranks * 4 + p[suits]
                self.assertEqual(
                    self.class_map[range_idx_of(permuted, self.hc_to_idx)],
                    base)

    def test_distinct_suit_patterns_stay_distinct(self):
        # As2s3h4h (double-suited) vs As2h3d4c (rainbow): same ranks,
        # strategically different -> different classes.
        ds = [_card(12, 2), _card(0, 2), _card(1, 0), _card(2, 0)]
        rb = [_card(12, 2), _card(0, 0), _card(1, 1), _card(2, 3)]
        self.assertNotEqual(self.class_map[range_idx_of(ds, self.hc_to_idx)],
                            self.class_map[range_idx_of(rb, self.hc_to_idx)])

    def test_matches_lut_getter_ordering(self):
        from PokerRL.game.games import FixedLimitOmahaHiLo
        lh = FixedLimitOmahaHiLo.get_lut_holder()
        rng = np.random.default_rng(5)
        for idx in rng.integers(0, RANGE_SIZE, size=50):
            ours = self.idx_to_hc[idx]
            theirs = lh.get_1d_hole_cards_from_range_idx(int(idx))
            np.testing.assert_array_equal(np.sort(ours), np.sort(theirs))
            self.assertEqual(
                range_idx_of(ours, self.hc_to_idx),
                lh.get_range_idx_from_hole_cards(lh.get_2d_cards(ours)))


class TestRolloutFeatures(unittest.TestCase):

    def test_equity_ordering_and_feature_ranges(self):
        rf = RolloutFeatures(rng=np.random.default_rng(11))
        premium = [_card(12, 0), _card(0, 0), _card(1, 1), _card(12, 1)]  # AA23 ds
        junk = [_card(7, 0), _card(5, 1), _card(1, 2), _card(11, 3)]      # 973K rb
        share_p, feats_p = rf.features(premium, [], n_rollouts=400)
        share_j, feats_j = rf.features(junk, [], n_rollouts=400)
        self.assertGreater(share_p, share_j + 0.1)
        for feats in (feats_p, feats_j):
            self.assertTrue(np.all(feats >= 0.0) and np.all(feats <= 1.0))
        self.assertGreater(feats_p[2], feats_j[2])  # scoop prob


if __name__ == '__main__':
    unittest.main()
