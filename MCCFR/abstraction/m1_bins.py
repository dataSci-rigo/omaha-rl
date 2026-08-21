"""
Milestone-1 bucketer: lossless preflop suit isomorphism + uniform postflop
equity bins.

Postflop bucket = floor(expected_pot_share * K), pot share estimated by
RolloutFeatures vs one uniform random opponent. Coarse by design -- M1's job
is a working, inspectable end-to-end MCCFR loop; M2 replaces this class with
k-means over outcome-distribution features behind the same interface.
"""
import numpy as np

from MCCFR.abstraction.equity import RolloutFeatures
from MCCFR.abstraction.preflop_iso import (build_idx_to_hole_cards,
                                           build_hole_cards_to_idx,
                                           build_preflop_class_map,
                                           range_idx_of, N_PREFLOP_CLASSES)


class M1Bucketer:
    """Bucketer interface used by the solver:

    .K                       -- buckets per street, index 0 = preflop
    .preflop_class(range_idx)
    .bucket(street, hole_1d, board_1d)
    """

    def __init__(self, k_postflop=50, n_rollouts=32, rng=None,
                 cache_max_entries=2_000_000):
        self.K = [N_PREFLOP_CLASSES, k_postflop, k_postflop, k_postflop]
        self.n_rollouts = n_rollouts
        self._features = RolloutFeatures(rng=rng)
        idx_to_hc = build_idx_to_hole_cards()
        self._hc_to_idx = build_hole_cards_to_idx(idx_to_hc)
        self._preflop_class = build_preflop_class_map(idx_to_hc)
        # cache: (street, packed board key) -> {range_idx: bucket}
        self._cache = {}
        self._cache_entries = 0
        self._cache_max = cache_max_entries
        self.n_misses = 0
        self.n_hits = 0

    def preflop_class(self, range_idx):
        return int(self._preflop_class[range_idx])

    def range_idx(self, hole_1d):
        return range_idx_of(hole_1d, self._hc_to_idx)

    def bucket(self, street, hole_1d, board_1d):
        board_key = 0
        for c in sorted(int(c) for c in board_1d):
            board_key = board_key * 52 + c
        hand_key = self.range_idx(hole_1d)

        per_board = self._cache.setdefault((street, board_key), {})
        hit = per_board.get(hand_key)
        if hit is not None:
            self.n_hits += 1
            return hit

        self.n_misses += 1
        share, _ = self._features.features(hole_1d, board_1d, self.n_rollouts)
        k = self.K[street]
        b = min(int(share * k), k - 1)

        if self._cache_entries >= self._cache_max:
            self._cache = {(street, board_key): per_board}
            self._cache_entries = len(per_board)
        per_board[hand_key] = b
        self._cache_entries += 1
        return b
