"""
Lossless preflop suit-isomorphism abstraction + lean hole-card index LUTs.

The canonicalization is the same 24-permutation lexicographic-minimum used by
_LutGetterPLO.get_range_idx_to_private_obs_LUT (PokerRL/game/_/look_up_table.py)
-- see the long comment there for why suit ISOMORPHISM (16,432 classes, exact
by Burnside over S4) is right for Omaha and suit-stripping is not. Here we
need the class id itself rather than a one-hot observation row, and we skip
building the full LutHolderPLO (its eager obs LUTs cost seconds and are
neural-net-only plumbing).

Card encoding matches PokerRL: 1-D id c = rank * 4 + suit, ranks 0..12, suits
0..3; a range idx is the position of the sorted 4-card combo in
itertools.combinations(range(52), 4) order (same ordering as _LutGetterPLO's
idx->hole-cards LUT, asserted in test_preflop_iso.py).
"""
import itertools

import numpy as np

N_CARDS = 52
N_SUITS = 4
N_RANKS = 13
RANGE_SIZE = 270725       # C(52, 4)
N_PREFLOP_CLASSES = 16432


def build_idx_to_hole_cards():
    """int8[270725, 4] -- sorted 1-D card ids per range idx."""
    flat = np.fromiter(itertools.chain.from_iterable(
        itertools.combinations(range(N_CARDS), 4)),
        dtype=np.int8, count=RANGE_SIZE * 4)
    return flat.reshape(RANGE_SIZE, 4)


def build_hole_cards_to_idx(idx_to_hc=None):
    """Dense int32[52^4] inverse: packed sorted 4-card key -> range idx.

    Vectorized (milliseconds) unlike the pure-Python quadruple loop in
    _LutGetterPLO.get_hole_card_2_idx_LUT. Unused keys stay -1.
    """
    if idx_to_hc is None:
        idx_to_hc = build_idx_to_hole_cards()
    c = idx_to_hc.astype(np.int64)
    keys = ((c[:, 0] * N_CARDS + c[:, 1]) * N_CARDS + c[:, 2]) * N_CARDS + c[:, 3]
    inv = np.full(N_CARDS ** 4, -1, dtype=np.int32)
    inv[keys] = np.arange(RANGE_SIZE, dtype=np.int32)
    return inv


def range_idx_of(hole_cards_1d, hole_cards_to_idx):
    """Range idx for 4 1-D card ids (any order)."""
    a, b, c, d = np.sort(np.asarray(hole_cards_1d, dtype=np.int64))
    return int(hole_cards_to_idx[((a * N_CARDS + b) * N_CARDS + c) * N_CARDS + d])


def build_preflop_class_map(idx_to_hc=None):
    """int32[270725] -> [0, 16432): suit-isomorphism class per range idx."""
    if idx_to_hc is None:
        idx_to_hc = build_idx_to_hole_cards()
    cards = idx_to_hc.astype(np.int32)
    ranks = cards // N_SUITS                          # (R, 4)
    suits = cards % N_SUITS                           # (R, 4)

    best_key = None
    for perm in itertools.permutations(range(N_SUITS)):
        p = np.array(perm, dtype=np.int32)
        codes = np.sort(ranks * N_SUITS + p[suits], axis=1)
        key = (((codes[:, 0] * 64 + codes[:, 1]) * 64 + codes[:, 2]) * 64
               + codes[:, 3])
        best_key = key if best_key is None else np.minimum(best_key, key)

    classes, class_ids = np.unique(best_key, return_inverse=True)
    assert len(classes) == N_PREFLOP_CLASSES, len(classes)
    return class_ids.astype(np.int32)
