"""
Monte Carlo rollout features for a 4-card O8 hand on a partial board.

Shared by the M1 bucketer (scalar expected pot share) and the M2 k-means
bucketer (4-dim outcome-distribution vector: hi equity, lo equity, scoop
probability, quarter risk). Rollouts play the hand to showdown vs ONE uniform
random opponent hand with a uniform random board completion, using the same
nit evaluator and payout rules as training showdowns.

These features only ASSIGN infosets to buckets -- actual training showdowns
use the exact dealt hands -- so modest rollout counts are fine: the noise a
bucketer needs to beat is bucket spacing, not payout accuracy.
"""
import numpy as np

from MCCFR.showdown import showdown_payoff
from PokerRL.game._.cpp_wrappers.CppHandEvalHiLo import CppHandEvalHiLo

# 1-D card id = rank * 4 + suit (PokerRL convention, verified vs lut_holder)
_CARD_2D = np.stack([np.arange(52, dtype=np.int8) // 4,
                     np.arange(52, dtype=np.int8) % 4], axis=1)


class RolloutFeatures:
    """One evaluator instance + scratch buffers; not thread-safe."""

    N_FEATURES = 4  # hi_eq, lo_eq, scoop_prob, quarter_risk

    def __init__(self, rng=None):
        self._evaluator = CppHandEvalHiLo()
        self._rng = rng if rng is not None else np.random.default_rng()

    def features(self, hole_1d, board_1d, n_rollouts):
        """Returns (mean_pot_share, np.float64[4] outcome features).

        hole_1d: 4 card ids. board_1d: 0/3/4/5 dealt board card ids.
        """
        hole_1d = np.asarray(hole_1d, dtype=np.int8)
        board_1d = np.asarray(board_1d, dtype=np.int8)
        rng = self._rng
        evaluate = self._evaluator.get_hand_rank_52_plo8

        seen = np.zeros(52, dtype=bool)
        seen[hole_1d] = True
        seen[board_1d] = True
        unseen = np.flatnonzero(~seen).astype(np.int8)
        n_missing = 5 - len(board_1d)

        hero_2d = _CARD_2D[hole_1d]
        board_2d = np.empty((5, 2), dtype=np.int8)
        board_2d[:len(board_1d)] = _CARD_2D[board_1d]

        share_sum = hi_sum = scoop_n = quarter_n = 0.0
        lo_share_sum = 0.0
        lo_n = 0

        for _ in range(n_rollouts):
            draw = rng.choice(len(unseen), size=4 + n_missing, replace=False)
            opp = unseen[draw[:4]]
            if n_missing:
                board_2d[len(board_1d):] = _CARD_2D[unseen[draw[4:]]]

            rank_h = evaluate(hand_2d=hero_2d, board_2d=board_2d)
            rank_o = evaluate(hand_2d=_CARD_2D[opp], board_2d=board_2d)

            chips_h, _ = showdown_payoff(rank_h, rank_o, pot=4)
            share = chips_h / 4.0
            share_sum += share
            scoop_n += share == 1.0
            quarter_n += share == 0.25

            if rank_h[0] > rank_o[0]:
                hi_sum += 1.0
            elif rank_h[0] == rank_o[0]:
                hi_sum += 0.5

            if rank_h[1] is not None or rank_o[1] is not None:
                lo_n += 1
                if rank_o[1] is None or (rank_h[1] is not None
                                         and rank_h[1] > rank_o[1]):
                    lo_share_sum += 1.0
                elif rank_h[1] is not None and rank_h[1] == rank_o[1]:
                    lo_share_sum += 0.5

        r = float(n_rollouts)
        feats = np.array([hi_sum / r,
                          (lo_share_sum / lo_n) if lo_n else 0.0,
                          scoop_n / r,
                          quarter_n / r], dtype=np.float64)
        return share_sum / r, feats
