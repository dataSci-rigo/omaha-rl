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

from MCCFR.showdown import showdown_payoff_multi
from PokerRL.game._.cpp_wrappers.CppHandEvalHiLo import CppHandEvalHiLo

# 1-D card id = rank * 4 + suit (PokerRL convention, verified vs lut_holder)
_CARD_2D = np.stack([np.arange(52, dtype=np.int8) // 4,
                     np.arange(52, dtype=np.int8) % 4], axis=1)


class RolloutFeatures:
    """One evaluator instance + scratch buffers; not thread-safe."""

    N_FEATURES = 4  # hi_eq, lo_eq, scoop_prob, quarter_risk

    def __init__(self, rng=None, n_opponents=1):
        self._evaluator = CppHandEvalHiLo()
        self._rng = rng if rng is not None else np.random.default_rng()
        self.n_opponents = n_opponents

    def features(self, hole_1d, board_1d, n_rollouts):
        """Returns (mean_pot_share, np.float64[4] outcome features).

        hole_1d: 4 card ids. board_1d: 0/3/4/5 dealt board card ids.
        Rollouts play vs self.n_opponents uniform random hands; the pot
        share is the hero's exact multiway hi/lo share.
        """
        hole_1d = np.asarray(hole_1d, dtype=np.int8)
        board_1d = np.asarray(board_1d, dtype=np.int8)
        rng = self._rng
        evaluate = self._evaluator.get_hand_rank_52_plo8
        n_opp = self.n_opponents
        n_seats = n_opp + 1
        pot = 4 * n_seats  # divisible by 4 so quarter shares stay exact

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
            draw = rng.choice(len(unseen), size=4 * n_opp + n_missing,
                              replace=False)
            if n_missing:
                board_2d[len(board_1d):] = _CARD_2D[unseen[draw[4 * n_opp:]]]

            rank_h = evaluate(hand_2d=hero_2d, board_2d=board_2d)
            ranks = [rank_h] + [
                evaluate(hand_2d=_CARD_2D[unseen[draw[4 * i:4 * i + 4]]],
                         board_2d=board_2d) for i in range(n_opp)]

            chips = showdown_payoff_multi(ranks, pot, n_seats)
            share = chips[0] / pot
            share_sum += share
            scoop_n += share == 1.0
            quarter_n += share == 0.25

            best_opp_hi = max(r[0] for r in ranks[1:])
            if rank_h[0] > best_opp_hi:
                hi_sum += 1.0
            elif rank_h[0] == best_opp_hi:
                hi_sum += 0.5

            opp_los = [r[1] for r in ranks[1:] if r[1] is not None]
            if rank_h[1] is not None or opp_los:
                lo_n += 1
                best_opp_lo = max(opp_los) if opp_los else None
                if rank_h[1] is not None and (best_opp_lo is None
                                              or rank_h[1] > best_opp_lo):
                    lo_share_sum += 1.0
                elif rank_h[1] is not None and rank_h[1] == best_opp_lo:
                    lo_share_sum += 0.5

        r = float(n_rollouts)
        feats = np.array([hi_sum / r,
                          (lo_share_sum / lo_n) if lo_n else 0.0,
                          scoop_n / r,
                          quarter_n / r], dtype=np.float64)
        return share_sum / r, feats
