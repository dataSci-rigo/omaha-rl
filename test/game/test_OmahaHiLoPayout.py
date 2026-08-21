"""
Env-level tests for the Omaha Hi/Lo split-pot payout (_payout_pots_hi_lo /
_get_hi_lo_shares in PokerEnv) at 2, 3, and side-pot configurations.

test_OmahaHiLoEval.py covers only the C++ evaluator and never constructs a
PokerEnv; until this file, the split-pot logic -- the most intricate code in the
variant, and the thing 6-max training depends on -- had zero coverage. Each test
fabricates a terminal showdown state directly (board, hands, pots, side_pot_rank)
and asserts exact chip movement plus conservation.

Card encoding (matches test_OmahaHiLoEval.py):
    rank: 0='2', 1='3', ..., 8='T', 9='J', 10='Q', 11='K', 12='A'
    suit: 0='h', 1='d', 2='s', 3='c'
"""
import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game.games import FixedLimitOmahaHiLo


def _cards(spec):
    """'Ah 2c 6h 7c' -> np.int8 array [[rank, suit], ...]"""
    ranks = "23456789TJQKA"
    suits = "hdsc"
    return np.array([[ranks.index(c[0]), suits.index(c[1])] for c in spec.split()],
                    dtype=np.int8)


class _PayoutTestBase(TestCase):
    N_SEATS = 3

    @classmethod
    def setUpClass(cls):
        cls.lut_holder = FixedLimitOmahaHiLo.get_lut_holder()

    def make_env(self, n_seats=None):
        n = n_seats or self.N_SEATS
        args = FixedLimitOmahaHiLo.ARGS_CLS(n_seats=n,
                                            starting_stack_sizes_list=[1000] * n,
                                            stack_randomization_range=(0, 0))
        env = FixedLimitOmahaHiLo(env_args=args, lut_holder=self.lut_holder,
                                  is_evaluating=True)
        env.reset()
        return env

    def run_showdown(self, env, board, hands, main_pot, side_pots=None,
                     side_pot_ranks=None, folded=None):
        """Fabricates a terminal state and runs the hi/lo payout. Returns the
        chips each seat gained. Also asserts chip conservation."""
        n = env.N_SEATS
        env.board = _cards(board)
        env.main_pot = main_pot
        env.side_pots = list(side_pots) if side_pots else [0] * n
        for i, p in enumerate(env.seats):
            p.hand = _cards(hands[i])
            p.folded_this_episode = bool(folded[i]) if folded else False
            p.side_pot_rank = side_pot_ranks[i] if side_pot_ranks else -1
            p.current_bet = 0

        total_pot = main_pot + sum(env.side_pots)
        stacks_before = [p.stack for p in env.seats]
        env._payout_pots()
        gains = [p.stack - s0 for p, s0 in zip(env.seats, stacks_before)]

        self.assertEqual(sum(gains), total_pot,
                         f"chips not conserved: pot={total_pot}, paid out={sum(gains)}")
        self.assertEqual(env.main_pot, 0)
        self.assertEqual(sum(env.side_pots), 0)
        return gains

    # ------------------------------------------------------------------ tests

    def test_multiway_scoop_no_qualifying_low(self):
        # Board has zero cards <= 8, so no low is possible anywhere: the best hi
        # hand must scoop the whole pot. P0 makes quad nines (9h9c + 9d9s on
        # board); P1 kings full; P2 junk.
        gains = self.run_showdown(
            self.make_env(),
            board="Ks Qs Th 9d 9s",
            hands=["9h 9c 3d 4c",
                   "Kd Kc 5h 6h",
                   "7d 8c 2s 3s"],
            main_pot=300,
        )
        self.assertEqual(gains, [300, 0, 0])

    def test_quartered_pot(self):
        # The classic Hi/Lo quartering. Board 3h 4d 5s Kh Qd.
        #   P0: Ah 2c 6h 7c -> hi = 7-high straight (67+345), lo = nut 5432A (A2+345)
        #   P1: As 2d 9h 9c -> hi = 5-high straight only, lo = same nut 5432A
        #   P2: Kd Kc Th Tc -> trip kings, no low
        # Hi half -> P0 alone. Lo half -> split P0/P1. P0 = 3/4, P1 = 1/4.
        gains = self.run_showdown(
            self.make_env(),
            board="3h 4d 5s Kh Qd",
            hands=["Ah 2c 6h 7c",
                   "As 2d 9h 9c",
                   "Kd Kc Th Tc"],
            main_pot=400,
        )
        self.assertEqual(gains, [300, 100, 0])

    def test_side_pot_short_stack_has_only_low(self):
        # P2 is all-in short: eligible for the main pot only (side_pot_rank=-1);
        # P0/P1 also contest side pot 0 (side_pot_rank=0).
        # Board 3h 4d 8s Kh Qd (low possible: 3,4,8).
        #   P0: Kd Kc Th Tc -> trip kings, no low
        #   P1: Qs Qc Jh Jc -> trip queens, no low
        #   P2: Ah 2c 7h 9c -> weak hi, but the ONLY qualifying low (8432A)
        # Main pot 300: hi half 150 -> P0, lo half 150 -> P2.
        # Side pot 200 (P0 vs P1, no low): P0 scoops.
        gains = self.run_showdown(
            self.make_env(),
            board="3h 4d 8s Kh Qd",
            hands=["Kd Kc Th Tc",
                   "Qs Qc Jh Jc",
                   "Ah 2c 7h 9c"],
            main_pot=300,
            side_pots=[200, 0, 0],
            side_pot_ranks=[0, 0, -1],
        )
        self.assertEqual(gains, [350, 0, 150])

    def test_folded_player_gets_nothing(self):
        # P1 folded holding what would be the nut low + nut hi; must be excluded.
        # P0 vs P2 only: P0 wins hi, P2 has the only remaining low.
        gains = self.run_showdown(
            self.make_env(),
            board="3h 4d 8s Kh Qd",
            hands=["Kd Kc Th Tc",
                   "Ah 2h 5h 6h",
                   "As 2c 7h 9c"],
            main_pot=300,
            folded=[False, True, False],
        )
        self.assertEqual(gains[1], 0)
        self.assertEqual(gains, [150, 0, 150])

    def test_scoop_when_one_player_has_both_halves(self):
        # P0 holds both the best hi (wheel is beaten -- P0 has 67 for the 7-high
        # straight) AND ties nobody on the nut low: full scoop of a 3-way pot.
        gains = self.run_showdown(
            self.make_env(),
            board="3h 4d 5s Kh Qd",
            hands=["Ah 2c 6h 7c",
                   "Kd Kc Th Tc",
                   "Qs Qc Jh Jc"],
            main_pot=600,
        )
        self.assertEqual(gains, [600, 0, 0])

    def test_odd_chip_goes_to_hi_deterministically(self):
        # Pot of 25 split hi/lo: the standard O8 rule gives the odd chip to the
        # HIGH side (hi_half = ceil(pot/2) = 13, lo_half = 12), deterministically.
        # This is also what pokerkit does, enabling exact differential testing.
        gains = self.run_showdown(
            self.make_env(),
            board="3h 4d 8s Kh Qd",
            hands=["Kd Kc Th Tc",     # hi winner
                   "Qs Qc Jh Jc",     # loses both ways
                   "Ah 2c 7h 9c"],    # only low
            main_pot=25,
        )
        self.assertEqual(gains, [13, 0, 12])

    def test_lo_half_remainder_goes_left_of_button(self):
        # Pot 10: hi half 5 -> P0. Lo half 5 split between two tied nut lows
        # (P1, P2): 2 each, remainder 1 to the earliest seat left of the button
        # (button is seat 0 at 3+ seats, so P1 gets it).
        gains = self.run_showdown(
            self.make_env(),
            board="3h 4d 8s Kh Qd",
            hands=["Kd Kc Th Tc",     # trip kings, hi winner, no low
                   "Ah 2c 9h 9c",     # nut low 8432A, hi = pair of nines
                   "As 2d Jh Jc"],    # identical nut low, hi = pair of jacks
            main_pot=10,
        )
        self.assertEqual(gains, [5, 3, 2])

    def test_heads_up_split_matches_three_way_logic(self):
        # Same payout path must hold at N_SEATS=2 (regression guard: the hi-only
        # code has a 2-seat fast path; hi/lo must not).
        gains = self.run_showdown(
            self.make_env(n_seats=2),
            board="3h 4d 8s Kh Qd",
            hands=["Kd Kc Th Tc",
                   "Ah 2c 7h 9c"],
            main_pot=200,
        )
        self.assertEqual(gains, [100, 100])


if __name__ == '__main__':
    unittest.main()
