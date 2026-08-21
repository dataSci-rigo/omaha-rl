"""
Differential test: MCCFR.showdown.showdown_payoff vs PokerEnv._payout_pots_hi_lo.

Fabricates HU terminal showdown states in the real env (pattern from
test/game/test_OmahaHiLoPayout.py) and asserts the pure function pays out the
exact same chips -- across random deals and constructed tie/odd-chip cases.
"""
import unittest

import numpy as np

from MCCFR.showdown import showdown_payoff
from PokerRL.game.games import FixedLimitOmahaHiLo


def _cards(spec):
    ranks = "23456789TJQKA"
    suits = "hdsc"
    return np.array([[ranks.index(c[0]), suits.index(c[1])] for c in spec.split()],
                    dtype=np.int8)


class TestShowdownPayoff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lut_holder = FixedLimitOmahaHiLo.get_lut_holder()
        args = FixedLimitOmahaHiLo.ARGS_CLS(
            n_seats=2, starting_stack_sizes_list=[2000] * 2,
            stack_randomization_range=(0, 0))
        cls.env = FixedLimitOmahaHiLo(env_args=args, lut_holder=cls.lut_holder,
                                      is_evaluating=True)

    def _env_payout(self, hand0_2d, hand1_2d, board_2d, pot):
        env = self.env
        env.reset()
        env.board = board_2d
        env.main_pot = pot
        env.side_pots = [0, 0]
        for p, hand in zip(env.seats, (hand0_2d, hand1_2d)):
            p.hand = hand
            p.folded_this_episode = False
            p.side_pot_rank = -1
            p.current_bet = 0
        stacks_before = [p.stack for p in env.seats]
        env._payout_pots()
        gains = tuple(p.stack - s for p, s in zip(env.seats, stacks_before))
        self.assertEqual(sum(gains), pot, "env did not conserve chips")
        return gains

    def _assert_match(self, hand0_2d, hand1_2d, board_2d, pot):
        env_gains = self._env_payout(hand0_2d, hand1_2d, board_2d, pot)
        ranks = [self.env.get_hand_rank(hand_2d=h, board_2d=board_2d)
                 for h in (hand0_2d, hand1_2d)]
        ours = showdown_payoff(ranks[0], ranks[1], pot)
        self.assertEqual(ours, env_gains,
                         f"ranks={ranks} pot={pot}: ours={ours} env={env_gains}")
        return ranks

    def test_random_deals(self):
        rng = np.random.default_rng(7)
        outcomes = set()
        for _ in range(500):
            deal = rng.permutation(52)
            to2d = self.lut_holder.get_2d_cards
            hand0, hand1 = to2d(deal[0:4]), to2d(deal[4:8])
            board = to2d(deal[8:13])
            pot = int(rng.integers(1, 25)) * 4  # even pots, like the real tree
            ranks = self._assert_match(hand0, hand1, board, pot)
            lo_exists = any(r[1] is not None for r in ranks)
            hi_tie = ranks[0][0] == ranks[1][0]
            outcomes.add((lo_exists, hi_tie))
        # random play must have covered both lo-present and lo-absent pots
        self.assertTrue(any(k[0] for k in outcomes))
        self.assertTrue(any(not k[0] for k in outcomes))

    def test_identical_hands_quarter_each_and_odd_chips(self):
        # Both players play the board-equivalent same hand: tie hi and tie lo.
        h0 = _cards("As 2s Kh Qd")
        h1 = _cards("Ah 2h Ks Qc")
        board = _cards("3d 4d 5c 8s 9h")  # both make wheel-low + straight hi
        for pot in (100, 101, 102, 103):
            self._assert_match(h0, h1, board, pot)

    def test_scoop_no_low_board(self):
        h0 = _cards("As Ks Ah Kd")
        h1 = _cards("Qs Qh Jc Td")
        board = _cards("9s 9d Tc Js Kc")  # no 3 low cards: no qualifying low
        for pot in (96, 97):
            self._assert_match(h0, h1, board, pot)

    def test_one_way_low_split(self):
        h0 = _cards("As 2d 3s 4c")   # nut low candidate
        h1 = _cards("Ks Kd Qs Jh")   # hi-only hand
        board = _cards("5h 6d 7s Kc 8c")
        for pot in (96, 97):
            self._assert_match(h0, h1, board, pot)


class TestShowdownPayoffMultiway(unittest.TestCase):
    """3-seat differential vs env._payout_pots(), incl. a folded seat whose
    chips are dead money and 3-way lo ties with odd chips."""

    @classmethod
    def setUpClass(cls):
        from MCCFR.showdown import showdown_payoff_multi
        cls.showdown_payoff_multi = staticmethod(showdown_payoff_multi)
        cls.lut_holder = FixedLimitOmahaHiLo.get_lut_holder()
        args = FixedLimitOmahaHiLo.ARGS_CLS(
            n_seats=3, starting_stack_sizes_list=[2000] * 3,
            stack_randomization_range=(0, 0))
        cls.env = FixedLimitOmahaHiLo(env_args=args, lut_holder=cls.lut_holder,
                                      is_evaluating=True)

    def _assert_match(self, hands_2d, board_2d, pot, folded):
        env = self.env
        env.reset()
        env.board = board_2d
        env.main_pot = pot
        env.side_pots = [0, 0, 0]
        for p, hand in zip(env.seats, hands_2d):
            p.hand = hand
            p.folded_this_episode = folded[p.seat_id]
            p.side_pot_rank = -1
            p.current_bet = 0
        before = [p.stack for p in env.seats]
        env._payout_pots()
        env_gains = tuple(p.stack - s for p, s in zip(env.seats, before))
        self.assertEqual(sum(env_gains), pot)

        ranks = [None if folded[p] else
                 env.get_hand_rank(hand_2d=hands_2d[p], board_2d=board_2d)
                 for p in range(3)]
        ours = self.showdown_payoff_multi(ranks, pot, 3)
        self.assertEqual(ours, env_gains, f"pot={pot} folded={folded}")

    def test_random_3way_and_dead_money(self):
        rng = np.random.default_rng(17)
        for i in range(300):
            deal = rng.permutation(52)
            to2d = self.lut_holder.get_2d_cards
            hands = [to2d(deal[4 * p:4 * p + 4]) for p in range(3)]
            board = to2d(deal[12:17])
            pot = int(rng.integers(1, 25)) * 4
            folded = [False, False, False]
            if i % 3 == 1:  # one seat folded: dead money, 2-way showdown
                folded[int(rng.integers(0, 3))] = True
            self._assert_match(hands, board, pot, folded)

    def test_three_way_lo_tie_odd_chips(self):
        # all three hold A2xx for the same nut low; distinct hi strength
        hands = [_cards("As 2s Ks Qs"), _cards("Ah 2h 9c 8d"),
                 _cards("Ad 2d 6h 6s")]
        board = _cards("3c 4c 5d Jh Jc")
        for pot in (99, 100, 101, 102):
            self._assert_match(hands, board, pot, [False] * 3)


if __name__ == '__main__':
    unittest.main()
