import random
import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game._.cpp_wrappers.CppHandEvalHiLo import CppHandEvalHiLo

# rank: 0='2', 1='3', ..., 8='T', 9='J', 10='Q', 11='K', 12='A'
# suit: 0='h', 1='d', 2='s', 3='c'  (arbitrary but must be consistent per card)
_RANKS = "23456789TJQKA"
_SUITS = "hdsc"


class TestOmahaHiLoEval(TestCase):

    def setUp(self):
        self.ev = CppHandEvalHiLo()

    def test_no_qualifying_low(self):
        # Ah Ac Kd Kc vs 2h3h4h5h6h: only the two aces are low-eligible hole cards,
        # but using both creates a pair, disqualifying any low hand.
        hand = np.array([[12, 0], [12, 3], [11, 1], [11, 3]], dtype=np.int8)
        board = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.int8)
        hi, lo = self.ev.get_hand_rank_52_plo8(hand_2d=hand, board_2d=board)
        self.assertIsInstance(hi, int)
        self.assertIsNone(lo)

    def test_nut_low_and_wheel_straight(self):
        # Ac 2d Kh Qh vs 3c4d5h9sTc: A2 + 345 board makes the wheel (5-high straight)
        # for hi, and the nut low (5432A).
        hand = np.array([[12, 3], [0, 1], [11, 0], [10, 0]], dtype=np.int8)
        board = np.array([[1, 3], [2, 1], [3, 0], [7, 2], [8, 3]], dtype=np.int8)
        hi, lo = self.ev.get_hand_rank_52_plo8(hand_2d=hand, board_2d=board)
        self.assertIsInstance(hi, int)
        self.assertIsNotNone(lo)

    def test_forced_worse_low_when_only_two_hole_ranks_qualify(self):
        # Ac 7d 9h 9c vs 2c3d4h5s6c: 9s don't qualify for low, so the only usable
        # low hole cards are A and 7, forcing a 7-low even though the board would
        # otherwise support a much better low.
        hand = np.array([[12, 3], [5, 1], [7, 0], [7, 3]], dtype=np.int8)
        board = np.array([[0, 3], [1, 1], [2, 0], [3, 2], [4, 3]], dtype=np.int8)
        hi, lo = self.ev.get_hand_rank_52_plo8(hand_2d=hand, board_2d=board)
        self.assertIsNotNone(lo)

        # A strictly better low (e.g. a wheel) must always outrank this forced 7-low.
        nut_low_hand = np.array([[12, 3], [0, 1], [11, 0], [10, 0]], dtype=np.int8)
        nut_low_board = np.array([[1, 3], [2, 1], [3, 0], [7, 2], [8, 3]], dtype=np.int8)
        _, nut_lo = self.ev.get_hand_rank_52_plo8(hand_2d=nut_low_hand, board_2d=nut_low_board)
        self.assertGreater(nut_lo, lo)

    def test_undealt_board_cards_are_skipped(self):
        # Flop-only board (turn/river marked as not dealt) must not crash and must
        # only use the 3 dealt cards.
        hand = np.array([[12, 3], [0, 1], [11, 0], [10, 0]], dtype=np.int8)
        board_flop_only = np.array([[1, 3], [2, 1], [3, 0], [-127, -127], [-127, -127]], dtype=np.int8)
        hi, lo = self.ev.get_hand_rank_52_plo8(hand_2d=hand, board_2d=board_flop_only)
        self.assertIsInstance(hi, int)
        self.assertIsNotNone(lo)

    def test_hi_rank_ordering(self):
        # A stronger hi hand (trip aces) must outrank a weaker one (one pair) from
        # this same evaluator's scale.
        trips_hand = np.array([[12, 0], [12, 1], [3, 2], [4, 3]], dtype=np.int8)
        trips_board = np.array([[12, 3], [6, 0], [7, 1], [8, 2], [9, 3]], dtype=np.int8)
        pair_hand = np.array([[12, 0], [11, 1], [3, 2], [4, 3]], dtype=np.int8)
        pair_board = np.array([[12, 3], [6, 0], [7, 1], [8, 2], [9, 3]], dtype=np.int8)

        trips_hi, _ = self.ev.get_hand_rank_52_plo8(hand_2d=trips_hand, board_2d=trips_board)
        pair_hi, _ = self.ev.get_hand_rank_52_plo8(hand_2d=pair_hand, board_2d=pair_board)
        self.assertGreater(trips_hi, pair_hi)

    def test_agrees_with_pokerkit_reference_evaluator(self):
        """
        Differential test against pokerkit's independent Omaha Hi/Lo evaluator
        (OmahaHoldemHand / OmahaEightOrBetterLowHand) across random deals: hi
        ordering, lo qualification, and lo ordering must all agree. Not a repo
        dependency -- skipped if pokerkit isn't installed in this environment.

        Note: pokerkit's Hand.__lt__ already inverts comparisons for low hands
        (a lower Entry.index is a *better* low hand), so `pkloA > pkloB` means
        "A is the better low hand" directly, matching this evaluator's own
        "higher rank int is better" convention with no extra inversion needed.
        """
        try:
            from pokerkit import Card, OmahaHoldemHand, OmahaEightOrBetterLowHand
        except ImportError:
            self.skipTest("pokerkit not installed; skipping cross-evaluator check")
            return

        def card_str(rank_idx, suit_idx):
            return _RANKS[rank_idx] + _SUITS[suit_idx]

        def parse(cards):
            return list(Card.parse("".join(card_str(r, s) for r, s in cards)))

        rng = random.Random(42)
        deck = [(r, s) for r in range(13) for s in range(4)]

        for _ in range(300):
            cards = rng.sample(deck, 13)
            hole_a, hole_b, board = cards[0:4], cards[4:8], cards[8:13]

            hi_a, lo_a = self.ev.get_hand_rank_52_plo8(
                hand_2d=np.array(hole_a, dtype=np.int8), board_2d=np.array(board, dtype=np.int8))
            hi_b, lo_b = self.ev.get_hand_rank_52_plo8(
                hand_2d=np.array(hole_b, dtype=np.int8), board_2d=np.array(board, dtype=np.int8))

            pk_board, pk_hole_a, pk_hole_b = parse(board), parse(hole_a), parse(hole_b)
            pk_hi_a = OmahaHoldemHand.from_game(pk_hole_a, pk_board)
            pk_hi_b = OmahaHoldemHand.from_game(pk_hole_b, pk_board)
            pk_lo_a = OmahaEightOrBetterLowHand.from_game_or_none(pk_hole_a, pk_board)
            pk_lo_b = OmahaEightOrBetterLowHand.from_game_or_none(pk_hole_b, pk_board)

            self.assertEqual((hi_a > hi_b) - (hi_a < hi_b), (pk_hi_a > pk_hi_b) - (pk_hi_a < pk_hi_b))
            self.assertEqual(lo_a is not None, pk_lo_a is not None)
            self.assertEqual(lo_b is not None, pk_lo_b is not None)

            if lo_a is not None and lo_b is not None:
                self.assertEqual((lo_a > lo_b) - (lo_a < lo_b), (pk_lo_a > pk_lo_b) - (pk_lo_a < pk_lo_b))


if __name__ == '__main__':
    unittest.main()
