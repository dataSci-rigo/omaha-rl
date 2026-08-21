"""
3-seat lockstep differential: MCCFR.betting_tree vs FixedLimitOmahaHiLo(n=3).

Same load-bearing role as the HU lockstep test, plus the merged-infoset
consistency check: every pair of decision nodes sharing an infoset row must
agree on street, actor, and legal actions -- otherwise a merged strategy row
would be ill-defined.
"""
import unittest

import numpy as np

from MCCFR.betting_tree import (load_or_build_tree, DECISION, FOLD_TERMINAL,
                                SHOWDOWN_TERMINAL)
from MCCFR.showdown import showdown_payoff_multi
from PokerRL.game.games import FixedLimitOmahaHiLo

N_HANDS = 2000
START_STACK = 2000


class TestBettingTree3Seat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lut_holder = FixedLimitOmahaHiLo.get_lut_holder()
        args = FixedLimitOmahaHiLo.ARGS_CLS(
            n_seats=3,
            starting_stack_sizes_list=[START_STACK] * 3,
            stack_randomization_range=(0, 0))
        cls.env = FixedLimitOmahaHiLo(env_args=args, lut_holder=cls.lut_holder,
                                      is_evaluating=True)
        cls.tree = load_or_build_tree(n_seats=3)

    def test_counts_match_measured(self):
        self.assertEqual(self.tree.n_decisions_per_street,
                         [90, 4412, 205962, 9501342])
        self.assertEqual(self.tree.n_infosets_per_street,
                         [90, 944, 2368, 6056])
        self.assertEqual(self.tree.n_nodes, 23937308)

    def test_merged_infosets_are_consistent(self):
        tree = self.tree
        dec = tree.node_type == DECISION
        legal_packed = ((tree.children[:, 0] >= 0).astype(np.int8)
                        + 2 * (tree.children[:, 1] >= 0)
                        + 4 * (tree.children[:, 2] >= 0))
        for s in range(tree.n_streets):
            sel = dec & (tree.street == s)
            idx = tree.infoset_idx[sel]
            n = tree.n_infosets_per_street[s]
            self.assertEqual(int(idx.max()) + 1, n)
            for values in (tree.actor[sel], legal_packed[sel],
                           tree.folded_mask[sel]):
                lo = np.full(n, 127, dtype=np.int16)
                hi = np.full(n, -1, dtype=np.int16)
                np.minimum.at(lo, idx, values)
                np.maximum.at(hi, idx, values)
                np.testing.assert_array_equal(
                    lo, hi, err_msg=f"street {s}: merged infosets disagree")

    def test_lockstep_random_hands(self):
        env, tree = self.env, self.tree
        rng = np.random.default_rng(0)
        seen_fold = seen_showdown = seen_dead_money = 0

        for _ in range(N_HANDS):
            env.reset()
            node = tree.ROOT

            while True:
                self.assertEqual(tree.node_type[node], DECISION)
                self.assertEqual(int(tree.actor[node]),
                                 env.current_player.seat_id)
                self.assertEqual(int(tree.street[node]), env.current_round)
                env_legal = sorted(env.get_legal_actions())
                self.assertEqual(tree.legal_actions(node), env_legal)
                committed_env = [START_STACK - p.stack for p in env.seats]
                self.assertEqual(list(tree.committed[node]), committed_env)
                env_folded = sum(1 << p.seat_id for p in env.seats
                                 if p.folded_this_episode)
                self.assertEqual(int(tree.folded_mask[node]), env_folded)

                a = int(rng.choice(env_legal))
                _, rew, done, _ = env.step(a)
                node = int(tree.children[node, a])
                if done:
                    break

            ntype = tree.node_type[node]
            self.assertIn(ntype, (FOLD_TERMINAL, SHOWDOWN_TERMINAL))
            mask = int(tree.folded_mask[node])
            if ntype == FOLD_TERMINAL:
                seen_fold += 1
                payoff = tree.terminal_payoff(node)
            else:
                seen_showdown += 1
                if mask:
                    seen_dead_money += 1
                ranks = [None if (mask >> p) & 1 else
                         env.get_hand_rank(hand_2d=env.seats[p].hand,
                                           board_2d=env.board)
                         for p in range(3)]
                pot = int(tree.committed[node].sum())
                chips = showdown_payoff_multi(ranks, pot, 3)
                payoff = tree.terminal_payoff(node, showdown_chips=chips)

            env_payoff = np.array(rew, dtype=np.float64) * env.REWARD_SCALAR
            np.testing.assert_allclose(payoff, env_payoff, atol=1e-6,
                                       err_msg=f"terminal node {node}")

        self.assertGreater(seen_fold, 200)
        self.assertGreater(seen_showdown, 100)
        self.assertGreater(seen_dead_money, 30)  # folded chips in showdown pots


if __name__ == '__main__':
    unittest.main()
