"""
Lockstep differential test: MCCFR.betting_tree vs FixedLimitOmahaHiLo.

This is the load-bearing test for the whole tabular MCCFR effort. It plays
thousands of seeded random hands, stepping the real env and a tree cursor in
parallel, and asserts at every node that the tree agrees with the env on
actor, legal actions, chips committed, street, and terminal payoffs (fold and
showdown, including hi/lo splits, quarters and scoops). No betting or payout
convention is assumed anywhere in MCCFR/ -- it is all proven here.
"""
import unittest

import numpy as np

from MCCFR.betting_tree import (build_tree, DECISION, FOLD_TERMINAL,
                                SHOWDOWN_TERMINAL)
from MCCFR.showdown import showdown_payoff
from PokerRL.game.games import FixedLimitOmahaHiLo

N_HANDS = 3000
START_STACK = 2000


class TestBettingTreeLockstep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lut_holder = FixedLimitOmahaHiLo.get_lut_holder()
        args = FixedLimitOmahaHiLo.ARGS_CLS(
            n_seats=2,
            starting_stack_sizes_list=[START_STACK] * 2,
            stack_randomization_range=(0, 0))
        cls.env = FixedLimitOmahaHiLo(env_args=args, lut_holder=cls.lut_holder,
                                      is_evaluating=True)
        cls.tree = build_tree()

    def test_node_counts(self):
        # 8 preflop decision nodes; 10 per postflop street subtree, with
        # 7/63/567 continuation prefixes reaching flop/turn/river.
        self.assertEqual(self.tree.n_decisions_per_street, [8, 70, 630, 5670])
        self.assertEqual(int((self.tree.node_type == DECISION).sum()), 6378)
        self.assertEqual(int((self.tree.node_type == FOLD_TERMINAL).sum()), 5103)
        self.assertEqual(int((self.tree.node_type == SHOWDOWN_TERMINAL).sum()), 5103)

    def test_lockstep_random_hands(self):
        env, tree = self.env, self.tree
        rng = np.random.default_rng(0)
        seen_fold = seen_showdown = 0

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

                a = int(rng.choice(env_legal))
                _, rew, done, _ = env.step(a)
                node = int(tree.children[node, a])

                if done:
                    break

            ntype = tree.node_type[node]
            self.assertIn(ntype, (FOLD_TERMINAL, SHOWDOWN_TERMINAL))
            if ntype == FOLD_TERMINAL:
                seen_fold += 1
                payoff = tree.terminal_payoff(node)
            else:
                seen_showdown += 1
                board = env.board
                ranks = [env.get_hand_rank(hand_2d=p.hand, board_2d=board)
                         for p in env.seats]
                pot = int(tree.committed[node].sum())
                chips = showdown_payoff(ranks[0], ranks[1], pot)
                payoff = tree.terminal_payoff(node, showdown_chips=chips)

            env_payoff = np.array(rew, dtype=np.float64) * env.REWARD_SCALAR
            np.testing.assert_allclose(payoff, env_payoff, atol=1e-6,
                                       err_msg=f"terminal node {node}")

        # make sure the run actually exercised both terminal kinds
        self.assertGreater(seen_fold, 100)
        self.assertGreater(seen_showdown, 100)


if __name__ == '__main__':
    unittest.main()
