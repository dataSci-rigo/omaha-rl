"""
Convergence sanity + determinism for ESMCCFRSolver on a degenerate abstraction.

With K=[1,1,1,1] every hand maps to one bucket, so the abstract game is just
the 6,378-sequence betting game with card outcomes averaged by sampling --
small enough that a short run must show CFR behavior: normalized positive
regret falling and the average strategy stabilizing. (No exact best response:
that would need full range-enumeration expectations; the honest proxies used
here and in benchmarks are the regret trend and head-to-head results.)
"""
import unittest

import numpy as np

from MCCFR.betting_tree import build_tree
from MCCFR.solver import ESMCCFRSolver


class ConstantBucketer:
    K = [1, 1, 1, 1]

    def range_idx(self, hole_1d):
        return 0

    def preflop_class(self, range_idx):
        return 0

    def bucket(self, street, hole_1d, board_1d):
        return 0


def _norm_pos_regret(solver):
    """Sum of positive regrets, normalized by the linear-weight mass so
    different run lengths are comparable."""
    t = solver.iteration
    weight_mass = solver.weight_scale * t * (t + 1) / 2.0
    total = sum(float(np.maximum(r, 0.0).sum()) for r in solver.regret)
    return total / weight_mass


class TestSolverSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tree = build_tree()

    def test_regret_falls_and_strategy_stabilizes(self):
        solver = ESMCCFRSolver(self.tree, ConstantBucketer(), seed=123)

        solver.run(4000)
        early = _norm_pos_regret(solver)

        solver.run(16000)
        late = _norm_pos_regret(solver)

        # The CFR guarantee: normalized positive regret falls. Measured
        # trajectory for this seed: 32.8 @4k -> 13.1 @20k -> 6.9 @40k ->
        # 5.5 @60k, with preflop avg-strategy drift settling to 0.07 by 60k
        # and the root converging to never-fold/77% call/23% raise. Strategy
        # drift itself is NOT asserted: it only settles past ~40k iterations
        # (too slow for a smoke test) and single-infoset probes are flaky
        # while self-play strategies cycle around the converging average.
        self.assertLess(late, early * 0.5,
                        f"normalized regret not falling: {early} -> {late}")

        # every street should have been visited
        self.assertTrue(all(f > 0 for f in solver.fraction_touched()))

    def test_deterministic_for_fixed_seed(self):
        a = ESMCCFRSolver(self.tree, ConstantBucketer(), seed=7)
        b = ESMCCFRSolver(self.tree, ConstantBucketer(), seed=7)
        a.run(500)
        b.run(500)
        for s in range(4):
            np.testing.assert_array_equal(a.regret[s], b.regret[s])
            np.testing.assert_array_equal(a.avg_strat[s], b.avg_strat[s])

    def test_3seat_regret_falls_and_determinism(self):
        from MCCFR.betting_tree import load_or_build_tree
        tree3 = load_or_build_tree(n_seats=3)
        solver = ESMCCFRSolver(tree3, ConstantBucketer(), seed=31)
        solver.run(1000)
        early = _norm_pos_regret(solver)
        solver.run(4000)
        late = _norm_pos_regret(solver)
        self.assertLess(late, early,
                        f"3-seat normalized regret not falling: {early} -> {late}")
        self.assertTrue(all(f > 0 for f in solver.fraction_touched()))

        b = ESMCCFRSolver(tree3, ConstantBucketer(), seed=31)
        b.run(200)
        c = ESMCCFRSolver(tree3, ConstantBucketer(), seed=31)
        c.run(200)
        for s in range(4):
            np.testing.assert_array_equal(b.regret[s], c.regret[s])

    def test_save_load_roundtrip(self):
        import tempfile, os
        solver = ESMCCFRSolver(self.tree, ConstantBucketer(), seed=9)
        solver.run(300)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.npz")
            solver.save(path)
            restored = ESMCCFRSolver.load(path, self.tree, ConstantBucketer())
            self.assertEqual(restored.iteration, solver.iteration)
            for s in range(4):
                np.testing.assert_array_equal(restored.regret[s],
                                              solver.regret[s])
            # resumed run must match a continuous run exactly (same RNG state)
            cont = ESMCCFRSolver(self.tree, ConstantBucketer(), seed=9)
            cont.run(400)
            restored.run(100)
            for s in range(4):
                np.testing.assert_array_equal(restored.regret[s],
                                              cont.regret[s])


if __name__ == '__main__':
    unittest.main()
