"""
External-Sampling MCCFR over the static betting tree.

Per iteration t and per traverser seat: shuffle once, deal both hole hands and
the FULL 5-card board (revealed street by street as the tree walk descends) --
so all of the traverser's action branches share one chance sample, textbook ES.
(Deliberately unlike DeepCFR's MultiOutcomeSampler, which reshuffles the
remaining deck between branches.)

At traverser nodes we recurse on every legal action and accumulate weighted
regrets; at opponent nodes we sample one action from the current
regret-matched strategy and accumulate the weighted average strategy.

Weighting is Linear MCCFR (increments weighted by t): standard, theoretically
supported alongside sampling (Brown & Sandholm 2019), much faster than vanilla
in practice. To keep t*x increments in floating range over long runs, tables
and the weight scale are rescaled together periodically -- exactly
proportion-preserving, so regret matching and the normalized average strategy
are unaffected.

Infoset row: decision_idx[node] * K[street] + bucket. The node id carries the
full public betting history (perfect recall on betting); buckets are
imperfect-recall on cards (current street only) -- the standard abstraction.
"""
import json

import numpy as np

from MCCFR.betting_tree import (DECISION, FOLD_TERMINAL, SHOWDOWN_TERMINAL,
                                N_ACTIONS)
from MCCFR.showdown import showdown_payoff
from PokerRL.game._.cpp_wrappers.CppHandEvalHiLo import CppHandEvalHiLo
from MCCFR.abstraction.equity import _CARD_2D

_RESCALE_TRIGGER = 1e12
_RESCALE_FACTOR = 1e-6


def regret_matching(regret_row, legal_mask):
    """Positive-part-normalized strategy over legal actions (float64[3])."""
    pos = np.where(legal_mask, np.maximum(regret_row, 0.0), 0.0)
    s = pos.sum()
    if s > 0.0:
        return pos / s
    out = legal_mask.astype(np.float64)
    return out / out.sum()


class ESMCCFRSolver:

    def __init__(self, tree, bucketer, seed=0):
        self.tree = tree
        self.bucketer = bucketer
        self.K = list(bucketer.K)
        self.regret = []
        self.avg_strat = []
        for s in range(tree.n_streets):
            n_rows = tree.n_decisions_per_street[s] * self.K[s]
            self.regret.append(np.zeros((n_rows, N_ACTIONS), dtype=np.float64))
            self.avg_strat.append(np.zeros((n_rows, N_ACTIONS), dtype=np.float64))
        self.iteration = 0
        self.weight_scale = 1.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._evaluator = CppHandEvalHiLo()
        self._legal_masks = tree.children >= 0  # bool[N,3], precomputed

    # ------------------------------------------------------------------ run

    def run(self, n_iters, stop_flag=None, progress_cb=None,
            progress_every=1000):
        for _ in range(n_iters):
            if stop_flag is not None and stop_flag():
                break
            self.iteration += 1
            t = float(self.iteration)
            if t * self.weight_scale > _RESCALE_TRIGGER:
                self._rescale()
            w = t * self.weight_scale
            for traverser in (0, 1):
                self._run_one_traversal(traverser, w)
            if progress_cb is not None and self.iteration % progress_every == 0:
                progress_cb(self)

    def _rescale(self):
        for s in range(self.tree.n_streets):
            self.regret[s] *= _RESCALE_FACTOR
            self.avg_strat[s] *= _RESCALE_FACTOR
        self.weight_scale *= _RESCALE_FACTOR

    def _run_one_traversal(self, traverser, w):
        perm = self.rng.permutation(52).astype(np.int8)
        self._holes = (perm[0:4], perm[4:8])
        self._board5 = perm[8:13]
        # per-deal memos
        self._buckets = [[-1] * self.tree.n_streets for _ in range(2)]
        self._showdown_chips = None
        self._traverse(self.tree.ROOT, traverser, w)

    # ------------------------------------------------------------- internals

    def _bucket(self, seat, street):
        b = self._buckets[seat][street]
        if b < 0:
            hole = self._holes[seat]
            if street == 0:
                b = self.bucketer.preflop_class(self.bucketer.range_idx(hole))
            else:
                b = self.bucketer.bucket(street, hole,
                                         self._board5[:street + 2])
            self._buckets[seat][street] = b
        return b

    def _showdown(self):
        if self._showdown_chips is None:
            board_2d = _CARD_2D[self._board5]
            ranks = [self._evaluator.get_hand_rank_52_plo8(
                hand_2d=_CARD_2D[h], board_2d=board_2d) for h in self._holes]
            # pot differs per terminal; cache ranks via a closure-free memo:
            self._ranks = ranks
            self._showdown_chips = {}
        return self._ranks

    def _traverse(self, node, traverser, w):
        tree = self.tree
        ntype = tree.node_type[node]

        if ntype == FOLD_TERMINAL:
            f = tree.folder[node]
            c = tree.committed[node]
            return float(c[1 - traverser]) if f != traverser else -float(c[traverser])

        if ntype == SHOWDOWN_TERMINAL:
            ranks = self._showdown()
            pot = int(tree.committed[node, 0]) + int(tree.committed[node, 1])
            chips = self._showdown_chips.get(pot)
            if chips is None:
                chips = showdown_payoff(ranks[0], ranks[1], pot)
                self._showdown_chips[pot] = chips
            return chips[traverser] - float(tree.committed[node, traverser])

        # decision node
        actor = tree.actor[node]
        street = tree.street[node]
        row = tree.decision_idx[node] * self.K[street] + self._bucket(actor, street)
        legal_mask = self._legal_masks[node]
        table = self.regret[street]
        sigma = regret_matching(table[row], legal_mask)
        children = tree.children[node]

        if actor == traverser:
            v = 0.0
            v_a = np.zeros(N_ACTIONS, dtype=np.float64)
            for a in range(N_ACTIONS):
                if legal_mask[a]:
                    v_a[a] = self._traverse(children[a], traverser, w)
                    v += sigma[a] * v_a[a]
            table[row] += np.where(legal_mask, w * (v_a - v), 0.0)
            return v

        # opponent: accumulate average strategy, sample one action
        self.avg_strat[street][row] += w * sigma
        u = self.rng.random()
        acc = 0.0
        a = 0
        for i in range(N_ACTIONS):
            if legal_mask[i]:
                acc += sigma[i]
                a = i
                if u < acc:
                    break
        return self._traverse(children[a], traverser, w)

    # ------------------------------------------------------------ average

    def average_strategy(self):
        """Normalized average strategy per street: rows sum to 1 where visited,
        all-zero where never visited (caller decides the fallback)."""
        out = []
        for s in range(self.tree.n_streets):
            table = self.avg_strat[s]
            sums = table.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                norm = np.where(sums > 0.0, table / sums, 0.0)
            out.append(norm.astype(np.float32))
        return out

    def fraction_touched(self):
        return [float((self.avg_strat[s].sum(axis=1) > 0).mean())
                for s in range(self.tree.n_streets)]

    # ---------------------------------------------------------- persistence

    def save(self, path):
        import os
        tmp = str(path) + ".tmp"
        arrays = {}
        for s in range(self.tree.n_streets):
            arrays[f"regret_{s}"] = self.regret[s]
            arrays[f"avg_{s}"] = self.avg_strat[s]
        meta = json.dumps({
            "iteration": self.iteration,
            "weight_scale": self.weight_scale,
            "seed": self.seed,
            "K": self.K,
            "rng_state": _encode_rng_state(self.rng.bit_generator.state),
        })
        with open(tmp, "wb") as f:
            np.savez(f, meta=np.frombuffer(meta.encode(), dtype=np.uint8),
                     **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, tree, bucketer):
        with np.load(path) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            solver = cls(tree, bucketer, seed=meta["seed"])
            assert solver.K == meta["K"], \
                f"checkpoint K={meta['K']} != bucketer K={solver.K}"
            for s in range(tree.n_streets):
                solver.regret[s][:] = data[f"regret_{s}"]
                solver.avg_strat[s][:] = data[f"avg_{s}"]
        solver.iteration = meta["iteration"]
        solver.weight_scale = meta["weight_scale"]
        solver.rng.bit_generator.state = _decode_rng_state(meta["rng_state"])
        return solver


def _encode_rng_state(state):
    def enc(x):
        if isinstance(x, dict):
            return {k: enc(v) for k, v in x.items()}
        if isinstance(x, np.ndarray):
            return {"__nd__": x.tolist(), "dtype": str(x.dtype)}
        if isinstance(x, (np.integer,)):
            return int(x)
        return x
    return enc(state)


def _decode_rng_state(state):
    def dec(x):
        if isinstance(x, dict):
            if "__nd__" in x:
                return np.array(x["__nd__"], dtype=x["dtype"])
            return {k: dec(v) for k, v in x.items()}
        return x
    return dec(state)
