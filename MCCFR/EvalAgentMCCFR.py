"""
EvalAgent wrapper for a tabular MCCFR average strategy.

Plugs into everything EvalAgentDeepCFR plugs into (AgentTournament_hu,
examples/eval_agent_vs_bots.py, InteractiveGame): the tournament code drives
agents only through reset / get_action / notify_of_action.

The internal env wrapper is a VanillaWrapper (no action history), so the
agent tracks the betting sequence itself as a cursor into the static tree,
advanced on every action (its own and, via notify_of_action, the
opponent's). Every get_action cross-checks tree actor + legal actions
against the env -- cursor drift fails loudly, never silently.
"""
import numpy as np

from MCCFR.betting_tree import load_or_build_tree, DECISION
from PokerRL.rl.base_cls.EvalAgentBase import EvalAgentBase


class EvalAgentMCCFR(EvalAgentBase):
    EVAL_MODE_AVG = "AVG"
    ALL_MODES = [EVAL_MODE_AVG]

    def __init__(self, t_prof, mode=None, device=None):
        super().__init__(t_prof=t_prof,
                         mode=mode if mode is not None else self.EVAL_MODE_AVG,
                         device=device)
        # the wrapper is unusable (and unpicklable) before its first reset;
        # every tournament resets again per hand, re-syncing the cursor
        self._internal_env_wrapper.reset()
        self.tree = load_or_build_tree(n_seats=getattr(t_prof, "n_seats", 2))
        self.avg_strategy = None   # list of float32[rows, 3], rows sum to 1 or 0
        self.K = None
        self._node = self.tree.ROOT
        self._bucketer = None
        self._rng = np.random.default_rng()
        self.n_unvisited_fallbacks = 0

    # ------------------------------------------------------------ strategy

    def update_weights(self, weights_for_eval_agent):
        self.avg_strategy = weights_for_eval_agent["avg_strategy"]
        self.K = list(weights_for_eval_agent["K"])

    def can_compute_mode(self):
        return self.avg_strategy is not None

    def _get_bucketer(self):
        if self._bucketer is None:
            t = self.t_prof
            if getattr(t, "bucketer_kind", "m1") != "m1":
                raise NotImplementedError(t.bucketer_kind)
            from MCCFR.abstraction.m1_bins import M1Bucketer
            self._bucketer = M1Bucketer(
                k_postflop=t.k_postflop, n_rollouts=t.n_rollouts,
                n_opponents=getattr(t, "n_seats", 2) - 1)
            assert list(self._bucketer.K) == self.K, \
                f"strategy K={self.K} != bucketer K={self._bucketer.K}"
        return self._bucketer

    # ------------------------------------------------------------- cursor

    def reset(self, deck_state_dict=None):
        super().reset(deck_state_dict=deck_state_dict)
        self._node = self.tree.ROOT

    def notify_of_reset(self):
        super().notify_of_reset()
        self._node = self.tree.ROOT

    def notify_of_action(self, p_id_acted, action_he_did):
        super().notify_of_action(p_id_acted=p_id_acted,
                                 action_he_did=action_he_did)
        self._node = int(self.tree.children[self._node, action_he_did])

    # ------------------------------------------------------------- acting

    def _current_probs(self):
        env = self._internal_env_wrapper.env
        tree = self.tree
        node = self._node
        p_id = env.current_player.seat_id

        assert tree.node_type[node] == DECISION, f"cursor at terminal {node}"
        assert tree.actor[node] == p_id, \
            f"cursor actor {tree.actor[node]} != env actor {p_id}"
        env_legal = sorted(env.get_legal_actions())
        assert tree.legal_actions(node) == env_legal, \
            f"cursor legal {tree.legal_actions(node)} != env legal {env_legal}"

        street = int(tree.street[node])
        assert street == env.current_round

        hand_2d = env.seats[p_id].hand
        hole_1d = (hand_2d[:, 0].astype(np.int64) * 4
                   + hand_2d[:, 1].astype(np.int64))
        bucketer = self._get_bucketer()
        if street == 0:
            bucket = bucketer.preflop_class(bucketer.range_idx(hole_1d))
        else:
            board_2d = env.board
            dealt = board_2d[:, 0] != np.int8(-127)  # CARD_NOT_DEALT_TOKEN
            board_1d = (board_2d[dealt, 0].astype(np.int64) * 4
                        + board_2d[dealt, 1].astype(np.int64))
            bucket = bucketer.bucket(street, hole_1d, board_1d)

        row = int(tree.infoset_idx[node]) * self.K[street] + bucket
        probs = self.avg_strategy[street][row].astype(np.float64)
        mask = tree.legal_mask(node)
        probs = np.where(mask, probs, 0.0)
        s = probs.sum()
        if s <= 0.0:
            self.n_unvisited_fallbacks += 1
            probs = mask.astype(np.float64)
            s = probs.sum()
        return probs / s

    def get_a_probs(self):
        return self._current_probs()

    def get_action(self, step_env=True, need_probs=False):
        probs = self._current_probs()
        action = int(self._rng.choice(3, p=probs))
        if step_env:
            self._internal_env_wrapper.step(action=action)
            self._node = int(self.tree.children[self._node, action])
        return action, None

    def get_a_probs_for_each_hand(self):
        raise NotImplementedError("tabular agent: per-hand probs unsupported")

    def get_action_frac_tuple(self, step_env=True):
        raise NotImplementedError("fixed-limit game: no raise fractions")

    # -------------------------------------------------------- persistence

    def _state_dict(self):
        return {"avg_strategy": self.avg_strategy, "K": self.K}

    def _load_state_dict(self, state):
        self.avg_strategy = state["avg_strategy"]
        self.K = list(state["K"]) if state["K"] is not None else None
