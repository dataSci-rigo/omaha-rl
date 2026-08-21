"""
Static betting tree for heads-up fixed-limit games.

The tree enumerates every betting sequence of a HU fixed-limit game once, up
front, so the MCCFR solver never touches the (slow) RL env in its hot loop.
Correctness is not argued here -- test/mccfr/test_betting_tree.py steps this
tree and FixedLimitOmahaHiLo in lockstep over thousands of random hands and
asserts identical actors, legal actions, commitments and terminal payoffs.

Node ids are stable for a given rule set (deterministic recursive build), so a
decision node id doubles as the perfect-recall betting component of an infoset
key: in a HU limit game all betting is public, and every node is a unique full
action history (prior-street history included, since street subtrees are
replicated per path).

Rules mirrored from the env (all proven by the differential test):
- actions: FOLD=0, CHECK_CALL=1, BET_RAISE=2 (Poker constants)
- FOLD is illegal when there is nothing to call; BET_RAISE is illegal once
  n_raises_this_round reaches the cap
- preflop the raise counter starts at 1 (the BB counts as a raise), postflop
  at 0; a raise sets the actor's per-round bet to (n_raises + 1) * bet_size
- bet_size is small_bet before ``big_bet_round`` and big_bet from it on
- a round ends when both players have acted and their bets are equal; the
  next street's first actor is seat 1 (BB) -- preflop it is seat 0 (SB/BTN)
- stacks are assumed deep enough that no all-in is ever reached (with the
  default rules max commitment is 8+8+16+16 = 48 chips per seat)
"""

import numpy as np

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
DECISION, FOLD_TERMINAL, SHOWDOWN_TERMINAL = 0, 1, 2
N_ACTIONS = 3


class BettingTree:
    """Struct-of-arrays HU fixed-limit betting tree. Built by build_tree()."""

    def __init__(self, node_type, street, actor, children, committed,
                 decision_idx, folder, n_decisions_per_street, n_streets):
        self.node_type = node_type            # int8[N]
        self.street = street                  # int8[N]
        self.actor = actor                    # int8[N], -1 for terminals
        self.children = children              # int32[N,3], -1 = illegal
        self.committed = committed            # int16[N,2], per seat at node
        self.decision_idx = decision_idx      # int32[N], dense per-street, -1 for terminals
        self.folder = folder                  # int8[N], -1 unless fold terminal
        self.n_decisions_per_street = n_decisions_per_street
        self.n_streets = n_streets
        self.n_nodes = len(node_type)
        self.ROOT = 0

    def legal_actions(self, node_id):
        return [a for a in range(N_ACTIONS) if self.children[node_id, a] >= 0]

    def legal_mask(self, node_id):
        return self.children[node_id] >= 0

    def terminal_payoff(self, node_id, showdown_chips=None):
        """Chip EV per seat at a terminal, relative to committed chips.

        For showdown terminals pass showdown_chips=(chips0, chips1) from
        MCCFR.showdown.showdown_payoff for the full pot at this node.
        Returns np.float64[2].
        """
        c = self.committed[node_id]
        if self.node_type[node_id] == FOLD_TERMINAL:
            f = self.folder[node_id]
            out = np.empty(2, dtype=np.float64)
            out[f] = -float(c[f])
            out[1 - f] = float(c[f])
            return out
        return np.array([showdown_chips[0] - c[0], showdown_chips[1] - c[1]],
                        dtype=np.float64)


def build_tree(small_blind=1, big_blind=2, small_bet=2, big_bet=4,
               n_streets=4, big_bet_round=2, max_raises_per_round=4,
               preflop_n_raises_start=1):
    """Build the full HU FL tree. Defaults match FixedLimitOmahaHiLo."""
    node_type, street_l, actor_l, children_l, committed_l = [], [], [], [], []
    decision_idx_l, folder_l = [], []
    n_decisions_per_street = [0] * n_streets

    def new_node(ntype, st, actor, committed, folder=-1):
        nid = len(node_type)
        node_type.append(ntype)
        street_l.append(st)
        actor_l.append(actor)
        children_l.append([-1, -1, -1])
        committed_l.append(list(committed))
        if ntype == DECISION:
            decision_idx_l.append(n_decisions_per_street[st])
            n_decisions_per_street[st] += 1
        else:
            decision_idx_l.append(-1)
        folder_l.append(folder)
        return nid

    def bet_size(st):
        return big_bet if st >= big_bet_round else small_bet

    def build(st, to_act, tot, round_start, n_raises, acted):
        """Returns node id of the subtree root.

        tot: total committed per seat; round_start: committed per seat when
        this street began (so current round bet = tot - round_start);
        acted: per-seat has-acted-this-round flags.
        """
        nid = new_node(DECISION, st, to_act, tot)
        other = 1 - to_act
        to_call = tot[other] - tot[to_act]

        # FOLD
        if to_call > 0:
            children_l[nid][FOLD] = new_node(FOLD_TERMINAL, st, -1, tot, folder=to_act)

        # CHECK_CALL
        tot_cc = list(tot)
        tot_cc[to_act] = tot[other]
        acted_cc = list(acted)
        acted_cc[to_act] = True
        if all(acted_cc):
            # round over
            if st == n_streets - 1:
                children_l[nid][CHECK_CALL] = new_node(SHOWDOWN_TERMINAL, st, -1, tot_cc)
            else:
                children_l[nid][CHECK_CALL] = build(
                    st + 1, 1, tot_cc, list(tot_cc), 0, [False, False])
        else:
            children_l[nid][CHECK_CALL] = build(
                st, other, tot_cc, round_start, n_raises, acted_cc)

        # BET_RAISE
        if n_raises < max_raises_per_round:
            tot_br = list(tot)
            tot_br[to_act] = round_start[to_act] + (n_raises + 1) * bet_size(st)
            acted_br = list(acted)
            acted_br[to_act] = True
            children_l[nid][BET_RAISE] = build(
                st, other, tot_br, round_start, n_raises + 1, acted_br)

        return nid

    root = build(st=0, to_act=0, tot=[small_blind, big_blind],
                 round_start=[0, 0], n_raises=preflop_n_raises_start,
                 acted=[False, False])
    assert root == 0

    return BettingTree(
        node_type=np.array(node_type, dtype=np.int8),
        street=np.array(street_l, dtype=np.int8),
        actor=np.array(actor_l, dtype=np.int8),
        children=np.array(children_l, dtype=np.int32),
        committed=np.array(committed_l, dtype=np.int16),
        decision_idx=np.array(decision_idx_l, dtype=np.int32),
        folder=np.array(folder_l, dtype=np.int8),
        n_decisions_per_street=n_decisions_per_street,
        n_streets=n_streets,
    )
