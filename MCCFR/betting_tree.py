"""
Static betting tree for N-seat fixed-limit games (heads-up and 3-handed).

The tree enumerates every betting sequence of a fixed-limit game once, up
front, so the MCCFR solver never touches the (slow) RL env in its hot loop.
Correctness is not argued here -- test/mccfr/test_betting_tree.py steps this
tree and FixedLimitOmahaHiLo in lockstep over thousands of random hands and
asserts identical actors, legal actions, commitments and terminal payoffs.

Node ids are stable for a given rule set (deterministic recursive build), so a
decision node id doubles as the perfect-recall betting component of an infoset
key: in a limit game all betting is public, and every node is a unique full
action history (prior-street history included, since street subtrees are
replicated per path).

Rules mirrored from the env (all proven by the differential test):
- actions: FOLD=0, CHECK_CALL=1, BET_RAISE=2 (Poker constants)
- FOLD is illegal when there is nothing to call; BET_RAISE is illegal once
  n_raises_this_round reaches the cap
- seats (PokerEnv): HU -> 0=BTN/SB, 1=BB, seat 0 first preflop, seat 1 first
  postflop; 3-handed -> 0=BTN, 1=SB, 2=BB, BTN first preflop, then postflop
  the smallest non-folded seat with the BTN treated as last (order 1,2,0)
- next to act: next non-folded seat id, cyclically
- preflop the raise counter starts at 1 (the BB counts as a raise), postflop
  at 0; a raise sets the actor's per-round bet to (n_raises + 1) * bet_size
- bet_size is small_bet before ``big_bet_round`` and big_bet from it on
- a round ends when every non-folded player has acted and matched the
  largest bet; a hand ends when one player remains or the river closes
- stacks are assumed deep enough that no all-in is ever reached (with the
  default rules max commitment is 8+8+16+16 = 48 chips per seat)
"""

import numpy as np

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
DECISION, FOLD_TERMINAL, SHOWDOWN_TERMINAL = 0, 1, 2
N_ACTIONS = 3


class BettingTree:
    """Struct-of-arrays fixed-limit betting tree. Built by build_tree()."""

    def __init__(self, node_type, street, actor, children, committed,
                 decision_idx, folder, folded_mask, n_decisions_per_street,
                 n_streets, n_seats, infoset_idx, n_infosets_per_street):
        self.node_type = node_type            # int8[N]
        self.street = street                  # int8[N]
        self.actor = actor                    # int8[N], -1 for terminals
        self.children = children              # int32[N,3], -1 = illegal
        self.committed = committed            # int16[N,n_seats], per seat
        self.decision_idx = decision_idx      # int32[N], dense per-street, -1 for terminals
        self.folder = folder                  # int8[N], seat whose fold made a FOLD_TERMINAL
        self.folded_mask = folded_mask        # uint8[N], bit p set = seat p folded
        self.n_decisions_per_street = n_decisions_per_street
        self.n_streets = n_streets
        self.n_seats = n_seats
        self.n_nodes = len(node_type)
        self.ROOT = 0
        # Betting component of the infoset key. Heads-up this is just the
        # decision node id (perfect recall -- the tree is tiny). For 3+ seats
        # perfect recall explodes (9.5M river decision nodes), so nodes are
        # merged by PUBLIC CHIP STATE: (street, entry active-set, entry chip
        # state, actions so far THIS street). Payoffs stay exact per node;
        # only strategy/regret rows are shared. What is forgotten is the
        # ordering of prior streets' actions -- their chip outcome and the
        # surviving players are remembered.
        self.infoset_idx = infoset_idx            # int32[N], -1 for terminals
        self.n_infosets_per_street = n_infosets_per_street

    def legal_actions(self, node_id):
        return [a for a in range(N_ACTIONS) if self.children[node_id, a] >= 0]

    def legal_mask(self, node_id):
        return self.children[node_id] >= 0

    def active_seats(self, node_id):
        m = int(self.folded_mask[node_id])
        return [p for p in range(self.n_seats) if not (m >> p) & 1]

    def terminal_payoff(self, node_id, showdown_chips=None):
        """Chip EV per seat at a terminal, relative to committed chips.

        For showdown terminals pass showdown_chips: per-seat chips awarded
        from the pot (folded seats 0), e.g. from MCCFR.showdown. Returns
        np.float64[n_seats].
        """
        c = self.committed[node_id].astype(np.float64)
        if self.node_type[node_id] == FOLD_TERMINAL:
            active = self.active_seats(node_id)
            assert len(active) == 1
            w = active[0]
            out = -c
            out[w] = c.sum() - c[w]
            return out
        return np.asarray(showdown_chips, dtype=np.float64) - c


def build_tree(n_seats=2, small_blind=1, big_blind=2, small_bet=2, big_bet=4,
               n_streets=4, big_bet_round=2, max_raises_per_round=4,
               preflop_n_raises_start=1):
    """Build the full fixed-limit tree. Defaults match FixedLimitOmahaHiLo HU;
    n_seats=3 matches the env's 3-handed rules (BTN=0, SB=1, BB=2)."""
    assert n_seats in (2, 3), "positions verified for 2 and 3 seats only"
    perfect_recall = n_seats == 2
    node_type, street_l, actor_l, children_l, committed_l = [], [], [], [], []
    decision_idx_l, folder_l, folded_l, infoset_l = [], [], [], []
    n_decisions_per_street = [0] * n_streets
    infoset_keys = [{} for _ in range(n_streets)]  # merged-key -> dense idx

    def new_node(ntype, st, actor, committed, folded, folder=-1,
                 infoset_key=None):
        nid = len(node_type)
        node_type.append(ntype)
        street_l.append(st)
        actor_l.append(actor)
        children_l.append([-1, -1, -1])
        committed_l.append(list(committed))
        if ntype == DECISION:
            decision_idx_l.append(n_decisions_per_street[st])
            n_decisions_per_street[st] += 1
            if perfect_recall:
                infoset_l.append(decision_idx_l[-1])
            else:
                idx = infoset_keys[st].setdefault(infoset_key,
                                                  len(infoset_keys[st]))
                infoset_l.append(idx)
        else:
            decision_idx_l.append(-1)
            infoset_l.append(-1)
        folder_l.append(folder)
        folded_l.append(sum(1 << p for p in range(n_seats) if folded[p]))
        return nid

    def bet_size(st):
        return big_bet if st >= big_bet_round else small_bet

    def next_actor(seat, folded):
        for i in range(1, n_seats + 1):
            p = (seat + i) % n_seats
            if not folded[p]:
                return p
        raise AssertionError

    def first_postflop(folded):
        # smallest non-folded seat id, BTN (seat 0) treated as last
        for p in list(range(1, n_seats)) + [0]:
            if not folded[p]:
                return p
        raise AssertionError

    def round_over(tot, round_start, acted, folded):
        bets = [tot[p] - round_start[p] for p in range(n_seats)]
        largest = max(bets)
        return all(folded[p] or (acted[p] and bets[p] == largest)
                   for p in range(n_seats))

    def advance(st, tot, folded):
        """Round complete with >1 active: next street or showdown."""
        if st == n_streets - 1:
            return new_node(SHOWDOWN_TERMINAL, st, -1, tot, folded)
        return build(st + 1, first_postflop(folded), tot, list(tot), 0,
                     [False] * n_seats, folded, tuple(folded), ())

    def build(st, to_act, tot, round_start, n_raises, acted, folded,
              entry_folded, seq):
        """Returns node id of the subtree root.

        tot: total committed per seat; round_start: committed per seat when
        this street began (current round bet = tot - round_start);
        acted: per-seat has-acted-this-round; folded: per-seat folded;
        entry_folded: folded set when this street began; seq: action tuple
        taken this street so far (the merged-infoset key components).
        """
        # merged public-chip-state key: what a player can see, minus the
        # ordering of PRIOR streets' actions. v = common entry commitment of
        # then-active seats (equal by the FL matched-bet invariant; 0
        # preflop where blinds are current-round bets), dead = chips left
        # behind by seats that folded on earlier streets.
        v = max((round_start[p] for p in range(n_seats) if not entry_folded[p]),
                default=0)
        dead = sum(round_start[p] for p in range(n_seats) if entry_folded[p])
        nid = new_node(DECISION, st, to_act, tot, folded,
                       infoset_key=(entry_folded, v, dead, seq))
        bets = [tot[p] - round_start[p] for p in range(n_seats)]
        to_call = max(bets) - bets[to_act]

        # FOLD
        if to_call > 0:
            folded_f = list(folded)
            folded_f[to_act] = True
            if sum(not f for f in folded_f) == 1:
                children_l[nid][FOLD] = new_node(FOLD_TERMINAL, st, -1, tot,
                                                 folded_f, folder=to_act)
            elif round_over(tot, round_start, acted, folded_f):
                children_l[nid][FOLD] = advance(st, tot, folded_f)
            else:
                children_l[nid][FOLD] = build(
                    st, next_actor(to_act, folded_f), tot, round_start,
                    n_raises, acted, folded_f, entry_folded, seq + (FOLD,))

        # CHECK_CALL
        tot_cc = list(tot)
        tot_cc[to_act] = round_start[to_act] + max(bets)
        acted_cc = list(acted)
        acted_cc[to_act] = True
        if round_over(tot_cc, round_start, acted_cc, folded):
            children_l[nid][CHECK_CALL] = advance(st, tot_cc, folded)
        else:
            children_l[nid][CHECK_CALL] = build(
                st, next_actor(to_act, folded), tot_cc, round_start,
                n_raises, acted_cc, folded, entry_folded, seq + (CHECK_CALL,))

        # BET_RAISE
        if n_raises < max_raises_per_round:
            tot_br = list(tot)
            tot_br[to_act] = round_start[to_act] + (n_raises + 1) * bet_size(st)
            acted_br = list(acted)
            acted_br[to_act] = True
            children_l[nid][BET_RAISE] = build(
                st, next_actor(to_act, folded), tot_br, round_start,
                n_raises + 1, acted_br, folded, entry_folded,
                seq + (BET_RAISE,))

        return nid

    tot0 = [0] * n_seats
    if n_seats == 2:
        tot0 = [small_blind, big_blind]      # 0=BTN/SB, 1=BB
        first = 0
    else:
        tot0[1], tot0[2] = small_blind, big_blind  # 0=BTN, 1=SB, 2=BB
        first = 0                             # BTN opens 3-handed preflop

    root = build(st=0, to_act=first, tot=tot0, round_start=[0] * n_seats,
                 n_raises=preflop_n_raises_start, acted=[False] * n_seats,
                 folded=[False] * n_seats, entry_folded=(False,) * n_seats,
                 seq=())
    assert root == 0

    if perfect_recall:
        n_infosets = list(n_decisions_per_street)
    else:
        n_infosets = [len(k) for k in infoset_keys]

    return BettingTree(
        node_type=np.array(node_type, dtype=np.int8),
        street=np.array(street_l, dtype=np.int8),
        actor=np.array(actor_l, dtype=np.int8),
        children=np.array(children_l, dtype=np.int32),
        committed=np.array(committed_l, dtype=np.int16),
        decision_idx=np.array(decision_idx_l, dtype=np.int32),
        folder=np.array(folder_l, dtype=np.int8),
        folded_mask=np.array(folded_l, dtype=np.uint8),
        n_decisions_per_street=n_decisions_per_street,
        n_streets=n_streets,
        n_seats=n_seats,
        infoset_idx=np.array(infoset_l, dtype=np.int32),
        n_infosets_per_street=n_infosets,
    )


def load_or_build_tree(n_seats=2, cache_dir=None):
    """Build the tree, caching the arrays on disk (the 3-seat build takes
    ~2 min and ~24M nodes; loading the cache takes seconds). HU builds in
    <100 ms and is never cached."""
    import os
    if n_seats == 2:
        return build_tree(n_seats=2)
    cache_dir = os.path.expanduser(cache_dir or "~/poker_ai_data/tree_cache")
    path = os.path.join(cache_dir, f"fl_tree_{n_seats}seat_v1.npz")
    if os.path.exists(path):
        with np.load(path) as d:
            return BettingTree(
                node_type=d["node_type"], street=d["street"],
                actor=d["actor"], children=d["children"],
                committed=d["committed"], decision_idx=d["decision_idx"],
                folder=d["folder"], folded_mask=d["folded_mask"],
                n_decisions_per_street=d["n_decisions_per_street"].tolist(),
                n_streets=int(d["n_streets"]), n_seats=int(d["n_seats"]),
                infoset_idx=d["infoset_idx"],
                n_infosets_per_street=d["n_infosets_per_street"].tolist(),
            )
    tree = build_tree(n_seats=n_seats)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, node_type=tree.node_type, street=tree.street,
                 actor=tree.actor, children=tree.children,
                 committed=tree.committed, decision_idx=tree.decision_idx,
                 folder=tree.folder, folded_mask=tree.folded_mask,
                 n_decisions_per_street=np.array(tree.n_decisions_per_street),
                 n_streets=tree.n_streets, n_seats=tree.n_seats,
                 infoset_idx=tree.infoset_idx,
                 n_infosets_per_street=np.array(tree.n_infosets_per_street))
    os.replace(tmp, path)
    return tree
