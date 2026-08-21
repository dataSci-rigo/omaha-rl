"""
Omaha Hi/Lo showdown payout, integer-exact, for 2 or more seats.

Mirrors PokerEnv._payout_pots_hi_lo for the no-side-pot case (the only case
the MCCFR tree can reach: max commitment 48 << stacks). Folded players'
chips stay in the pot as dead money; only active seats are ranked.
Differential tested against the env in test/mccfr/.

Rules (proven, not assumed -- see the env docstrings and tests):
- hand rank is a (hi_rank, lo_rank_or_None) pair, higher is better on both
  scales; lo_rank is None when the hand has no qualifying 8-or-better low
- no qualifying low anywhere -> hi winner(s) scoop the whole pot
- otherwise lo_half = pot // 2 and the HIGH side gets the odd chip
- ties within a half split it; remainder chips go to the earliest seat left
  of the button (BTN = seat 0), i.e. seat order 1, 2, ..., n-1, 0
"""


def _button_order(seats, n_seats):
    return sorted(seats, key=lambda p: (p - 1) % n_seats)


def _split_half(amount, winners, n_seats):
    """Split `amount` among winner seat ids; remainder chips by button order."""
    base, remainder = divmod(int(amount), len(winners))
    out = {p: base for p in winners}
    for p in _button_order(winners, n_seats)[:remainder]:
        out[p] += 1
    return out


def showdown_payoff_multi(ranks, pot, n_seats):
    """Chips awarded to each seat from `pot` at showdown.

    ranks: per-seat (hi_rank, lo_rank_or_None), or None for folded seats.
    Returns a tuple of n_seats chip counts summing to pot.
    """
    active = [p for p in range(n_seats) if ranks[p] is not None]
    assert len(active) >= 2

    hi_best = max(ranks[p][0] for p in active)
    hi_winners = [p for p in active if ranks[p][0] == hi_best]

    lo_qualified = [p for p in active if ranks[p][1] is not None]
    if lo_qualified:
        lo_best = max(ranks[p][1] for p in lo_qualified)
        lo_winners = [p for p in lo_qualified if ranks[p][1] == lo_best]
    else:
        lo_winners = []

    if not lo_winners:
        payouts = _split_half(pot, hi_winners, n_seats)
    else:
        lo_half = int(pot) // 2
        hi_half = int(pot) - lo_half
        payouts = _split_half(hi_half, hi_winners, n_seats)
        for p, amt in _split_half(lo_half, lo_winners, n_seats).items():
            payouts[p] = payouts.get(p, 0) + amt

    return tuple(payouts.get(p, 0) for p in range(n_seats))


def showdown_payoff(rank0, rank1, pot):
    """Heads-up convenience wrapper: returns (chips0, chips1)."""
    return showdown_payoff_multi((rank0, rank1), pot, n_seats=2)
