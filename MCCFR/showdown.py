"""
HU Omaha Hi/Lo showdown payout, integer-exact.

Mirrors PokerEnv._payout_pots_hi_lo for the heads-up, no-side-pot case (the
only case the MCCFR tree can reach: max commitment 48 << stacks). Differential
tested against the env in test/mccfr/test_showdown_payoff.py.

Rules (proven, not assumed -- see the env docstrings and tests):
- hand rank is a (hi_rank, lo_rank_or_None) pair, higher is better on both
  scales; lo_rank is None when the hand has no qualifying 8-or-better low
- no qualifying low anywhere -> hi winner(s) scoop the whole pot
- otherwise lo_half = pot // 2 and the HIGH side gets the odd chip
- ties within a half split it; remainder chips go to the earliest seat left
  of the button -- in HU with BTN = seat 0 that means seat 1 first
"""

_HU_BUTTON_ORDER = (1, 0)  # earliest seat left of the button first (BTN = seat 0)


def _split_half(amount, winners):
    """Split `amount` among winner seat ids; remainder chips by button order."""
    base, remainder = divmod(int(amount), len(winners))
    out = {p: base for p in winners}
    for p in [p for p in _HU_BUTTON_ORDER if p in out][:remainder]:
        out[p] += 1
    return out


def showdown_payoff(rank0, rank1, pot):
    """Chips awarded to each seat from `pot` at showdown.

    rank0, rank1: (hi_rank, lo_rank_or_None) for seats 0 and 1.
    Returns (chips0, chips1) with chips0 + chips1 == pot.
    """
    ranks = (rank0, rank1)
    hi_best = max(r[0] for r in ranks)
    hi_winners = [p for p in (0, 1) if ranks[p][0] == hi_best]

    lo_qualified = [p for p in (0, 1) if ranks[p][1] is not None]
    if lo_qualified:
        lo_best = max(ranks[p][1] for p in lo_qualified)
        lo_winners = [p for p in lo_qualified if ranks[p][1] == lo_best]
    else:
        lo_winners = []

    if not lo_winners:
        payouts = _split_half(pot, hi_winners)
    else:
        lo_half = int(pot) // 2
        hi_half = int(pot) - lo_half
        payouts = _split_half(hi_half, hi_winners)
        for p, amt in _split_half(lo_half, lo_winners).items():
            payouts[p] = payouts.get(p, 0) + amt

    return payouts.get(0, 0), payouts.get(1, 0)
