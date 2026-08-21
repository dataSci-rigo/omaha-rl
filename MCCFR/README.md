# Tabular MCCFR for Heads-Up Fixed-Limit Omaha Hi/Lo (8-or-better)

*A one-page summary of the theory, the model, and the training results.*

## Theory

Counterfactual Regret Minimization (CFR) solves two-player zero-sum games by
self-play: at every information set it tracks how much better each action
would have done than the strategy actually played (regret), and plays in
proportion to positive regret. The *average* strategy over iterations
converges to a Nash equilibrium. **Monte Carlo CFR (external sampling)**
makes iterations cheap: sample one deal and one opponent line, but branch on
every action of the player being updated — unbiased regret estimates at a
tiny fraction of a full tree walk. We add **linear weighting** (iteration *t*
weighted by *t*), which discounts early garbage iterations and speeds
convergence in practice.

The full game is far too large to tabulate (270,725 four-card starting hands
× boards × betting sequences), so we solve an **abstraction**: betting is
kept exact (perfect recall — every public action sequence is distinct), and
hands are grouped into buckets per street. Preflop bucketing is *lossless*:
suit isomorphism collapses the 270,725 hands into 16,432 strategically
distinct classes (a wrong-but-tempting alternative, ignoring suits entirely,
would conflate double-suited and rainbow hands — fatal in a game where the
high half runs on flush potential). Postflop buckets are Monte Carlo
**expected-pot-share** bins — cheap and coarse, but bucket noise only affects
which strategy row a hand reads; the training signal itself (showdown
payoffs) is always exact.

## Model

The betting tree of HU fixed-limit O8 (blinds 1/2, bets 2/2/4/4, cap 4
raises/street) is tiny and fully enumerable: **16,584 nodes** (6,378 decision
nodes; max commitment 48 chips, so at deep stacks no all-in is ever
reached). We precompute it once and never touch the RL environment in the
training loop — deals are numpy permutations, showdowns go straight to the
C++ hi/lo evaluator, and the hi/lo split-pot arithmetic (scoops, quarters,
odd chips) is integer-exact. Correctness is not assumed: a lockstep
differential test plays thousands of hands through the real engine and this
tree in parallel and asserts identical actors, legal actions, chip counts,
and payouts at every node.

An information set is *(betting-tree node, hand bucket)*. Milestone-1 sizing:
16,432 preflop classes + 50 equity bins per postflop street =
**449,956 infosets ≈ 22 MB** of regret + average-strategy tables. The whole
strategy is inspectable — dump any spot's action frequencies by hand class.

## Training and results

One CPU core, ~370 MB RAM, ~28–32 iterations/s (two ES traversals each).
After **~6 hours (572k iterations)** more than 90% of infosets on every
street had been visited, and the average strategy cleared every benchmark we
have — each match 20,000 hands, 95% CI:

| Opponent | Result (milli-BB/hand) | Verdict |
|---|---|---|
| ABCBot (rule-based equity bot) | **+156** [+99, +214] | winning |
| BayesianBot (opponent-modeling bot) | **+218** [+172, +264] | winning |
| Deep CFR agent (same engine, one night × 4 workers) | **+195** [+148, +242] | winning |

The Deep CFR baseline is the interesting row: trained on the identical
engine with ~5× the compute, it had only reached statistical *parity* with
the two rule bots. At laptop-scale budgets, dense tabular visits to a small,
well-chosen abstraction beat a data-starved neural approximator — while
converging with guarantees and staying fully debuggable.

**Next:** Milestone 2 replaces the postflop equity bins with k-means buckets
over outcome-distribution features (high equity, low equity, scoop
probability, quarter risk) — the "scoopiness" geometry that matters in
split-pot games — en route to a Pluribus-style blueprint-plus-search design
for multiway pot-limit O8.
