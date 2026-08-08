"""
Dry run: BayesianBot vs. ABCBot (both from ~/Documents/omaha) playing many
hands of this repo's own FixedLimitOmahaHiLo through a small adapter -- a
stand-in for the eventual "trained EvalAgent vs. real people" bridge.
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.expanduser("~/Documents/omaha"))
from abc_omaha_hilo import ABCBot  # noqa: E402
from bayesian_bot import BayesianBot  # noqa: E402
from pokerkit import Card  # noqa: E402

from PokerRL.game.games import FixedLimitOmahaHiLo  # noqa: E402
from PokerRL.game.Poker import Poker  # noqa: E402

N_HANDS = 200
START_CHIPS = 2000

RANKS = "23456789TJQKA"
SUITS = "hdsc"


def card_str(c):
    return RANKS[c[0]] + SUITS[c[1]]


def cards_str(cards_2d):
    return "".join(card_str(c) for c in cards_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D)


def decide_action(bot, env, seat, history):
    hole = list(Card.parse(cards_str(env.seats[seat].hand)))
    board = list(Card.parse(cards_str(env.board)))
    to_call = env._get_biggest_bet_out_there_aka_total_to_call() - env.seats[seat].current_bet
    # env.main_pot only holds bets already swept in between rounds -- add the
    # bets committed so far *this* round (e.g. the blinds preflop) so bots
    # doing pot-odds math see the real pot, not an artificially empty one.
    pot = env.main_pot + sum(p.current_bet for p in env.seats)
    can_raise = Poker.BET_RAISE in env.get_legal_actions()
    street = Poker.INT2STRING_ROUND[env.current_round]

    decision = bot.decide(hole, board, pot, to_call, can_raise, street,
                          history=history, hero_index=seat)
    if decision == "raise" and can_raise:
        return Poker.BET_RAISE
    if decision == "fold" and Poker.FOLD in env.get_legal_actions():
        return Poker.FOLD
    return Poker.CHECK_CALL


args = FixedLimitOmahaHiLo.ARGS_CLS(n_seats=2,
                                    starting_stack_sizes_list=[START_CHIPS] * 2,
                                    stack_randomization_range=(0, 0))
env = FixedLimitOmahaHiLo(env_args=args, lut_holder=FixedLimitOmahaHiLo.get_lut_holder(), is_evaluating=True)

bots = {0: BayesianBot(name="Bayesian"), 1: ABCBot()}
names = {0: "BayesianBot", 1: "ABCBot"}

wins = {0: 0, 1: 0}
splits = 0
crashes = 0
chip_errors = 0

for hand_no in range(N_HANDS):
    # NOTE: is_evaluating=True resets every player to a fixed baseline stack
    # on each .reset() (see _PokerPlayer.reset()), rather than continuing
    # from the previous hand's ending stack. So "before" must be captured
    # *after* reset (post-blinds), not before it -- comparing across hands
    # would spuriously read as a "push" every time, since both snapshots
    # land on the same fixed baseline.
    env.reset()
    stacks_before = [p.stack for p in env.seats]
    # include current_bet: blinds are already "in flight" out of stack right
    # after reset, so stack alone undercounts total chips at this snapshot.
    total_before = sum(p.stack + p.current_bet for p in env.seats)
    history = []

    done = False
    steps = 0
    try:
        while not done:
            seat = env.current_player.seat_id
            action = decide_action(bots[seat], env, seat, history)
            street_str = Poker.INT2STRING_ROUND[env.current_round]

            action_str = {Poker.FOLD: "fold", Poker.CHECK_CALL: "call", Poker.BET_RAISE: "raise"}[action]
            history.append({"actor": seat, "street": street_str, "action": action_str})

            _obs, _rew, done, _info = env.step(action)
            steps += 1
            if steps > 200:
                raise RuntimeError("hand did not terminate after 200 steps")
    except Exception as e:
        crashes += 1
        print(f"CRASH on hand {hand_no}: {e}")
        break

    stacks_after = [p.stack for p in env.seats]
    if sum(stacks_after) != total_before:
        chip_errors += 1
        print(f"CHIP MISMATCH hand {hand_no}: before={total_before} after={sum(stacks_after)}")

    delta0 = stacks_after[0] - stacks_before[0]
    if delta0 > 0:
        wins[0] += 1
    elif delta0 < 0:
        wins[1] += 1
    else:
        splits += 1

print(f"\n{N_HANDS} hands played. crashes={crashes} chip_mismatches={chip_errors}")
print(f"{names[0]} won {wins[0]} hands, {names[1]} won {wins[1]} hands, {splits} pushes")
print(f"Final stacks: {names[0]}={env.seats[0].stack}  {names[1]}={env.seats[1].stack}")
if wins[0] + wins[1] > 0:
    print(f"{names[0]} win rate (of decisive hands): {wins[0] / (wins[0] + wins[1]):.1%}")
