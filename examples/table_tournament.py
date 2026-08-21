"""
N-seat table tournament for FixedLimitOmahaHiLo -- the multiway evaluation
harness the repo lacked (AgentTournament_hu is heads-up only: hardcoded
`1 - seat`, scores by negation).

Seats take any mix of players:
    bayes | abc | caller | random          rule bots from ~/Documents/omaha
    agent:/path/to/eval_agentSINGLE.pkl    a trained EvalAgentDeepCFR

IMPORTANT: a DeepCFR agent can only sit at a table whose seat count matches the
n_seats it was TRAINED with -- the observation width and one network layer are
seat-count-dependent, and its internal mirror env is built from its training
profile. Current snapshots are heads-up (n_seats=2); seating one at a 4-seat
table raises immediately with a clear error. Bots are N-player clean.

Fairness and variance control:
  - every deck is played N_SEATS times, rotating the player->seat assignment
    cyclically, so each player sees every position on identical cards
    (duplicate poker); one sample per player = mean over its N rotations
  - odd chips are deterministic (hi side, then left of button), so mirrored
    legs are exactly comparable

Scores are per-player mBB/hand with a 95% CI over deck-samples; the per-deck
table sum is asserted ~0 (chip conservation).

Run:
    python3 examples/table_tournament.py --seats bayes abc caller random --decks 250
    python3 examples/table_tournament.py --seats bayes abc --decks 500      # heads-up works too
"""
import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.expanduser("~/Documents/omaha"))
from abc_omaha_hilo import ABCBot, CallerBot, RandomBot  # noqa: E402
from bayesian_bot import BayesianBot  # noqa: E402
from pokerkit import Card as PKCard  # noqa: E402

import numpy as np  # noqa: E402

from PokerRL.game.games import FixedLimitOmahaHiLo  # noqa: E402
from PokerRL.game.Poker import Poker  # noqa: E402

RANKS = "23456789TJQKA"
SUITS = "hdsc"
ACTION_STR = {Poker.FOLD: "fold", Poker.CHECK_CALL: "call", Poker.BET_RAISE: "raise"}


def cards_str(cards_2d):
    return "".join(RANKS[c[0]] + SUITS[c[1]] for c in cards_2d
                   if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D)


class BotSeat:
    """Adapter for the ~/Documents/omaha rule bots (proven translation layer
    from bot_dry_run_FLO_HiLo.py / eval_agent_vs_bots.py)."""

    def __init__(self, name, bot):
        self.name = name
        self._bot = bot

    def new_hand(self, env, deck_state_dict):
        pass

    def act(self, env, seat_id, history):
        hole = list(PKCard.parse(cards_str(env.seats[seat_id].hand)))
        board = list(PKCard.parse(cards_str(env.board)))
        to_call = env._get_biggest_bet_out_there_aka_total_to_call() - env.seats[seat_id].current_bet
        pot = env.main_pot + sum(env.side_pots) + sum(p.current_bet for p in env.seats)
        can_raise = Poker.BET_RAISE in env.get_legal_actions()
        street = Poker.INT2STRING_ROUND[env.current_round]
        decision = self._bot.decide(hole, board, pot, to_call, can_raise, street,
                                    history=history, hero_index=seat_id)
        if decision == "raise" and can_raise:
            return Poker.BET_RAISE
        if decision == "fold" and Poker.FOLD in env.get_legal_actions():
            return Poker.FOLD
        return Poker.CHECK_CALL

    def observe(self, actor_seat, action_int):
        pass  # bots read the shared history instead


class DeepCFRSeat:
    """Adapter for a trained EvalAgentDeepCFR. Mirrors the table in the agent's
    internal env, so the agent's training n_seats must equal the table's."""

    def __init__(self, name, path, table_n_seats):
        from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
        self.name = name
        self._agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path)
        trained_seats = self._agent.env_bldr.N_SEATS
        if trained_seats != table_n_seats:
            raise SystemExit(
                f"'{name}' was trained heads-up (n_seats={trained_seats}) and cannot sit at a "
                f"{table_n_seats}-seat table: its observation width and one net layer are "
                f"seat-count-dependent. Train a run with n_seats={table_n_seats} first.")

    def new_hand(self, env, deck_state_dict):
        self._agent.reset(deck_state_dict=deck_state_dict)

    def act(self, env, seat_id, history):
        action_int, _ = self._agent.get_action(step_env=True, need_probs=False)
        return action_int

    def observe(self, actor_seat, action_int):
        self._agent.notify_of_action(p_id_acted=actor_seat, action_he_did=action_int)


def make_seat(spec, table_n_seats):
    if spec == "bayes":
        return BotSeat("BayesianBot", BayesianBot(name="Bayesian"))
    if spec == "abc":
        return BotSeat("ABCBot", ABCBot())
    if spec == "caller":
        return BotSeat("CallerBot", CallerBot())
    if spec == "random":
        return BotSeat("RandomBot", RandomBot())
    if spec.startswith("agent:"):
        path = os.path.expanduser(spec.split(":", 1)[1])
        return DeepCFRSeat(os.path.basename(os.path.dirname(os.path.dirname(path))) or "agent",
                           path, table_n_seats)
    raise SystemExit(f"unknown seat spec '{spec}' (want bayes|abc|caller|random|agent:<pkl>)")


def play_hand(env, seat_map, deck_state_dict, history):
    """One hand: seat_map[seat_id] -> player. Returns per-seat rewards."""
    _, r_for_all, done, _ = env.reset(deck_state_dict=deck_state_dict)
    for p in set(seat_map):
        p.new_hand(env, env.cards_state_dict())
    while not done:
        actor = env.current_player.seat_id
        street = Poker.INT2STRING_ROUND[env.current_round]
        action = seat_map[actor].act(env, actor, history)
        history.append({"actor": actor, "street": street, "action": ACTION_STR[action]})
        for s, p in enumerate(seat_map):
            if s != actor:
                p.observe(actor, action)
        _, r_for_all, done, _ = env.step(action)
    return [r * env.REWARD_SCALAR * env.EV_NORMALIZER for r in r_for_all]


def run_table(seat_specs, n_decks, start_chips=2000):
    n = len(seat_specs)
    players = [make_seat(s, n) for s in seat_specs]
    args = FixedLimitOmahaHiLo.ARGS_CLS(n_seats=n,
                                        starting_stack_sizes_list=[start_chips] * n,
                                        stack_randomization_range=(0, 0))
    env = FixedLimitOmahaHiLo(env_args=args, is_evaluating=True,
                              lut_holder=FixedLimitOmahaHiLo.get_lut_holder())

    # samples[d, i] = player i's mean mBB/hand over the n rotations of deck d
    samples = np.zeros((n_decks, n), dtype=np.float64)

    for d in range(n_decks):
        env.reset()
        deck = env.cards_state_dict()
        per_player = np.zeros(n)
        for offset in range(n):
            # player i sits at seat (i + offset) % n
            seat_map = [None] * n
            for i, p in enumerate(players):
                seat_map[(i + offset) % n] = p
            history = []
            rews = play_hand(env, seat_map, deck, history)
            assert abs(sum(rews)) < 1e-6 * max(1, abs(max(rews))) + 1e-6, \
                f"chips not conserved on deck {d} offset {offset}: {rews}"
            for i in range(n):
                per_player[i] += rews[(i + offset) % n]
        samples[d] = per_player / n
        if d % 50 == 0:
            print(f"deck {d}/{n_decks}")

    print(f"\n{n_decks} decks x {n} rotations = {n_decks * n} hands per player\n")
    order = np.argsort(-samples.mean(axis=0))
    for i in order:
        m = samples[:, i].mean()
        ci = 1.96 * samples[:, i].std() / np.sqrt(n_decks)
        print(f"  {players[i].name:<14} {m:+9.1f} mBB/hand  [{m - ci:+8.1f}, {m + ci:+8.1f}]")
    total = samples.sum()
    print(f"\n(table sum {total:+.6f} -- must be ~0)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", nargs="+", required=True,
                    help="bayes|abc|caller|random|agent:<eval_agentSINGLE.pkl>, one per seat (2-6)")
    ap.add_argument("--decks", type=int, default=250,
                    help="decks per rotation cycle; hands per player = decks * n_seats")
    a = ap.parse_args()
    if not 2 <= len(a.seats) <= 6:
        raise SystemExit("need 2-6 seats")
    run_table(a.seats, a.decks)
