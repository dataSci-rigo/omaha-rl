"""
Plays a trained EvalAgentDeepCFR snapshot head-to-head against ABCBot and
BayesianBot (both from ~/Documents/omaha) -- the actual "did we beat them
yet" benchmark, as opposed to nightly_progress_check.py's self-play-only
"tonight vs last night" plateau check.

Reuses the same bot-adapter pattern validated in bot_dry_run_FLO_HiLo.py
(card-string conversion, to_call/pot computation, action-history tracking
for BayesianBot's opponent modeling) plus the eval_agent.reset()/get_action()/
notify_of_action() pattern from AgentTournament_hu.py.

Also compares "new" (latest) vs. "old" (an earlier snapshot, e.g. step 0)
via AgentTournament, the same approach nightly_progress_check.py uses.

Run:  python3 examples/eval_agent_vs_bots.py [--step N] [--old-step N]
"""
import argparse
import gc
import os
import re
import sys

os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.expanduser("~/Documents/omaha"))
from abc_omaha_hilo import ABCBot  # noqa: E402
from bayesian_bot import BayesianBot  # noqa: E402
from pokerkit import Card as PKCard  # noqa: E402

import numpy as np  # noqa: E402

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR  # noqa: E402
from MCCFR.EvalAgentMCCFR import EvalAgentMCCFR  # noqa: E402
from PokerRL.game.AgentTournament_hu import AgentTournament  # noqa: E402
from PokerRL.game.games import FixedLimitOmahaHiLo  # noqa: E402
from PokerRL.game.Poker import Poker  # noqa: E402

PROFILE_NAME = "FLO_HiLo_HU_dense_residual"
DATA_PATH = os.path.expanduser("~/poker_ai_data")
DEFAULT_HANDS = 500  # total hands per opponent, split evenly across both seats

RANKS = "23456789TJQKA"
SUITS = "hdsc"

# DriverBase.__init__ silently appends "_" to the profile name on every
# resume (see FLO_HiLo_nightly_run.py's module docstring for why) -- exports
# alternate between these two names cycle to cycle, so any lookup by step
# has to check both.
_CANDIDATE_NAMES = (PROFILE_NAME, PROFILE_NAME + "_")


def card_str(c):
    return RANKS[c[0]] + SUITS[c[1]]


def cards_str(cards_2d):
    return "".join(card_str(c) for c in cards_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D)


def find_last_eval_agent_step():
    """Returns (name, step) for the highest step across both candidate names."""
    best_name, best_step = None, None
    for name in _CANDIDATE_NAMES:
        eval_agent_dir = os.path.join(DATA_PATH, "eval_agent", name)
        if not os.path.isdir(eval_agent_dir):
            continue
        steps = [int(d) for d in os.listdir(eval_agent_dir)
                if re.fullmatch(r"\d+", d)
                and os.path.exists(os.path.join(eval_agent_dir, d, "eval_agentSINGLE.pkl"))]
        if steps:
            step = max(steps)
            if best_step is None or step > best_step:
                best_name, best_step = name, step
    return best_name, best_step


def find_eval_agent_step_exact(step):
    """Returns the name whose directory actually has this exact step."""
    for name in _CANDIDATE_NAMES:
        p = os.path.join(DATA_PATH, "eval_agent", name, str(step), "eval_agentSINGLE.pkl")
        if os.path.exists(p):
            return name
    return None


def eval_agent_path(name, step):
    return os.path.join(DATA_PATH, "eval_agent", name, str(step), "eval_agentSINGLE.pkl")


def load_eval_agent(path):
    """
    Loads the complete agent -- every network snapshot, i.e. the exact SD-CFR
    weighted average.

    This used to trim the StrategyBuffer down to the N most recent nets, because
    a full agent cost ~20.2GB at step 70 and was OOM-killing the machine. That
    cost turned out to be a bug, not a property of the algorithm: every net was
    allocating its own float32 copy of the range LUTs (~140MB each). With those
    shared (PokerRL/rl/neural/_shared_luts.py) a full agent is a few tens of MB,
    so there is nothing left to trim around.

    Trimming is not merely unnecessary now, it was harmful: keeping the newest N
    is a biased estimator. Weights are linear in cfr_iteration, so the newest 8
    of 70 carry only ~23% of total weight -- the other ~77% was silently
    discarded. As N shrinks it converges toward CFR's *current* iterate, which
    has no convergence guarantee; only the average does. Any benchmark run
    through the old trimmer measured that biased approximation, not this agent.
    Dispatches on the pickled profile type, so Deep CFR and tabular MCCFR
    snapshots can be benchmarked through the identical harness: both agent
    classes expose the same reset/get_action/notify_of_action surface.
    """
    from PokerRL.util.file_util import load_pickle
    state = load_pickle(path=path)
    if type(state["t_prof"]).__name__ == "MCCFRProfile":
        agent = EvalAgentMCCFR(t_prof=state["t_prof"])
        agent.load_state_dict(state=state)
        return agent
    return EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path)


def describe_strat_buffers(agent, label):
    if not hasattr(agent, "_strategy_buffers"):
        return  # tabular MCCFR agent: no net buffers to describe
    sizes = [b.size for b in agent._strategy_buffers]
    print(f"  ({label}: strategy nets per seat = {sizes})")


def bot_decision(bot, env, bot_seat, history):
    hole = list(PKCard.parse(cards_str(env.seats[bot_seat].hand)))
    board = list(PKCard.parse(cards_str(env.board)))
    to_call = env._get_biggest_bet_out_there_aka_total_to_call() - env.seats[bot_seat].current_bet
    pot = env.main_pot + sum(p.current_bet for p in env.seats)
    can_raise = Poker.BET_RAISE in env.get_legal_actions()
    street = Poker.INT2STRING_ROUND[env.current_round]
    decision = bot.decide(hole, board, pot, to_call, can_raise, street, history=history, hero_index=bot_seat)
    if decision == "raise" and can_raise:
        return Poker.BET_RAISE
    if decision == "fold" and Poker.FOLD in env.get_legal_actions():
        return Poker.FOLD
    return Poker.CHECK_CALL


def play_match(eval_agent, bot, bot_name, n_hands_per_seat):
    env_bldr = eval_agent.env_bldr
    env = env_bldr.env_cls(env_args=env_bldr.env_args, is_evaluating=True, lut_holder=env_bldr.lut_holder)
    winnings = np.empty(shape=(n_hands_per_seat * env.N_SEATS), dtype=np.float32)

    for agent_seat in range(env.N_SEATS):
        bot_seat = 1 - agent_seat
        for hand_nr in range(n_hands_per_seat):
            env.reset()
            eval_agent.reset(deck_state_dict=env.cards_state_dict())
            history = []
            done = False
            while not done:
                actor = env.current_player.seat_id
                street_str = Poker.INT2STRING_ROUND[env.current_round]
                if actor == agent_seat:
                    action, _ = eval_agent.get_action(step_env=True, need_probs=False)
                else:
                    action = bot_decision(bot, env, bot_seat, history)
                    _obs, r_for_all, done, _info = env.step(action)
                    eval_agent.notify_of_action(p_id_acted=bot_seat, action_he_did=action)
                    continue
                action_str = {Poker.FOLD: "fold", Poker.CHECK_CALL: "call", Poker.BET_RAISE: "raise"}[action]
                history.append({"actor": actor, "street": street_str, "action": action_str})
                # eval_agent.get_action(step_env=True) already stepped its
                # internal mirrored env; step the real env with the same action.
                _obs, r_for_all, done, _info = env.step(action)

            winnings[hand_nr + (agent_seat * n_hands_per_seat)] = (
                r_for_all[agent_seat] * env.REWARD_SCALAR * env.EV_NORMALIZER
            )
            # scale progress reporting to run length so short runs still show life
            if hand_nr % max(1, n_hands_per_seat // 10) == 0:
                print(f"  [{bot_name}] seat {agent_seat}: hand {hand_nr}/{n_hands_per_seat}")

    mean = float(np.mean(winnings))
    std = float(np.std(winnings))
    ci = 1.96 * std / np.sqrt(len(winnings))
    return mean, mean - ci, mean + ci


def play_bot_match(bot_a, bot_b, name_a, n_hands_per_seat):
    """
    ABCBot vs BayesianBot, no trained agent involved -- the baseline that makes
    the agent's two results interpretable. If the agent loses badly to ABCBot
    but ties BayesianBot, this answers whether ABCBot simply dominates
    BayesianBot as well (ranking is self-consistent, agent sits around
    BayesianBot's level) or whether the bots are close to each other (which
    would instead point at something specific about how the agent plays ABCBot).

    Returns (mean, lower95, upper95) in milliBB/hand from bot_a's perspective,
    on the same scale as play_match() so the numbers are directly comparable.
    bot_a occupies each seat for n_hands_per_seat hands to cancel positional edge.
    """
    env_args = FixedLimitOmahaHiLo.ARGS_CLS(n_seats=2,
                                            starting_stack_sizes_list=[2000] * 2,
                                            stack_randomization_range=(0, 0))
    env = FixedLimitOmahaHiLo(env_args=env_args,
                              lut_holder=FixedLimitOmahaHiLo.get_lut_holder(),
                              is_evaluating=True)
    winnings = np.empty(shape=(n_hands_per_seat * env.N_SEATS), dtype=np.float32)

    for a_seat in range(env.N_SEATS):
        seats = {a_seat: bot_a, 1 - a_seat: bot_b}
        for hand_nr in range(n_hands_per_seat):
            env.reset()
            # Every action is recorded with its actor. Each bot's
            # _opponent_estimate() skips entries whose actor == its own
            # hero_index, so one shared history serves both correctly.
            history = []
            done = False
            r_for_all = None
            while not done:
                actor = env.current_player.seat_id
                street_str = Poker.INT2STRING_ROUND[env.current_round]
                action = bot_decision(seats[actor], env, actor, history)
                history.append({
                    "actor": actor,
                    "street": street_str,
                    "action": {Poker.FOLD: "fold", Poker.CHECK_CALL: "call",
                               Poker.BET_RAISE: "raise"}[action],
                })
                _obs, r_for_all, done, _info = env.step(action)

            winnings[hand_nr + (a_seat * n_hands_per_seat)] = (
                r_for_all[a_seat] * env.REWARD_SCALAR * env.EV_NORMALIZER
            )
            if hand_nr % max(1, n_hands_per_seat // 10) == 0:
                print(f"  [{name_a} in seat {a_seat}] hand {hand_nr}/{n_hands_per_seat}")

    mean = float(np.mean(winnings))
    std = float(np.std(winnings))
    ci = 1.96 * std / np.sqrt(len(winnings))
    return mean, mean - ci, mean + ci


def verdict_for(lower95, upper95):
    if lower95 > 0:
        return "BEATING"
    if upper95 < 0:
        return "LOSING TO"
    return "STATISTICALLY TIED WITH"


def build_parser():
    p = argparse.ArgumentParser(
        prog="eval_agent_vs_bots.py",
        description="Benchmark a trained FLO Hi/Lo agent against ABCBot, BayesianBot, "
                    "and/or an older snapshot of itself.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # 500 hands vs ABCBot (the default, fast + memory-light)
  %(prog)s

  # 500 hands vs BayesianBot only
  %(prog)s --bot bayesian

  # 2000 hands vs every opponent, including the step-0 snapshot
  %(prog)s --bot all --hands 2000

  # evaluate the TRUE untrimmed agent (warning: 20GB+ and slow)
  %(prog)s --max-strat-nets 0
""")
    p.add_argument("--bot", choices=["abc", "bayesian", "old", "baseline", "all"], default="abc",
                   help="opponent(s) to play: abc (default), bayesian, old (earlier snapshot), "
                        "baseline (ABCBot vs BayesianBot, no agent -- a sanity check on the "
                        "other results), or all")
    p.add_argument("--hands", type=int, default=DEFAULT_HANDS,
                   help=f"total hands per opponent, split evenly across both seats (default: {DEFAULT_HANDS})")
    p.add_argument("--step", type=int, default=None,
                   help="eval_agent step to benchmark (default: the latest available)")
    p.add_argument("--old-step", type=int, default=0,
                   help="eval_agent step to use as the 'old' opponent (default: 0)")
    p.add_argument("--profile", default=None,
                   help="eval_agent profile name to benchmark (default: "
                        f"{PROFILE_NAME}); e.g. MCCFR_M1 for the tabular agent")
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()

    if args.profile:
        PROFILE_NAME = args.profile
        _CANDIDATE_NAMES = (PROFILE_NAME, PROFILE_NAME + "_")

    if args.hands < 2:
        raise SystemExit("--hands must be at least 2 (one per seat).")
    n_per_seat = args.hands // 2
    total = n_per_seat * 2

    if args.step is not None:
        new_name = find_eval_agent_step_exact(args.step)
        new_step = args.step if new_name is not None else None
    else:
        new_name, new_step = find_last_eval_agent_step()
    if new_step is None:
        raise SystemExit(f"No eval_agent export found for profile '{PROFILE_NAME}' at the requested step.")

    new_path = eval_agent_path(new_name, new_step)
    print(f"Benchmarking step {new_step} ({new_path})")
    print(f"Hands per opponent: {total} ({n_per_seat} per seat)\n")

    run_abc = args.bot in ("abc", "all")
    run_bayes = args.bot in ("bayesian", "all")
    run_old = args.bot in ("old", "all")
    run_baseline = args.bot in ("baseline", "all")

    if run_baseline:
        print(f"=== BASELINE: ABCBot vs. BayesianBot (no agent): {total} hands ===")
        mean_mbb, lower95, upper95 = play_bot_match(ABCBot(), BayesianBot(name="Bayesian"),
                                                    "ABCBot", n_per_seat)
        print(f"ABCBot vs BayesianBot: {mean_mbb:+.1f} milliBB/hand [{lower95:+.1f}, {upper95:+.1f}] (95% CI)")
        print(f"VERDICT: ABCBot is {verdict_for(lower95, upper95)} BayesianBot\n")

    matchups = []
    if run_abc:
        matchups.append(("ABCBot", ABCBot))
    if run_bayes:
        matchups.append(("BayesianBot", lambda: BayesianBot(name="Bayesian")))

    # Only one agent is held in memory at a time -- loading the next before
    # releasing the previous would transiently need both.
    eval_agent = None
    for bot_name, bot_factory in matchups:
        del eval_agent
        gc.collect()
        eval_agent = load_eval_agent(new_path)  # fresh internal env state per match
        describe_strat_buffers(eval_agent, "loaded")
        bot = bot_factory()
        print(f"=== Trained agent (step {new_step}) vs. {bot_name}: {total} hands ===")
        mean_mbb, lower95, upper95 = play_match(eval_agent, bot, bot_name, n_per_seat)
        print(f"Trained agent vs {bot_name}: {mean_mbb:+.1f} milliBB/hand [{lower95:+.1f}, {upper95:+.1f}] (95% CI)")
        print(f"VERDICT: {verdict_for(lower95, upper95)} {bot_name}\n")
    del eval_agent
    gc.collect()

    if run_old:
        old_name = find_eval_agent_step_exact(args.old_step)
        if old_name is None or (old_name == new_name and args.old_step == new_step):
            print(f"No distinct 'old' snapshot at step {args.old_step} to compare against -- skipping new-vs-old.")
        else:
            old_path = eval_agent_path(old_name, args.old_step)
            print(f"=== New (step {new_step}) vs. Old (step {args.old_step}): {total} hands ===")
            # Both agents must be live at once for the tournament.
            new_agent = load_eval_agent(new_path)
            old_agent = load_eval_agent(old_path)
            tourney = AgentTournament(env_cls=new_agent.env_bldr.env_cls,
                                      env_args=new_agent.env_bldr.env_args,
                                      eval_agent_1=new_agent,
                                      eval_agent_2=old_agent,
                                      logfile=None)
            # NOTE: AgentTournament.run() returns (mean, UPPER, LOWER) -- not
            # (mean, lower, upper) like play_match(). Unpacking it in the wrong
            # order printed the CI backwards and, worse, handed verdict_for()
            # swapped bounds, which can invert the verdict for any result near
            # zero -- i.e. exactly the plateau case this comparison exists to
            # detect. nightly_progress_check.py unpacks it correctly.
            mean_mbb, upper95, lower95 = tourney.run(n_games_per_seat=n_per_seat)
            print(f"New (step {new_step}) vs Old (step {args.old_step}): "
                 f"{mean_mbb:+.1f} milliBB/hand [{lower95:+.1f}, {upper95:+.1f}] (95% CI)")
            print(f"VERDICT: New agent is {verdict_for(lower95, upper95)} the old (step {args.old_step}) snapshot\n")
            del new_agent, old_agent
            gc.collect()

    print("Benchmark complete.")
