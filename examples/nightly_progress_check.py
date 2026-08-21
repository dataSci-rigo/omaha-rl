"""
Compares "tonight's" trained agent against "last night's" (the agent from
the start of the current training session, recorded by
FLO_HiLo_nightly_run.py's session_start_eval_agent_step.txt marker) by
playing them head-to-head via PokerRL's own AgentTournament -- the same
approach examples/interactive_agent_v_agent.py uses to compare two saved
EvalAgent snapshots.

Prints a milliBB/hand result with a 95% confidence interval and a verdict:
  IMPROVED   -- tonight's agent is ahead, and the interval excludes 0
  REGRESSED  -- tonight's agent is behind, and the interval excludes 0
  PLATEAU    -- can't tell apart at this sample size (the "stop training"
                signal, per the criterion of "tonight can't beat last night")

Run manually any morning:  python3 examples/nightly_progress_check.py
"""
import json
import os
import re

os.environ["OMP_NUM_THREADS"] = "1"

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.AgentTournament_hu import AgentTournament

PROFILE_NAME = "FLO_HiLo_HU_dense_residual"
DATA_PATH = os.path.expanduser("~/poker_ai_data")
SESSION_START_MARKER = os.path.join(DATA_PATH, "checkpoint", PROFILE_NAME, "session_start_eval_agent_step.txt")
# 20,000 hands total, split evenly across seats. Raised from 3,000/seat on Aug 20:
# 6,000 hands gives a 95% CI of roughly +/-148 mBB/hand, which cannot resolve the
# size of gain a night of training actually produces -- so this check reported
# PLATEAU essentially unconditionally. Measured cost at 10,000/seat is ~190s for
# a CI of ~+/-91, which is free at 07:15.
N_GAMES_PER_SEAT = 10000

# DriverBase.__init__ silently appends "_" to the profile name on every
# resume (see FLO_HiLo_nightly_run.py's module docstring for why) -- exports
# alternate between these two names cycle to cycle, so any lookup by step
# has to check both.
_CANDIDATE_NAMES = (PROFILE_NAME, PROFILE_NAME + "_")


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
    """Returns the name whose directory actually has this exact step (for
    resolving "last night's" step, which could be under either name)."""
    for name in _CANDIDATE_NAMES:
        p = os.path.join(DATA_PATH, "eval_agent", name, str(step), "eval_agentSINGLE.pkl")
        if os.path.exists(p):
            return name
    return None


def eval_agent_path(name, step):
    return os.path.join(DATA_PATH, "eval_agent", name, str(step), "eval_agentSINGLE.pkl")


if __name__ == '__main__':
    if not os.path.exists(SESSION_START_MARKER):
        print("No session_start marker yet -- this is written at the start of each "
             "night's run by FLO_HiLo_nightly_run.py. Nothing to compare until at "
             "least one full night has run since this tool was added.")
        raise SystemExit(0)  # not a failure -- just nothing to report yet
    # Marker is JSON ({"step": N, "written": iso}) since Aug 20; older markers are
    # a bare int. See FLO_HiLo_nightly_run.record_session_start_marker().
    with open(SESSION_START_MARKER) as f:
        raw = f.read().strip()
    try:
        last_night_step = json.loads(raw)["step"]
    except (ValueError, KeyError, TypeError):
        last_night_step = int(raw)
    last_night_name = find_eval_agent_step_exact(last_night_step)
    if last_night_name is None:
        print(f"Marker points at step {last_night_step} but no eval_agent export for it exists "
             f"under either candidate name -- can't compare.")
        raise SystemExit(0)

    tonight_name, tonight_step = find_last_eval_agent_step()
    if tonight_step is None or tonight_step <= last_night_step:
        print(f"No new eval_agent export since last night's marker (step {last_night_step}). "
             f"Nothing to compare yet -- check back after training has run further.")
        raise SystemExit(0)  # not a failure -- just nothing to report yet

    print(f"Comparing last night (step {last_night_step}) vs. tonight (step {tonight_step}) "
         f"over {N_GAMES_PER_SEAT * 2} hands...\n")

    last_night_agent = EvalAgentDeepCFR.load_from_disk(
        path_to_eval_agent=eval_agent_path(last_night_name, last_night_step))
    tonight_agent = EvalAgentDeepCFR.load_from_disk(
        path_to_eval_agent=eval_agent_path(tonight_name, tonight_step))

    tourney = AgentTournament(env_cls=tonight_agent.env_bldr.env_cls,
                              env_args=tonight_agent.env_bldr.env_args,
                              eval_agent_1=tonight_agent,
                              eval_agent_2=last_night_agent,
                              logfile=None)
    mean_mbb, upper95, lower95 = tourney.run(n_games_per_seat=N_GAMES_PER_SEAT)

    print(f"\nTonight (step {tonight_step}) vs. last night (step {last_night_step}): "
         f"{mean_mbb:+.1f} milliBB/hand [{lower95:+.1f}, {upper95:+.1f}] (95% CI)\n")

    if lower95 > 0:
        print("VERDICT: IMPROVED -- tonight's agent is ahead of last night's, keep training.")
    elif upper95 < 0:
        print("VERDICT: REGRESSED -- tonight's agent is behind last night's. Worth investigating "
             "(e.g. a bad LR step) before continuing, not necessarily stopping.")
    else:
        print("VERDICT: PLATEAU -- can't statistically distinguish tonight's agent from last "
             "night's at this sample size. This is the 'stop training' signal you asked for.")
