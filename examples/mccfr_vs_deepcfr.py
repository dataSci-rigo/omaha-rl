"""
Head-to-head: tabular MCCFR snapshot vs Deep CFR snapshot (cross-profile).

eval_agent_vs_bots.py's new-vs-old only compares snapshots of one profile;
this script runs AgentTournament_hu across the two agent types. Positive
result = MCCFR is winning.

    python3 examples/mccfr_vs_deepcfr.py --hands 20000
    python3 examples/mccfr_vs_deepcfr.py --mccfr-step 572000 --deepcfr-step 170
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR             # noqa: E402
from MCCFR.EvalAgentMCCFR import EvalAgentMCCFR                   # noqa: E402
from PokerRL.game.AgentTournament_hu import AgentTournament       # noqa: E402
from PokerRL.util.file_util import load_pickle                    # noqa: E402

DATA_PATH = os.path.expanduser("~/poker_ai_data")
MCCFR_PROFILE = "MCCFR_M1"
# DriverBase appends "_" on resume: check both (see FLO_HiLo_nightly_run.py)
DEEPCFR_NAMES = ("FLO_HiLo_HU_dense_residual", "FLO_HiLo_HU_dense_residual_")


def latest(names, step=None):
    best = None
    for name in names:
        d = os.path.join(DATA_PATH, "eval_agent", name)
        if not os.path.isdir(d):
            continue
        for s in os.listdir(d):
            if not re.fullmatch(r"\d+", s):
                continue
            if step is not None and int(s) != step:
                continue
            p = os.path.join(d, s, "eval_agentSINGLE.pkl")
            if os.path.exists(p) and (best is None or int(s) > best[0]):
                best = (int(s), p)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--mccfr-step", type=int, default=None)
    ap.add_argument("--deepcfr-step", type=int, default=None)
    args = ap.parse_args()

    m = latest((MCCFR_PROFILE, MCCFR_PROFILE + "_"), args.mccfr_step)
    d = latest(DEEPCFR_NAMES, args.deepcfr_step)
    if m is None or d is None:
        sys.exit(f"missing snapshot: mccfr={m}, deepcfr={d}")

    print(f"MCCFR step {m[0]} vs Deep CFR step {d[0]}, {args.hands} hands")

    m_state = load_pickle(path=m[1])
    mccfr = EvalAgentMCCFR(t_prof=m_state["t_prof"])
    mccfr.load_state_dict(m_state)
    deepcfr = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=d[1])

    tourney = AgentTournament(env_cls=mccfr.env_bldr.env_cls,
                              env_args=mccfr.env_bldr.env_args,
                              eval_agent_1=mccfr,
                              eval_agent_2=deepcfr,
                              logfile=None)
    # NOTE: returns (mean, UPPER, LOWER) -- the documented ordering trap
    mean, upper95, lower95 = tourney.run(n_games_per_seat=args.hands // 2)
    print(f"MCCFR (step {m[0]}) vs Deep CFR (step {d[0]}): "
          f"{mean:+.1f} milliBB/hand [{lower95:+.1f}, {upper95:+.1f}] (95% CI)")
    if lower95 > 0:
        print("VERDICT: MCCFR is BEATING Deep CFR")
    elif upper95 < 0:
        print("VERDICT: MCCFR is LOSING TO Deep CFR")
    else:
        print("VERDICT: STATISTICALLY TIED")


if __name__ == "__main__":
    main()
