"""
Tabular ES-MCCFR training runner (Milestone 1: M1 equity-bin abstraction).

Usage:
    python3 examples/mccfr_run.py --profile MCCFR_M1 --iters 2000000 [--resume]

Conventions shared with the Deep CFR scripts:
- artifacts under ~/poker_ai_data/{checkpoint,eval_agent}/<profile>/
- eval agent snapshots at eval_agent/<profile>/<iteration>/eval_agentSINGLE.pkl,
  the exact layout examples/eval_agent_vs_bots.py discovers
- refuses to run inside the 23:00-07:00 window (that belongs to the Deep CFR
  systemd units) unless --allow-night; if a run crosses into it, it
  checkpoints and exits cleanly
- SIGTERM/SIGINT checkpoint and exit; --resume continues exactly (tables,
  iteration counter, solver RNG). The bucketer cache is not checkpointed --
  it refills on resume; only throughput, not correctness, depends on it.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import datetime
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MCCFR.abstraction.m1_bins import M1Bucketer                  # noqa: E402
from MCCFR.betting_tree import load_or_build_tree                 # noqa: E402
from MCCFR.EvalAgentMCCFR import EvalAgentMCCFR                   # noqa: E402
from MCCFR.profile import MCCFRProfile                            # noqa: E402
from MCCFR.solver import ESMCCFRSolver                            # noqa: E402

DATA_PATH = os.path.expanduser(os.environ.get("MCCFR_DATA_PATH",
                                              "~/poker_ai_data"))
CHUNK_ITERS = 200


def in_night_window(now=None):
    now = now or datetime.datetime.now()
    return now.hour >= 23 or now.hour < 7


def export_eval_agent(solver, args):
    t_prof = MCCFRProfile(name=args.profile, k_postflop=args.k_postflop,
                          n_rollouts=args.rollouts, n_seats=args.n_seats)
    agent = EvalAgentMCCFR(t_prof=t_prof)
    agent.update_weights({"avg_strategy": solver.average_strategy(),
                          "K": solver.K})
    path = os.path.join(DATA_PATH, "eval_agent", args.profile,
                        str(solver.iteration))
    agent.store_to_disk(path=path, file_name="eval_agentSINGLE")
    print(f"[export] eval agent at iteration {solver.iteration} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None,
                    help="default: MCCFR_M1 (2 seats) / MCCFR_3S_M1 (3)")
    ap.add_argument("--n-seats", type=int, default=2, choices=[2, 3])
    ap.add_argument("--milestone", type=int, default=1, choices=[1])
    ap.add_argument("--iters", type=int, required=True,
                    help="target TOTAL iteration count (not additional)")
    ap.add_argument("--k-postflop", type=int, default=50)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every-s", type=int, default=600)
    ap.add_argument("--export-every-iters", type=int, default=200000)
    ap.add_argument("--allow-night", action="store_true")
    args = ap.parse_args()
    if args.profile is None:
        args.profile = "MCCFR_M1" if args.n_seats == 2 else "MCCFR_3S_M1"

    if in_night_window() and not args.allow_night:
        sys.exit("refusing to start inside 23:00-07:00 (Deep CFR's window); "
                 "pass --allow-night to override")

    ckpt_dir = os.path.join(DATA_PATH, "checkpoint", args.profile)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "mccfr_state.npz")

    tree = load_or_build_tree(n_seats=args.n_seats)
    bucketer = M1Bucketer(k_postflop=args.k_postflop, n_rollouts=args.rollouts,
                          rng=np.random.default_rng(args.seed + 1),
                          n_opponents=args.n_seats - 1)

    if os.path.exists(ckpt_path):
        if not args.resume:
            sys.exit(f"{ckpt_path} exists; pass --resume to continue it "
                     "(or move it aside for a fresh start)")
        solver = ESMCCFRSolver.load(ckpt_path, tree, bucketer)
        print(f"[resume] {ckpt_path} at iteration {solver.iteration}")
    else:
        solver = ESMCCFRSolver(tree, bucketer, seed=args.seed)
        print(f"[fresh] profile {args.profile}, K={solver.K}")

    stop = {"flag": False}

    def on_signal(signum, _frame):
        print(f"[signal] {signal.Signals(signum).name}: finishing iteration, "
              "checkpointing, exiting")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    last_ckpt = time.time()
    last_export_iter = solver.iteration
    last_log = time.time()
    last_log_iter = solver.iteration

    while solver.iteration < args.iters and not stop["flag"]:
        chunk = min(CHUNK_ITERS, args.iters - solver.iteration)
        solver.run(chunk, stop_flag=lambda: stop["flag"])

        now = time.time()
        if now - last_log >= 60:
            rate = (solver.iteration - last_log_iter) / (now - last_log)
            hits, misses = bucketer.n_hits, bucketer.n_misses
            hit_rate = hits / (hits + misses) if hits + misses else 0.0
            touched = ", ".join(f"{f:.3f}" for f in solver.fraction_touched())
            print(f"[{solver.iteration}] {rate:.1f} it/s | cache "
                  f"{bucketer._cache_entries} entries, {hit_rate:.2%} hits | "
                  f"touched/street: {touched}", flush=True)
            last_log, last_log_iter = now, solver.iteration

        if now - last_ckpt >= args.checkpoint_every_s:
            solver.save(ckpt_path)
            last_ckpt = now
            print(f"[checkpoint] iteration {solver.iteration}", flush=True)

        if solver.iteration - last_export_iter >= args.export_every_iters:
            export_eval_agent(solver, args)
            last_export_iter = solver.iteration

        if in_night_window() and not args.allow_night:
            print("[night] 23:00 reached: checkpointing and exiting")
            break

    solver.save(ckpt_path)
    export_eval_agent(solver, args)
    print(f"[done] iteration {solver.iteration}, checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
