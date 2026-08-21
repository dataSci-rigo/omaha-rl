"""
Automated hyperparameter search for FLO Hi/Lo Deep CFR training.

Each trial FORKS from a shared checkpoint (so it starts from real training
progress instead of iteration 0), runs exactly --iters more CFR iterations with
one config override, exports its agent, and is scored head-to-head against the
FROZEN fork-point agent over --hands hands. Scoring every trial against the same
fixed opponent yields one scalar per trial and therefore a total order --
sidestepping head-to-head non-transitivity (we measured an apparent A>B>C>A
cycle once; cycles only arise when trials play each other).

Results append to results.jsonl (one row per trial: config, mBB/hand, 95% CI,
wall clock). Trials are sequential -- memory forbids concurrency -- and each
trial's data dir is deleted after scoring except the exported agent.

Run:
    python3 examples/hp_search.py                       # full round-1 grid
    python3 examples/hp_search.py --trials control n_traversals_30k
    python3 examples/hp_search.py --iters 15 --hands 20000
    python3 examples/hp_search.py --list                # show grid and exit

The fork checkpoint is discovered automatically: the highest checkpoint step
(across both shadow profile names) that also has an eval_agent export at the
same step. Both exist at multiples of 10 with the nightly config.

Internal subcommands (used by the orchestrator, not by hand):
    --run-trial JSON     train one trial in this process (called inside a
                         systemd-run memory scope)
    --score NEW REF N    play NEW vs REF for N hands, print one JSON line
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

PROFILE_NAME = "FLO_HiLo_HU_dense_residual"
# HP_DATA_PATH override: lets tests point fork discovery at an archived data
# tree without symlinking anything into the live one (where the nightly resume
# scan could pick it up).
DATA_PATH = os.path.expanduser(os.environ.get("HP_DATA_PATH", "~/poker_ai_data"))
SEARCH_ROOT = os.path.expanduser("~/poker_ai_hpsearch")
_CANDIDATE_NAMES = (PROFILE_NAME, PROFILE_NAME + "_")

# Must mirror examples/FLO_HiLo_nightly_run.py -- trials fork from ITS
# checkpoints, and any silent divergence here (worker count, buffer size,
# net shape) makes the fork unloadable or the comparison meaningless.
BASE_KWARGS = dict(
    name=None,  # set per trial
    nn_type="dense_residual",
    DISTRIBUTED=True,
    CLUSTER=False,
    n_learner_actor_workers=4,
    max_buffer_size_adv=1000000,
    eval_agent_max_strat_buf_size=500,
    export_each_net=False,
    checkpoint_freq=10 ** 9,          # trials are disposable: never checkpoint
    eval_agent_export_freq=10 ** 9,   # exported once, explicitly, at the end
    n_actions_traverser_samples=3,
    n_traversals_per_iter=15000,
    n_batches_adv_training=750,
    max_n_las_sync_simultaneously=4,
    use_pre_layers_adv=True,
    n_cards_state_units_adv=192,
    n_merge_and_table_layer_units_adv=64,
    n_units_final_adv=64,
    lr_patience_adv=100,
    lr_adv=0.004,
    mini_batch_size_adv=2000,
    init_adv_model="last",
    n_seats=2,
    start_chips=2000,
    use_simplified_headsup_obs=True,
    log_verbose=False,
)

# Round-1 grid: fork-safe axes only, one change per trial, plus the control
# TWICE -- the pair calibrates the noise floor and catches harness bugs (their
# scores must agree within CI, or state is leaking between trials).
GRID = {
    "control": {},
    "control_repeat": {},
    "n_traversals_30k": {"n_traversals_per_iter": 30000},
    "n_traversals_60k": {"n_traversals_per_iter": 60000},
    "n_traversals_150k": {"n_traversals_per_iter": 150000},
    "n_batches_1500": {"n_batches_adv_training": 1500},
    "mini_batch_4000": {"mini_batch_size_adv": 4000},
    "action_samples_4": {"n_actions_traverser_samples": 4},
}


def find_fork_point():
    """(name, step) of the newest checkpoint that also has an eval_agent export
    at the same step under the same name. Returns (None, None) if none."""
    import re
    best = (None, None)
    for name in _CANDIDATE_NAMES:
        ckpt_dir = os.path.join(DATA_PATH, "checkpoint", name)
        if not os.path.isdir(ckpt_dir):
            continue
        for d in os.listdir(ckpt_dir):
            if not re.fullmatch(r"\d+", d):
                continue
            step = int(d)
            export = os.path.join(DATA_PATH, "eval_agent", name, d, "eval_agentSINGLE.pkl")
            if os.path.exists(export) and (best[1] is None or step > best[1]):
                best = (name, step)
    return best


def fork_checkpoint_dir(name, step):
    return os.path.join(DATA_PATH, "checkpoint", name, str(step))


def reference_agent_path(name, step):
    return os.path.join(DATA_PATH, "eval_agent", name, str(step), "eval_agentSINGLE.pkl")


# ------------------------------------------------------------------ trial worker

def run_trial_worker(spec):
    """Runs inside its own process (and systemd scope). Trains one trial and
    exports its agent; prints the export path on the last line."""
    from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
    from DeepCFR.TrainingProfile import TrainingProfile
    from DeepCFR.workers.driver.Driver import Driver
    from PokerRL.game.games import FixedLimitOmahaHiLo
    from PokerRL.game.wrappers import VanillaEnvBuilder

    trial_id = spec["trial_id"]
    path_data = spec["path_data"]
    fork_name, fork_step = spec["fork_name"], spec["fork_step"]

    # Symlink the shared fork checkpoint into this trial's tree (3.9GB -- never
    # copy). The trial writes only under its own name, and DriverBase's
    # cleanup only touches <path_checkpoint>/<t_prof.name>/, so the source
    # stays untouched (verified by md5 in the orchestrator).
    link_parent = os.path.join(path_data, "checkpoint", fork_name)
    os.makedirs(link_parent, exist_ok=True)
    link = os.path.join(link_parent, str(fork_step))
    if not os.path.exists(link):
        os.symlink(fork_checkpoint_dir(fork_name, fork_step), link)

    kwargs = dict(BASE_KWARGS)
    kwargs.update(spec["overrides"])
    kwargs["name"] = trial_id          # != fork_name => no shadow "_" rename
    kwargs["path_data"] = path_data
    kwargs["game_cls"] = FixedLimitOmahaHiLo
    kwargs["env_bldr_cls"] = VanillaEnvBuilder
    kwargs["eval_modes_of_algo"] = (EvalAgentDeepCFR.EVAL_MODE_SINGLE,)

    ctrl = Driver(t_prof=TrainingProfile(**kwargs),
                  eval_methods={},
                  n_iterations=spec["n_iterations"],   # exactly N more iterations
                  iteration_to_import=fork_step,
                  name_to_import=fork_name)
    ctrl.run()
    ctrl.export_eval_agent()   # single explicit export at the final iteration

    # Don't compute the step -- scan for it. _cfr_iter starts at fork_step + 1
    # and increments AFTER each iteration, so the export lands at
    # fork_step + n_iterations + 1; guessing invites off-by-ones (it caused one).
    import re
    export_root = os.path.join(path_data, "eval_agent", trial_id)
    steps = [int(d) for d in os.listdir(export_root) if re.fullmatch(r"\d+", d)
             and os.path.exists(os.path.join(export_root, d, "eval_agentSINGLE.pkl"))]
    export = os.path.join(export_root, str(max(steps)), "eval_agentSINGLE.pkl")
    print(f"TRIAL_EXPORT={export}")


# ------------------------------------------------------------------ scoring

def run_score_worker(new_path, ref_path, n_hands):
    """Plays NEW vs REF, prints one JSON line: mean/lower/upper mBB per hand
    from NEW's perspective."""
    from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
    from PokerRL.game.AgentTournament_hu import AgentTournament

    new_agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=new_path)
    ref_agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=ref_path)
    tourney = AgentTournament(env_cls=new_agent.env_bldr.env_cls,
                              env_args=new_agent.env_bldr.env_args,
                              eval_agent_1=new_agent,
                              eval_agent_2=ref_agent,
                              logfile=None)
    # NOTE: returns (mean, UPPER, LOWER) -- not (mean, lower, upper).
    mean, upper, lower = tourney.run(n_games_per_seat=n_hands // 2)
    print("SCORE_JSON=" + json.dumps({"mean_mbb": mean, "lower95": lower, "upper95": upper}))


# ------------------------------------------------------------------ orchestrator

def _run_in_scope(desc, argv, mem_max="20G"):
    """Runs argv inside a transient memory-capped scope so an overrun kills only
    the trial. MemoryMax only -- MemoryHigh is a soft throttle that stalls
    rather than fails (it has cost a full night and a 95-min benchmark hang)."""
    cmd = ["systemd-run", "--user", "--scope", "--collect", "-q",
           f"-p", f"MemoryMax={mem_max}", "-p", "MemorySwapMax=0",
           f"--setenv=PYTHONPATH={REPO}", "--setenv=PYTHONUNBUFFERED=1"]
    if "HP_DATA_PATH" in os.environ:  # propagate test override into workers
        cmd.append(f"--setenv=HP_DATA_PATH={os.environ['HP_DATA_PATH']}")
    cmd += argv
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def _md5_dir(path):
    out = subprocess.run(
        ["bash", "-c", f"cd '{path}' && md5sum $(ls | sort) | md5sum"],
        capture_output=True, text=True)
    return out.stdout.strip()


def orchestrate(args):
    fork_name, fork_step = find_fork_point()
    if fork_step is None:
        raise SystemExit(
            "No fork point found: need a checkpoint AND an eval_agent export at the "
            f"same step under {DATA_PATH}. Run training first (both land at "
            "multiples of 10 with the nightly config).")
    ref_path = reference_agent_path(fork_name, fork_step)
    fork_dir = fork_checkpoint_dir(fork_name, fork_step)
    fork_md5 = _md5_dir(fork_dir)

    if subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                       "flo-hilo-training.service"]).returncode == 0:
        raise SystemExit("flo-hilo-training.service is running -- a trial would "
                         "compete for memory/CPU with it. Stop it first.")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SEARCH_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")

    trials = args.trials or list(GRID.keys())
    unknown = [t for t in trials if t not in GRID]
    if unknown:
        raise SystemExit(f"unknown trial(s): {unknown}; available: {list(GRID)}")

    print(f"Fork point: {fork_name} step {fork_step}")
    print(f"Reference agent: {ref_path}")
    print(f"Run dir: {run_dir}")
    print(f"Trials ({args.iters} iters, {args.hands} hands each): {trials}\n")

    for trial_id in trials:
        overrides = GRID[trial_id]
        path_data = os.path.join(run_dir, trial_id)
        spec = {"trial_id": trial_id, "overrides": overrides, "path_data": path_data,
                "fork_name": fork_name, "fork_step": fork_step,
                "n_iterations": args.iters}
        row = {"trial": trial_id, "overrides": overrides, "iters": args.iters,
               "hands": args.hands, "fork_step": fork_step, "started": time.strftime("%F %T")}

        print(f"=== {trial_id}  overrides={overrides} ===")
        t0 = time.time()
        r = _run_in_scope(trial_id, [PY, os.path.abspath(__file__),
                                     "--run-trial", json.dumps(spec)])
        row["train_seconds"] = round(time.time() - t0, 1)

        export = None
        for line in r.stdout.splitlines():
            if line.startswith("TRIAL_EXPORT="):
                export = line.split("=", 1)[1]
        if r.returncode != 0 or export is None or not os.path.exists(export or ""):
            row["error"] = (r.stderr or r.stdout)[-2000:]
            print(f"  TRAIN FAILED (rc={r.returncode}) -- see results.jsonl")
        else:
            # isolation check: the shared fork checkpoint must be byte-identical
            if _md5_dir(fork_dir) != fork_md5:
                row["error"] = "FORK CHECKPOINT MUTATED -- harness bug, aborting"
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                raise SystemExit(row["error"])
            trial_ckpt = os.path.join(path_data, "checkpoint", trial_id)
            row["wrote_own_checkpoints"] = os.path.isdir(trial_ckpt) and bool(os.listdir(trial_ckpt))

            t0 = time.time()
            s = _run_in_scope(trial_id + "_score",
                              [PY, os.path.abspath(__file__),
                               "--score", export, ref_path, str(args.hands)])
            row["score_seconds"] = round(time.time() - t0, 1)
            for line in s.stdout.splitlines():
                if line.startswith("SCORE_JSON="):
                    row.update(json.loads(line.split("=", 1)[1]))
            if "mean_mbb" not in row:
                row["error"] = "score failed: " + (s.stderr or s.stdout)[-2000:]
                print("  SCORE FAILED -- see results.jsonl")
            else:
                print(f"  {row['mean_mbb']:+.1f} mBB/hand "
                      f"[{row['lower95']:+.1f}, {row['upper95']:+.1f}]  "
                      f"(train {row['train_seconds']}s, score {row['score_seconds']}s)")

            # keep only the exported agent; delete buffers/checkpoints/symlink
            keep = os.path.join(run_dir, "agents", trial_id + ".pkl")
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            if export and os.path.exists(export):
                shutil.copy(export, keep)
                row["agent"] = keep

        shutil.rmtree(path_data, ignore_errors=True)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. Results: {results_path}")
    _print_leaderboard(results_path)


def _print_leaderboard(results_path):
    rows = [json.loads(l) for l in open(results_path)]
    scored = [r for r in rows if "mean_mbb" in r]
    scored.sort(key=lambda r: r["mean_mbb"], reverse=True)
    print("\nLeaderboard (vs fork-point agent):")
    for r in scored:
        sig = "+" if r["lower95"] > 0 else ("-" if r["upper95"] < 0 else " ")
        print(f"  [{sig}] {r['trial']:<22} {r['mean_mbb']:+8.1f} "
              f"[{r['lower95']:+7.1f}, {r['upper95']:+7.1f}]  {r['train_seconds']:.0f}s train")
    for r in rows:
        if "error" in r:
            print(f"  [!] {r['trial']:<22} FAILED")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--iters", type=int, default=15,
                    help="CFR iterations per trial beyond the fork point (default 15)")
    ap.add_argument("--hands", type=int, default=20000,
                    help="hands per trial for scoring vs the fork-point agent (default 20000)")
    ap.add_argument("--trials", nargs="*", default=None,
                    help=f"subset of trials to run (default: all). Available: {list(GRID)}")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    ap.add_argument("--run-trial", metavar="JSON", help=argparse.SUPPRESS)
    ap.add_argument("--score", nargs=3, metavar=("NEW", "REF", "HANDS"),
                    help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.list:
        for k, v in GRID.items():
            print(f"{k:<22} {v or '(baseline config)'}")
    elif a.run_trial:
        run_trial_worker(json.loads(a.run_trial))
    elif a.score:
        run_score_worker(a.score[0], a.score[1], int(a.score[2]))
    else:
        orchestrate(a)
