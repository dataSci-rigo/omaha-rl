"""
Nightly training launcher for FixedLimitOmahaHiLo. Meant to be started by a
systemd timer each night and auto-stopped by systemd's RuntimeMaxSec= after
a fixed window (see the plan's "Night-only training schedule" section) --
this script itself has no time budget logic.

On each run it:
  - Scans for the highest existing checkpoint step across BOTH possible
    profile names (see "Aug 18" note below) and resumes from it. Driver's
    __init__ does NOT auto-resume just because the profile name matches --
    it needs iteration_to_import/name_to_import passed explicitly.
  - Otherwise starts fresh (first night).

Aug 18: found why training looked stalled for days despite genuinely
running fine each cycle -- DriverBase.__init__ has a "safety measure to
avoid overwriting logs when reloading" (PokerRL/rl/base_cls/workers/
DriverBase.py:65-66): whenever name_to_import == t_prof.name (true on every
resume here, since we import the same profile we're training), it silently
appends "_" to t_prof.name for that whole session, so checkpoints save under
"<PROFILE_NAME>_" instead of "<PROFILE_NAME>". Since a fresh TrainingProfile
is built each launch from the bare PROFILE_NAME, this doesn't compound
across cycles -- it just alternates: cycle N saves under the trailing-
underscore name, cycle N+1 (importing FROM that name) doesn't collide so
saves under the bare name, cycle N+2 alternates back, etc. The bug wasn't a
stall at all; every cycle's real progress was just landing in whichever name
the *previous* run's resume-check didn't look at.

checkpoint_freq is kept low (every 10 iterations) so a SIGTERM at the nightly
cutoff loses at most ~20 minutes of progress. It is not lower because each
checkpoint pickles the full pre-allocated reservoir buffers (~0.9GB at
max_buffer_size_adv=1M), which at every iteration cost ~15-20% of wall clock.
"""
import datetime
import json
import os
import re

os.environ["OMP_NUM_THREADS"] = "1"

from PokerRL.game.games import FixedLimitOmahaHiLo
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

PROFILE_NAME = "FLO_HiLo_HU_dense_residual"
DATA_PATH = os.path.expanduser("~/poker_ai_data")
# Records the eval_agent step this session *started* from, i.e. last
# session's ending point -- read by nightly_progress_check.py the next
# morning to compare "tonight" against "last night". Always under the bare
# name regardless of which shadow name this session ends up saving under.
SESSION_START_MARKER = os.path.join(DATA_PATH, "checkpoint", PROFILE_NAME, "session_start_eval_agent_step.txt")
# Hour the nightly timer fires; defines the night boundary for the marker above.
NIGHT_START_HOUR = 23

# The two names a session's data can end up under -- see module docstring.
_CANDIDATE_NAMES = (PROFILE_NAME, PROFILE_NAME + "_")


def _find_latest(base_dir_name, is_valid_step_dir):
    """
    Returns (name, step) with the highest step across both candidate names,
    or (None, None) if neither has anything.
    """
    best_name, best_step = None, None
    for name in _CANDIDATE_NAMES:
        d = os.path.join(DATA_PATH, base_dir_name, name)
        if not os.path.isdir(d):
            continue
        steps = [int(x) for x in os.listdir(d) if re.fullmatch(r"\d+", x) and is_valid_step_dir(d, x)]
        if steps:
            step = max(steps)
            if best_step is None or step > best_step:
                best_name, best_step = name, step
    return best_name, best_step


def find_last_checkpoint():
    return _find_latest("checkpoint", lambda d, x: True)


def find_last_eval_agent_step():
    return _find_latest("eval_agent", lambda d, x: os.path.exists(os.path.join(d, x, "eval_agentSINGLE.pkl")))


def _current_night_started_at(now=None):
    """
    The most recent NIGHT_START_HOUR boundary at or before `now` -- i.e. when the
    night currently in progress began. Before that hour, the night began yesterday.
    """
    now = now or datetime.datetime.now()
    boundary = now.replace(hour=NIGHT_START_HOUR, minute=0, second=0, microsecond=0)
    if now < boundary:
        boundary -= datetime.timedelta(days=1)
    return boundary


def read_session_start_step():
    """Returns the marker's step, or None. Tolerates the old bare-int format."""
    try:
        with open(SESSION_START_MARKER) as f:
            raw = f.read().strip()
    except OSError:
        return None
    try:
        return json.loads(raw)["step"]
    except (ValueError, KeyError, TypeError):
        try:
            return int(raw)  # pre-Aug-20 format
        except ValueError:
            return None


def record_session_start_marker():
    """
    Records the eval_agent step this NIGHT started from, so the morning progress
    check can compare a night's worth of training against its starting point.

    Must write at most once per night. This runs on every process launch, but
    RuntimeMaxSec + Restart=always cycle the process ~16 times a night, so the
    original unconditional write left the marker holding the LAST cycle's start.
    On Aug 20 that made the 07:15 check compare step 350 against step 330 -- a
    28-minute window -- and report "PLATEAU ... the 'stop training' signal you
    asked for" after a night that had actually completed 272 iterations. A
    20-iteration gap is far below the check's resolution, so it would have
    reported PLATEAU essentially unconditionally.
    """
    night_start = _current_night_started_at()
    try:
        written_at = datetime.datetime.fromisoformat(
            json.loads(open(SESSION_START_MARKER).read())["written"])
        if written_at >= night_start:
            return  # already recorded for the night in progress; this is a mid-night restart
    except (OSError, ValueError, KeyError, TypeError):
        pass  # missing, unreadable, or old bare-int format -- (re)write it

    _, step = find_last_eval_agent_step()
    if step is not None:
        os.makedirs(os.path.dirname(SESSION_START_MARKER), exist_ok=True)
        with open(SESSION_START_MARKER, "w") as f:
            json.dump({"step": step, "written": datetime.datetime.now().isoformat()}, f)


if __name__ == '__main__':
    resume_name, last_step = find_last_checkpoint()
    # Snapshot tonight's starting point (= last night's ending point) before
    # doing any new work, so tomorrow's progress check can compare against it.
    record_session_start_marker()

    t_prof_kwargs = dict(
        name=PROFILE_NAME,
        nn_type="dense_residual",

        DISTRIBUTED=True,
        CLUSTER=False,
        # CORRECTION (Aug 20): an earlier comment here claimed 8 workers were
        # "8x-ing memory for zero speed benefit" and cut this 8 -> 2. That was
        # wrong. TrainingProfile.py:264-269 does:
        #     if DISTRIBUTED or CLUSTER: self.n_learner_actors = n_learner_actor_workers
        #     else:                      self.n_learner_actors = 1
        # so with DISTRIBUTED=False this value is DISCARDED and exactly one
        # LearnerActor is ever built (Driver.py:31-36 is its only consumer).
        # It was always 1. Cutting 8 -> 2 saved zero bytes and changed nothing.
        #
        # Memory scales as n_learner_actors * n_seats * max_buffer_size_adv ONLY
        # under DISTRIBUTED=True. Here it is just n_seats * max_buffer_size_adv.
        #
        # This value is live only when DISTRIBUTED=True, where each LA is a real
        # Ray process that generates a FULL n_traversals_per_iter of its own
        # (la/local.py:114 -- the count is per-worker, never divided), so N
        # workers multiply data per iteration, cores used, and memory alike.
        n_learner_actor_workers=4,

        # Each learner-actor keeps its OWN independent reservoir buffer per
        # player (DeepCFR/workers/la/local.py) -- total memory scales as
        # n_learner_actor_workers * n_seats * this value, at ~490 bytes/entry.
        # At 2 workers that is ~2GB here, on top of a ~2.2GB baseline.
        #
        # Aug 20: restored 75,000 -> 1,000,000. The earlier cuts (1M -> 200k ->
        # 75k) were all chasing OOMs that turned out to be per-net LUT
        # duplication, not buffer size (see PokerRL/rl/neural/_shared_luts.py).
        # Shrinking it this far starved learning: each iteration draws
        # mini_batch_size_adv * n_batches_adv_training = 1.5M training samples,
        # so a 75k buffer meant a 20:1 resample ratio, and after 350 iterations
        # (~5.25M entries seen) reservoir acceptance had fallen to ~1.4%, so
        # fresh data barely entered. A full night of 272 iterations produced no
        # measurable improvement (-41.2 mBB/hand [-124.8, +42.4] over 20k hands).
        # 1M brings the resample ratio to 1.5:1 and keeps acceptance ~19%.
        # The repo's own examples use 1M-3M.
        max_buffer_size_adv=1000000,

        # Bounds the Chief's SD-CFR strategy buffer (one net per player per CFR
        # iteration). Left unbounded it grew ~2.4GB per training night forever --
        # the real cause of the memory ceiling, not the advantage buffers.
        # StrategyBuffer.add() reservoir-samples once this is set, keeping a uniform
        # sample of ALL iterations with each net's iteration weight attached, so the
        # played strategy stays a consistent estimator of the full average.
        # ~4.8MB/net live => 2 seats x 500 = ~4.8GB, flat forever.
        # NOTE: requires eval_methods={} -- see the guard in Driver.__init__.
        eval_agent_max_strat_buf_size=500,
        export_each_net=False,
        # Checkpoints pickle the FULL pre-allocated reservoir tensors, so at
        # max_buffer_size_adv=1M each write is ~0.9GB. At freq=1 that cost ~15-20%
        # of wall-clock and forced every buffer page resident. At 10 a SIGTERM at
        # the 07:00 cutoff loses at most ~20min of work.
        # 5, not 10: at ~3.1 min/iteration (4 distributed workers) plus ~1 min
        # startup, iteration 10 lands at ~32 min -- past RuntimeMaxSec. With
        # freq=10 every 30-min cycle died just before its FIRST checkpoint and
        # restarted from zero (caught in the Aug 20 verification hour; a full
        # night would have produced nothing). freq=5 checkpoints at ~16 min,
        # safely inside any cycle; the ~0.9GB pickle every 5 iterations costs
        # ~6% of wall clock.
        checkpoint_freq=5,
        eval_agent_export_freq=10,

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

        game_cls=FixedLimitOmahaHiLo,
        env_bldr_cls=VanillaEnvBuilder,
        n_seats=2,
        start_chips=2000,

        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_SINGLE,
        ),

        use_simplified_headsup_obs=True,
        log_verbose=True,
    )

    if last_step is not None:
        print(f"Resuming '{resume_name}' from checkpoint step {last_step}.")
        ctrl = Driver(t_prof=TrainingProfile(**t_prof_kwargs),
                      eval_methods={},
                      n_iterations=None,
                      iteration_to_import=last_step,
                      name_to_import=resume_name)
    else:
        print(f"No existing checkpoint for '{PROFILE_NAME}' -- starting fresh.")
        ctrl = Driver(t_prof=TrainingProfile(**t_prof_kwargs),
                      eval_methods={},
                      n_iterations=None)

    ctrl.run()
