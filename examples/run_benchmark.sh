#!/bin/bash
# Standalone benchmark runner for the trained FLO Hi/Lo agent.
#
#   ./examples/run_benchmark.sh                    # 500 hands vs ABCBot
#   ./examples/run_benchmark.sh --bot bayesian     # 500 hands vs BayesianBot
#   ./examples/run_benchmark.sh --bot all --hands 2000
#   ./examples/run_benchmark.sh --help             # all options
#
# All arguments are passed straight through to eval_agent_vs_bots.py.
#
# Results print to your terminal AND get written to benchmark_results.log
# (in this examples/ directory) so they can be reviewed afterward.
#
# This evaluates the COMPLETE agent -- every historical net snapshot, i.e. the
# exact SD-CFR average. It used to trim to the newest 8 nets because a full
# agent cost ~20.2GB at step 70; that was a bug (per-net float32 copies of the
# range LUTs, ~140MB each), fixed in PokerRL/rl/neural/_shared_luts.py. A full
# 69-net agent now loads in well under 1GB, so there is nothing to trim.
#
# Still runs isolated in its own memory-capped systemd scope. Without isolation
# a runaway triggers the kernel OOM killer against whatever cgroup this happens
# to run under (e.g. a VS Code integrated terminal), which previously took VS
# Code down as collateral damage. Its own scope means an overrun kills only
# this benchmark, cleanly.
#
# NOTE on MemoryHigh: deliberately NOT set, and do not add it. MemoryHigh is a
# *soft* throttle -- once crossed the kernel forces continuous reclaim instead
# of killing. An earlier version set MemoryHigh=18G, below what the untrimmed
# agent then needed, and the run was throttled into a total stall: 17 seconds
# of CPU across 95 minutes of wall clock, blocked in mem_cgroup_handle_over_high
# with 315k throttle events and zero progress. The same trap on the training
# service (MemoryHigh=12G) cost an entire night. With MemorySwapMax=0 there is
# no swap to fall back on, so a too-low MemoryHigh does not degrade performance,
# it halts it. Only MemoryMax (the hard cap) is set: run at full speed, and if
# it genuinely exceeds the cap, die cleanly and alone.
#
# MemoryMax is a ceiling, not a reservation -- it costs nothing when unused, so
# it is left generous.
set -e
cd "$(dirname "$0")/.."

PY=/home/ai1/anaconda3/envs/omaha/bin/python3

# --help should not need a systemd scope or the heavy imports' side effects.
for a in "$@"; do
  if [ "$a" = "--help" ] || [ "$a" = "-h" ]; then
    PYTHONPATH="$(pwd)" "$PY" examples/eval_agent_vs_bots.py --help
    exit 0
  fi
done

LOG=examples/benchmark_results.log

echo "Starting benchmark run at $(date)" | tee "$LOG"
# PYTHONUNBUFFERED=1: stdout is piped to tee (not a TTY), so Python would
# block-buffer it and the progress lines would not reach the log until the
# buffer flushed -- making a working run look identical to a stalled one.
# This bit us earlier debugging the training service; do not remove.
systemd-run --user --scope --collect \
  -p MemoryMax=25G -p MemorySwapMax=0 \
  --setenv=PYTHONPATH="$(pwd)" \
  --setenv=PYTHONUNBUFFERED=1 \
  "$PY" examples/eval_agent_vs_bots.py "$@" 2>&1 | tee -a "$LOG"
echo "Finished at $(date)" | tee -a "$LOG"
