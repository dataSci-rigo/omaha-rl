#!/usr/bin/env bash
# Nightly training window for omaha-rl. Launched by Windows Task Scheduler at 23:00
# (see ../TRAINING_PLAN.md). Safe to run manually; exits if a run is already active.

LOG_DIR="$HOME/omaha_rl_data/night_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$(date +%F).log"
exec >>"$LOG" 2>&1

echo "=== run_night start $(date) ==="

exec 9>/tmp/omaha_rl_training.lock
if ! flock -n 9; then
    echo "another training run holds the lock; exiting"
    exit 0
fi

mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
if [ "$mem_kb" -lt 3500000 ]; then
    echo "WSL MemTotal is only $((mem_kb / 1024)) MB (<3500 MB); check WSL memory config. Skipping."
    exit 1
fi

# Run until the next 07:00.
if [ "$(date +%H)" -lt 7 ]; then
    stop=$(date -d "07:00" +%s)
else
    stop=$(date -d "tomorrow 07:00" +%s)
fi
budget=$(( stop - $(date +%s) ))
if [ "$budget" -lt 600 ]; then
    echo "less than 10 minutes until 07:00; not starting"
    exit 0
fi

echo "training for up to $((budget / 60)) minutes"
cd "$HOME/code20/omaha-rl" || exit 1

# Restart after OOM kills (rc=137): training resumes from its latest complete
# checkpoint, so a kill costs at most checkpoint_freq iterations, not the night.
while :; do
    left=$(( stop - $(date +%s) ))
    if [ "$left" -lt 600 ]; then
        echo "under 10 minutes left before 07:00; stopping for tonight"
        rc=0
        break
    fi
    timeout --signal=INT --kill-after=120 "$left" \
        nice -n 10 "$HOME/anaconda3/envs/omaha/bin/python" -u laptop/PLO_laptop_training.py \
        | awk '{ print strftime("[%F %T]"), $0; fflush() }'
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 137 ]; then
        echo "training was killed (rc=137, likely OOM); restarting from last checkpoint in 60s"
        sleep 60
        continue
    fi
    break
done
echo "=== run_night end $(date) rc=$rc (124 means stopped at 07:00, resumes next night) ==="
