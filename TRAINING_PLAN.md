# Overnight PLO training plan (laptop)

Train the Deep CFR / SD-CFR Pot Limit Omaha agent on this laptop, running only
**11pm–7am nightly** so daytime use is unaffected, resuming automatically from
checkpoints each night.

## Hardware constraints (measured 2026-08-08)

- Ryzen 5 5600H — 6 cores / 12 threads, **no CUDA GPU** (CPU-only torch)
- 6 GB physical RAM; WSL2 capped at **3.8 GB + 8 GB swap** — sufficient, no increase needed
- The stock config in `examples/PLO_training_start.py` (14 ray workers, 3M-entry
  buffers) is sized for a server and would OOM here — it is scaled down, not reused.

## RAM budget (measured, not estimated)

Advantage buffers are preallocated dense tensors at construction
(`DeepCFR/workers/la/buffers/_ReservoirBufferBase.py:24-40`). For heads-up PLO with
simplified obs and the PL_2 bet set: `pub_obs_size=109`, `N_ACTIONS=4` →
**480 bytes/entry**.

| Component | RAM |
|---|---|
| Both players' 1M-entry buffers (preallocated at startup) | 0.96 GB |
| Torch runtime + PLO lookup tables | ~0.7 GB |
| Transient copy during checkpoint pickling | ~1.0 GB |
| **Peak** | **~2.6–3 GB** — fits the 3.8 GB cap with ~1 GB headroom |

Fallback if a night log ever shows an OOM kill: drop `max_buffer_size_adv` to 600k
(~1.7–2 GB peak).

## Time estimate

CPU-only, single process: ~15–30 min/iteration (sample generation dominates; the
pure-Python Omaha hand evaluator is the wild card) → 64 iterations ≈ **16–32 h of
compute ≈ 2–4 nights** at 8 h/night. Uncertainty is ~2–3× until the first night's
log gives real per-iteration timings; refine the forecast after night 1.

## Implementation steps

### 1. Conda env `omaha` (dedicated; ai_10 stays untouched)

```bash
conda create -n omaha python=3.10 -y
~/anaconda3/envs/omaha/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/anaconda3/envs/omaha/bin/pip install gym scipy psutil pytz requests pycrayon
```

No ray: non-distributed mode never imports it (`PokerRL/rl/MaybeRay.py` tunnels all
calls through when `DISTRIBUTED=False`). gym is only used for `from gym import
spaces` (`PokerRL/game/_/rl_env/base/PokerEnv.py:8`), so any version works. CPU
torch wheel (~200 MB), not the CUDA one.

### 2. Patch `PokerRL/_/CrayonWrapper.py`

`DriverBase.__init__` unconditionally constructs `CrayonClient`, which raises if no
crayon/TensorBoard bridge server is listening — training won't start without this
patch. Wrap the client construction in try/except → `self._crayon = None` plus a
warning, and guard the crayon calls in `update_from_log_buffer()` / `export_all()`.
JSON disk logs are kept where possible.

### 3. New `laptop/PLO_laptop_training.py`

Copy of the stock script with a laptop-scaled `TrainingProfile`:

| Parameter | Stock | Laptop | Why |
|---|---|---|---|
| `DISTRIBUTED` / workers | True / 14 | **False / 1** | no ray, single process |
| `max_buffer_size_adv` | 3,000,000 | **1,000,000** | fits 3.8 GB (see budget) |
| `n_traversals_per_iter` | 150,000 | **50,000** | CPU generation time |
| `n_batches_adv_training` | 1,500 | **1,000** | CPU training time |
| `mini_batch_size_adv` | 5,000 | **2,500** | CPU training time |
| `checkpoint_freq` | 9999 | **2** | old checkpoints auto-deleted (`Driver.py:138-143`), disk-safe |
| `eval_agent_export_freq` | 1 | 4 | fewer exports |
| OMP / torch threads | 1 | **5** | stock starves single-process mode |

Unchanged: `nn_type="dense_residual"`, net sizes 192/64/64, lr 0.004, patience 350,
`init_adv_model="last"`, PLO / PL_2 / 2 seats / 10k chips, SD-CFR single eval mode,
`n_iterations=64`. Data goes to `path_data=~/omaha_rl_data`.

**Auto-resume**: scan `~/omaha_rl_data/checkpoint/<name>/` for the highest step dir;
if found, pass `iteration_to_import=<step>, name_to_import=<dirname>` to `Driver`
(pattern from `examples/load_checkpoint.py`; use the dir's basename since the stored
name can differ by a trailing `_`).

### 4. New `laptop/run_night.sh` (what Task Scheduler launches)

- Appends timestamped output to `~/omaha_rl_data/night_logs/<date>.log`
- `flock` lockfile → double-starts are no-ops
- RAM guard: exits with a logged warning if `MemTotal` < 3.5 GB (misconfigured WSL)
- Computes seconds until next 07:00, then
  `timeout --signal=INT <budget>s nice -n 10 <omaha-python> -u laptop/PLO_laptop_training.py`
  — SIGINT at 7am kills the run; the last even-iteration checkpoint survives and the
  next night resumes from it. Exits cleanly once 64 iterations complete.

### 5. Windows scheduled task (current-user, no admin)

Via `powershell.exe Register-ScheduledTask "OmahaRL Nightly Training"`:
daily 23:00 trigger, action `wsl.exe -d <distro> -- .../laptop/run_night.sh`,
settings `-WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
-ExecutionTimeLimit 9h`.

**User notes:** keep the laptop plugged in overnight; sleep is fine (WakeToRun),
power-off is not.

## Verification

1. **Smoke test** (~2–5 min, low RAM): run the training script with a throwaway
   name, `n_traversals_per_iter=50`, `n_batches_adv_training=20`, 2 iterations,
   `path_data` in a temp dir — proves imports, the CrayonWrapper patch, the PLO
   env, checkpoint write, and resume.
2. Run `run_night.sh` manually once and confirm it starts (or cleanly guards).
3. `schtasks /Query /TN "OmahaRL Nightly Training"` confirms registration.
4. After night 1: read `~/omaha_rl_data/night_logs/` for real per-iteration
   timings → refine the 2–4 night forecast.

## Out of scope

- LBR / head-to-head evaluation (run after training via `examples/eval_agent_lbr.py`
  and `examples/interactive_agent_v_agent.py`)
- ray-distributed mode (version pinning pain and RAM overhead for ~2× generation
  speedup; revisit only if nightly wall time proves too slow)
