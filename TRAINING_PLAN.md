# Nightly FLO Hi/Lo training plan (desktop)

Train the Deep CFR / SD-CFR **Fixed-Limit Omaha Hi/Lo (8-or-better)** heads-up
agent on the desktop, running only **11pm–7am nightly** so daytime use is
unaffected, resuming automatically from checkpoints each night.

The laptop's PLO plan is in `TRAINING_PLAN_LAPTOP.md`. This box is the primary
trainer; the custom O8 game variant (`FixedLimitOmahaHiLo` in
`PokerRL/game/games.py`, `nit`-based hi/lo evaluator, split-pot payout) lives in
this repo and is tested (`test/game/test_OmahaHiLoEval.py`,
`test_OmahaHiLoPayout.py`).

## Hardware (measured)

- 8 cores, 30 GB RAM, Ubuntu (native, systemd)
- Quadro P4000 present but **unusable**: CUDA capability sm_61, installed torch
  supports sm_75+ → effectively CPU-only. Do not chase the GPU.
- Conda env `omaha` (`/home/ai1/anaconda3/envs/omaha/bin/python3`), ray 2.56
- Data root `~/poker_ai_data`

## Current lineage — restarted 2026-08-20, do not resurrect older checkpoints

Everything trained before 2026-08-20 was **card-blind**: the fork's PLO LUT
builder used `np.empty` and left the private-card observation rows mostly
garbage (113 distinct rows for 270,725 hands postflop; **2** preflop). Those
agents beat their own snapshots while losing ~500 mBB/hand to a simple rule
bot. Fixed (np.zeros + true suit-isomorphic preflop bucketing, 16,432 classes);
old artifacts archived in `~/poker_ai_data/archive_cardblind_20260820_1649/`.
Anything found there is for archaeology only.

## Schedule (systemd user units; linger is enabled)

| Unit | When | What |
|---|---|---|
| `flo-hilo-training.timer` → `.service` | 23:00 | starts `examples/FLO_HiLo_nightly_run.py` |
| (service) `RuntimeMaxSec=1800`, `Restart=always` | every 30 min | cycles the process; resume loses ≤ `checkpoint_freq` iterations |
| `flo-hilo-training-stop.timer` | 07:00 | outer stop (needed because `Restart=always`) |
| `flo-hilo-progress-check.timer` | 07:15 | `nightly_progress_check.py`: tonight vs the night's start marker, 20k hands |

**Never run training manually during the day without asking.** Manual runs are
fine when requested; stop them with
`systemctl --user stop flo-hilo-training.service`.

## Config (`examples/FLO_HiLo_nightly_run.py`) — every value is load-bearing

| Parameter | Value | Why |
|---|---|---|
| `DISTRIBUTED` / workers | **True / 4** | 4 real ray processes, ~3.5 cores. With `DISTRIBUTED=False` the worker count is silently forced to 1 (`TrainingProfile.py:264-269`) |
| `n_traversals_per_iter` | 15,000 **per worker** | count is per-LA, never divided → 60k fresh entries/iteration |
| `max_buffer_size_adv` | 1,000,000 | 472 B/entry/seat. At 75k the 1.5M draws/iteration meant 20:1 resampling — a full 272-iteration night produced **zero** measurable gain; at 1M, 30 iterations produced +150.7 mBB/hand. Grow-only on resume (migration in `_ReservoirBufferBase.load_state_dict`) |
| `eval_agent_max_strat_buf_size` | 500 | bounds the Chief's SD-CFR net history via reservoir sampling (consistent estimator). Unbounded it grew ~2.4 GB/night forever. Requires `eval_methods={}` (guarded in `Driver`) |
| `checkpoint_freq` | 10 | each checkpoint pickles the full 1M buffers (~0.9 GB); at freq=1 that cost 15–20% of wall clock |
| `n_batches_adv_training` × `mini_batch_size_adv` | 750 × 2,000 | 1.5M draws/iteration |
| net | dense_residual 192/64/64, lr 0.004 | matches the repo's reference configs |

## Memory budget (measured, not estimated)

Steady state ≈ **8–12 GB**: ~3.8 GB reservoir buffers (4 LA × 2 seats × 1M ×
472 B) + bounded Chief (≤ 2×500 nets ≈ 4.8 GB at saturation, months away) +
LUTs/nets/ray overhead. Service cap `MemoryMax=20G`, `MemorySwapMax=0`.

**Never set `MemoryHigh`.** It is a soft throttle: crossing it puts the cgroup
into continuous reclaim instead of failing — with swap off it does not degrade,
it *halts* (117 s CPU across a 27-min run; one full night lost this way, plus a
95-minute benchmark hang). `MemoryMax` either fits or dies cleanly, and
`Restart=always` + checkpoint/resume recovers a clean kill.

If usage ever climbs toward the cap again, that is a leak to diagnose, not a
cap to raise. The two growth bugs found so far: per-net float32 LUT copies
(~140 MB/net, fixed via `PokerRL/rl/neural/_shared_luts.py`) and the unbounded
Chief buffer (fixed via the bound above).

## Throughput (measured)

- ~**3 min/iteration** at 4 workers (single-core was 1.85 — 4× data for 1.6×
  time, ~2.5× net) → ~**160 iterations/night**
- Fresh start pays ~40 s of LUT construction per worker process

## Known repo pitfalls

- **Shadow name:** resuming with `name_to_import == t_prof.name` makes
  `DriverBase` append `_`; artifacts alternate between
  `FLO_HiLo_HU_dense_residual` and `..._` nightly. All lookups must scan both
  (`_CANDIDATE_NAMES` pattern).
- **`AgentTournament.run()` returns `(mean, UPPER, LOWER)`** — not
  (mean, lower, upper). Already bit us once.
- **Exact BR and LBR do not work for this variant** (Leduc-only tree code;
  `get_hand_rank_all_hands_on_given_boards` raises). Evaluation is
  head-to-head only.
- The session-start marker (`session_start_eval_agent_step.txt`, JSON) is
  written at most once per night — the 30-min restart cycle used to overwrite
  it 16×/night, making the morning check compare a 28-minute window.

## Morning routine

```bash
journalctl --user -u flo-hilo-progress-check.service --since 07:00 | tail -20
./examples/run_benchmark.sh --bot all --hands 5000     # vs ABCBot, BayesianBot, step-0
```

20,000 hands ≈ 3 min ≈ ±91 mBB/hand CI; 5,000 ≈ ±180. The two external bots
come from `~/Documents/omaha`; BayesianBot beats ABCBot by ~300–400 mBB/hand
(stable baseline — if a run shows otherwise, suspect the harness first).

## Hyperparameter search

`examples/hp_search.py` — forks trials from the newest checkpoint+export pair,
runs N extra iterations per config in a memory-capped scope, scores each vs the
frozen fork-point agent (fixed opponent ⇒ total order, no h2h intransitivity),
appends to `~/poker_ai_hpsearch/<run>/results.jsonl`, tears down. Round-1 grid
targets data volume (`n_traversals_per_iter` 15k→150k) — both the buffer result
and the laptop plan's 50k setting say our fresh-data budget is the binding
constraint. Full grid ≈ 6 h: start in the morning so it clears the 23:00 timer
(it refuses to start while training is active, but will not stop itself).

## Roadmap

1. Nights of HU training + HP search until the agent beats ABCBot, then
   BayesianBot (the Phase-2 bar from the original plan).
2. Long-term goal is a **6-max table** (~10× HU cost; needs an N-seat
   tournament — `AgentTournament_hu` is HU-only). Multiway engine + hi/lo
   split-pot payout are already tested. Only one net layer (~1% of params)
   depends on seat count, so the card tower warm-starts; 3-player
   (~1.5–2× HU) is the natural stepping stone.
