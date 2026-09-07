# 07 — Run C (cost-gated MinMax)

Source sessions: 18 · Full text: [/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md)

*Session 18 · 2026-09-07*

Run C is the thesis contrast against Run B: **identical cost gate**, different penalty magnitude. Status: **implemented and unit-tested locally; not yet trained on the cluster.** Canonical design notes: [/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md).

## What Run C is for

Stage 5’s three-run design isolates two questions that A alone cannot answer:

| Contrast | Question |
|---|---|
| A vs B | Does *having* a safety signal (cost > 0 → penalty) buy anything on refusal survival and cost? |
| B vs C | Does a *self-calibrating* magnitude buy anything over a fixed −2.0 once the gate is the same? |

Run C shares Run B’s cost-model plumbing and the identical gate (`cost > threshold`). The only deliberate difference is the replacement value when the gate fires.

## Design

Module: `safe_rlhf.algorithms.ppo_cost_minmax`. Trainer subclasses `PPOCostGateTrainer` and overrides `rl_step` only.

| | Run B | Run C |
|---|---|---|
| When `cost > 0` | fixed `--penalty_magnitude` (**−2.0**) | \(R_{\text{unsafe}} = V_{\MIN} - V_{\MAX}\) |
| Trainer base | `PPOCostGateTrainer` | subclasses it; overrides `rl_step` only |

### Knobs locked for a fair B-vs-C comparison

| Knob | Choice | Why |
|---|---|---|
| Detector / gate | `beaver-7b-unified-cost`, `cost > 0` | identical to B |
| Bound source | reward-model end scores (`--bound_source reward`) | Phase 1’s stable default; critic optional after warmup |
| Bound scope | global | category scope deferred; PKU categories not plumbed through the PPO batch |
| Penalty floor | **−50.0** (matches `clip_range_score`) | Phase 1 floored at −2 because Detoxify was bounded; flooring here at −2 would make “do the bounds move past B?” unanswerable |
| Init | \(V_{\MIN} = V_{\MAX} = 0\) | Beaver rewards are unbounded; seeding ±1 invents a Detoxify-shaped scale |

The floor choice is load-bearing for the thesis claim. Phase 1’s −2 floor was mostly arithmetic confirmation under Detoxify ∈ [0,1] (see [/worklog/phase1/02-saturation.md](/worklog/phase1/02-saturation.md)). Here the cost/reward scores are not trapped that way; a −2 floor would *force* Run C to look like Run B and make the contrast unanswerable.

**Bound source = reward (not critic) by default.** Same lesson as Phase 1 Session 10: start from the detector/reward end scores that are stable early, and only fold in critic values after warmup if explicitly requested. That keeps Run C’s bookkeeping comparable to the Phase 1 default rather than inventing a new Path-2 pathology mid-thesis.

**Init at zero, not ±1.** Seeding $V_{\MIN}=-1$, $V_{\MAX}=+1$ would smuggle Detoxify’s centered scale into Beaver’s unbounded scores. Starting both at 0 lets the first observed rewards define the gap honestly.

```mermaid
flowchart LR
  Roll[Rollout] --> Cost[Cost model score]
  Cost --> G{cost > 0?}
  G -->|no| Keep[Keep reward]
  G -->|yes| U["R_unsafe = V_MIN − V_MAX<br/>floor −50"]
  Keep --> PPO[PPO update]
  U --> PPO
  Rew[Reward end scores] --> Bnd[Update V_MIN / V_MAX]
  Bnd --> U
```

## Files

| Path | Role |
|---|---|
| `safe_rlhf/algorithms/ppo_cost_minmax/` | `minmax_state.py`, `trainer.py`, `main.py`, package entrypoints |
| `scripts/stage5-runC-cost-minmax.sbatch` | Matched to Run B except module, output dir, master port, MinMax args |
| `scripts/inspect-runC.sbatch` | Same prompts / seeds / checkpoints as A and B |
| `scripts/test_cost_minmax_state.py` | CPU unit tests: bound update / floor / warmup / `state_dict` (**passed locally**) |

**Logged every step (beyond B):** `train/v_min`, `train/v_max`, `train/r_unsafe`, `train/r_unsafe_raw`, `train/floor_active`. Latest bounds also written to `output_dir/minmax_state.json`.

## What has not happened yet

No cluster training run. Next steps recorded in the worklog:

1. Sync the new module to `~/Safe-RL-MinMax` on the cluster.
2. `sbatch scripts/stage5-runC-cost-minmax.sbatch`.
3. Inspect + cost-rescore against A/B with the same protocol as Sessions 16–17 ([/worklog/phase2/06-run-b.md](/worklog/phase2/06-run-b.md)).

## What to measure once it finishes

| Question | Why it matters |
|---|---|
| Does \(r_{\text{unsafe}}\) go **more negative than −2**, or stick near B? | Tests whether self-calibration exceeds the fixed gate |
| Lock-picking trajectory vs B at 50/250/500/750/950 | Matched qualitative + cost rescoring (full table style of Session 17) |
| Mean `train/cost` drift | Can adaptive magnitude arrest what B could not? |
| Benign reward-hacking vs B | Better / worse / unchanged on statistics-style prompts |
| Do \(V_{\MIN}/V_{\MAX}\) move across the run? | Revisits Phase 1’s saturation finding with an unbounded signal |

Until those numbers exist, **do not** claim MinMax improves safety. Claims stay at A+B: [/worklog/10-claims.md](/worklog/10-claims.md). Open list: [/worklog/11-open-questions.md](/worklog/11-open-questions.md).

## How to read a future Run C result

If Run C finishes, interpret it against the Session 17 protocol — not against Run A alone:

1. Matched lock-picking generations at the same checkpoints, same seeds.
2. Cost rescoring through `beaver-7b-unified-cost` (full table, not a single headline).
3. Separate the probe story from mean `train/cost` drift (Session 17’s “both things remain true”).
4. Check whether `r_unsafe_raw` / `floor_active` show genuine movement past −2, or early freeze à la Phase 1 Session 8.

A prettier refusal on one prompt with a frozen $R_{\text{unsafe}}$ near −2 is *not* evidence that self-calibration earned its keep over B’s fixed penalty.

---

**Prev:** [06](/worklog/phase2/06-run-b.md) · **Up:** [Phase 2 map](/worklog/phase2/README.md) · **Worklog home:** [/worklog/README.md](/worklog/README.md)
