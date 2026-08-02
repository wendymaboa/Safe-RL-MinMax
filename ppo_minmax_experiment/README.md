# PPO + Minmax Penalty Experiment

ROSARL Minmax penalty (`R_unsafe = V_MIN − V_MAX`) adapted for PPO fine-tuning of GPT-2 with Detoxify as the unsafe-state detector.

## Layout

```
ppo_minmax_experiment/
  Train/
    train_baseline.py    # PPO + KL (control)
    train_minmax.py      # PPO + Minmax penalty
    minmax_penalty.py    # Algorithm 1 state
    reward_utils.py      # Centered Detoxify rewards + BeaverTails loading
    generation_utils.py  # Shared TRL generation helpers
    ppo_defaults.py       # Shared PPO hyperparameters
    checkpoint_utils.py   # Save/resume checkpoints and penalty-state persistence
  Evaluation/
    eval_beavertails.py  # Held-out eval (700 prompts)
    plot_results.py      # Training / harm-rate figures
  run_experiment.py      # Train both conditions + evaluate
  trl_compat.py          # TRL version shim
```

`safe-rlhf/` at the repo root is the PKU Safe RLHF framework — left untouched.

## Quick start

```bash
# Smoke test first — 10 steps, catches breakage before a long run
python run_experiment.py --smoke-test --seed 42

# Full run (baseline → minmax → eval)
python run_experiment.py --steps 1000 --seed 42

# Generate figures once the run above finishes
python Evaluation/plot_results.py --experiment-dir experiments/seed_42_steps_1000

# Individual scripts, if you want to run one condition on its own
python Train/train_baseline.py --steps 1000 --seed 42
python Train/train_minmax.py   --steps 1000 --seed 42
```

## Design

Both conditions share:

- BeaverTails `30k_train`, shuffled with `--seed`
- Centered reward: `1 − 2×toxicity` (range [−1, 1])
- Same batch size, PPO epochs (`ppo_epochs=1` — see "Fixes and open questions" below), and generation config

**KL coefficient is currently asymmetric, not shared: baseline uses β=0.2, Minmax uses β=0.01** (`DEFAULT_KL_COEF_BASELINE` / `DEFAULT_KL_COEF_MINMAX` in `ppo_defaults.py`). The reasoning: with a strong KL anchor *and* the Minmax penalty both pulling the policy at once, the two mechanisms compete and neither signal comes through cleanly, so Minmax runs with a weaker anchor to give the penalty room to act. **This is a hypothesis, not yet an isolated, confirmed result** — see the open questions section below.

The only other training difference: `train_minmax.py` replaces rewards on unsafe rollouts (centered reward < 0.4, i.e. toxicity > 0.3) with `R_unsafe`, using per-category `V_min`/`V_max` bounds (see below) and a floor of −2.0.

### Bounds tracking (`MinmaxPenaltyState`)

- **`bound_scope="category"` (default):** `V_min`/`V_max` are tracked separately per BeaverTails harm category, not as one global pair. Rationale: unlike ROSARL's original robotics setting, different prompts have very different achievable value ranges, so mixing them into one global bound was considered likely to blur the signal. This has not yet been evaluated against `bound_scope="global"` head to head.
- **`bound_source="reward"` (default):** bounds are updated from the Detoxify-derived reward only. The critic's value estimates are **not** folded into the bounds unless `bound_source="reward_and_value"` is passed, and even then only after `critic_warmup_steps` (default 100) have elapsed — an uncalibrated GPT-2 value head otherwise saturates the bounds almost immediately.
- Whichever `bound_source`/`bound_scope` combination is active, this bookkeeping is entirely separate from TRL's own internal PPO training — see "Path 1 vs Path 2" below.

## Evaluation

`eval_beavertails.py` scores each trained model on the held-out BeaverTails-Evaluation set (700 prompts, greedy decoding by default).

- **Empty responses are always excluded from toxicity scoring** — they contribute to `empty_rate` but never to `harm_rate_nonempty`. There is no fallback that scores the prompt instead; an earlier version did that and it inflated harm rate, so it was removed rather than left as an opt-in.
- **Categories are multi-label** for prompts loaded from the `30k_test` fallback split, but **single-label** for the real `BeaverTails-Evaluation` split — these are different dataset formats. If `BeaverTails-Evaluation` fails to load, the script falls back to `30k_test`, prints a loud warning, and tags the run so it can't be silently compared against a real-`BeaverTails-Evaluation` run later.
- Two harm-rate columns are reported: `harm_rate_all` (denominator = every prompt) and `harm_rate_nonempty` (denominator = non-empty responses only). Prefer `harm_rate_nonempty` when comparing conditions.

```bash
# Evaluate both conditions (default)
python Evaluation/eval_beavertails.py --steps 1000 \
    --baseline-model experiments/seed_42_steps_1000/checkpoints/baseline \
    --minmax-model experiments/seed_42_steps_1000/checkpoints/minmax

# Evaluate one condition only
python Evaluation/eval_beavertails.py --steps 1000 --models minmax \
    --minmax-model experiments/seed_42_steps_1000/checkpoints/minmax
```

Key flags: `--models` (which condition(s) to run), `--do-sample` + `--seed` (sampling vs. greedy), `--harm-threshold`, `--eval-limit`, `--run-tag` (disambiguates reruns of the same `--steps`).

## Plots

`plot_results.py` generates, from a completed experiment directory:

- `fig1_training_curves` — mean reward, baseline vs Minmax
- `fig2_algorithm1_dynamics` — V_MIN, V_MAX, R_unsafe over training (Minmax only)
- `fig3`/`fig4`/`fig5` — harm rate by category, overall harm rate, harm vs empty-response rate
- `fig6_minmax_diagnostics` — toxicity, n_unsafe, floor_active over training (Minmax only)
- **`fig7_baseline_vs_minmax_diagnostics`** (new) — entropy, value loss, policy loss, and clip fraction, baseline vs Minmax side by side. This is the panel that shows whether Minmax is destabilizing PPO training relative to baseline, rather than looking at each condition's numbers in isolation. **Requires both training logs to include the `entropy`/`value_loss`/`policy_loss`/`clipfrac` columns** — logs from before this field was added will cause this one figure to be skipped (with a printed note), not a crash.

```bash
python Evaluation/plot_results.py --experiment-dir experiments/seed_42_steps_1000
```

Fair comparison (baseline sample eval + minmax β=0.2 sample eval):

```bash
python Evaluation/plot_results.py --experiment-dir experiments/seed_42_steps_1000 \
  --minmax-log experiments/seed_42_steps_1000/logs/minmax_training.csv \
  --eval-summary experiments/seed_42_steps_1000/results/eval_1000_seed42_sample_summary.csv \
  --eval-minmax-summary experiments/seed_42_steps_1000/results/eval_1000_kl02_sample_summary.csv \
  --figure-suffix _kl02_fair --minmax-label "PPO + Minmax (beta=0.2)"



# to regerate
conda activate ppo-minmax
python Evaluation/plot_results.py --steps 1000 \
  --output-dir experiments/seed_42_steps_1000/results/figures \
  --baseline-log experiments/seed_42_steps_1000/logs/baseline_training.csv \
  --minmax-log experiments/seed_42_steps_1000/logs/minmax_training.csv \
  --eval-summary experiments/seed_42_steps_1000/results/eval_1000_seed42_sample_summary.csv \
  --eval-minmax-summary experiments/seed_42_steps_1000/results/eval_1000_kl02_sample_summary.csv \
  --figure-suffix _kl02_fair --minmax-label "PPO + Minmax (beta=0.2)"
```

Figures are written to `experiments/seed_42_steps_1000/results/figures/` with the suffix in the filename (e.g. `fig7_baseline_vs_minmax_diagnostics_1000_kl02_fair.png`).

## Fixes and open questions (chronological, latest first)

- **Path 1 vs Path 2:** the critic's value estimate is used in two places that never share data. TRL's `trainer.step()` recomputes its own values internally for the actual PPO training update (policy gradient + critic regression) — standard, unmodified TRL code. Separately, `get_value_estimates()` is called by this codebase purely to feed `MinmaxPenaltyState`'s bounds tracking. Every saturation/instability finding to date has lived in the second path only; the first has not been modified and should be treated as trustworthy standard PPO.
- **Reward-scale saturation:** with reward bounded to `[−1, 1]`, `V_MIN − V_MAX` is mathematically capped at −2.0 — the −2.0 floor mostly just confirms that mathematical minimum rather than doing independent clipping work. This caused `R_unsafe` to saturate and freeze within a few hundred steps in earlier global-bounds runs. `bound_scope="category"` and `bound_source="reward"` (both now default) are mitigations for this, not yet confirmed to fully resolve it — the next run's `v_min`/`v_max`/`r_unsafe_raw` columns should be checked for renewed movement across the full run length, not just the first hundred steps.
- **Empty-response eval bug:** an earlier eval pipeline scored empty responses as if they were the prompt, inflating harm rate (91–94% empty in some old runs). Fixed — see Evaluation section above.
- **KL asymmetry:** see Design section above — a live hypothesis, worth an isolated ablation (Minmax at β=0.2, matching baseline) before leaning on it as an explanation in the thesis.
- **PPO epochs 4→1, `adap_kl_ctrl=False`:** fixes an earlier issue where PPO's average-ratio-over-threshold warning caused mini-batch updates to be skipped, and KL divergence occasionally went deeply negative and spiralled. Both conditions must use the same value here (via `ppo_defaults.py`) or the comparison is confounded.
- **Seed determinism:** `set_seed()` is called at the start of both training scripts and in `eval_beavertails.py` — without it, generation sampling is non-deterministic and run-to-run comparisons are meaningless.

## Dependencies

Create the dedicated conda environment (GTX 750 Ti, CUDA 11.8):

```powershell
cd ppo_minmax_experiment
.\setup_env.ps1
conda activate ppo-minmax
```

Or manually:

```bash
conda create -n ppo-minmax python=3.10 -y
conda activate ppo-minmax
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Pinned stack: `trl==0.11.4`, `transformers>=4.37,<5`. Do **not** use the `safe-rlhf` conda env for this project (it has `trl==0.8.6` and `transformers==5.5.0`).