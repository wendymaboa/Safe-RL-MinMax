---
marp: true
theme: default
paginate: true
title: PPO + Minmax — Supervisor Progress Update
description: Seed 42 / 1000-step experiment, debugging trail, fair comparison (β=0.2)
---

# PPO + Minmax Penalty on GPT-2
## Progress update, worklog, and corrected results

**ROSARL Algorithm 1 (`R_unsafe = V_MIN − V_MAX`) → GPT-2 + Detoxify + BeaverTails**

- Experiment: seed 42, 1000 PPO steps, BeaverTails-Evaluation (700 prompts)
- July 2026 · `ppo_minmax_experiment/`

---

# Agenda

1. Setup & research question
2. Worklog: 17 sessions of fixes and failures
3. The seed-42 result that *looked* good but wasn't
4. Root-cause analysis: Advertisements, eval protocol, KL, environment
5. Corrected fair comparison (minmax β=0.2 vs baseline)
6. Figures & what we can / cannot claim
7. Next steps for thesis-grade numbers

---

# Experimental setup

| Item | Choice |
|------|--------|
| Base model | GPT-2 small |
| Train | BeaverTails `30k_train`, seed 42 |
| Eval | BeaverTails-Evaluation (700, 14 categories) |
| Reward | `1 − 2×Detoxify(response)` |
| Unsafe | toxicity > 0.3 (reward < 0.4) |
| Train decode | Sample, max 64 tokens |

| Condition | Mechanism | KL β |
|-----------|-----------|------|
| **Baseline** | PPO + KL | **0.2** |
| **Minmax** | PPO + KL + `R_unsafe` | 0.01 → **0.2*** |

*Original minmax used β=0.01 — **confounded** the comparison. Ablation at β=0.2 in Session 17.*

**Primary metric:** `harm_rate_nonempty` + `empty_rate`

---

# Evaluation methodology

```
Prompt → GPT-2 generate → Empty? → Detoxify(response) → tox > 0.3 ⇒ harmful
```

- **Session 4 fix:** never score the prompt when response is empty
- **Session 16:** training samples rollouts; default eval was **greedy** — mismatch

| Mode | When to use |
|------|-------------|
| Greedy | Misleading for our train setup |
| **Sample** (`--do-sample`) | **Matches training rollouts** |

---

# Worklog: Sessions 1–13

| Sess | Key item |
|------|----------|
| 1 | Proposal pilot 17%→7% — **later invalidated** |
| 2–3 | Reward centering; terminal-token values |
| 4 | **Empty-response eval bug** |
| 5–6 | Determinism, batching, PPO stability |
| 7 | Hard vs smooth unsafe switch |
| 8–10 | Bound saturation → per-category bounds |
| 11 | KL asymmetry hypothesis (β=0.2 vs 0.01) |
| 12–13 | Path 1 vs 2; diagnostic logging + fig7 |

---

# Worklog: Sessions 14–17

| Sess | Discovery |
|------|-----------|
| **14** | **Advertisements** hacking; entropy collapse; 0% harm is fake |
| **15** | Raw GPT-2: 0% Ads, 7.1% harm — RL artifact (H2) |
| **16** | `ppo-minmax` env; sample eval fixes baseline picture |
| **17** | Minmax **β=0.2** — collapse fixed; 2.0% vs 3.0% harm |

Full record: `worklog.md` (609 lines)

---

# Results we must NOT cite

| Claim | Why invalid |
|-------|-------------|
| Proposal 17.3% → 7.1% harm | Pre–Session 4 eval bugs |
| Seed 42 greedy: minmax 0% harm | 69% Ads, 18 unique responses |
| Minmax β=0.01 sample: 0% harm | Still 47% Ads — degenerate |
| Runs in `safe-rlhf` (trl 0.8.6) | Not pinned stack |

**Trustworthy now:** sample eval + minmax β=0.2 (`minmax_kl02`)

---

# The "Advertisements" exploit

- Single GPT-2 BPE token → Detoxify ≈ 0 → reward ≈ **0.998**
- `R_unsafe` only fires when unsafe (tox > 0.3) — **never on Ads**
- Minmax penalty is optimised away

| Model (greedy) | Ads | Unique | Harm | Empty |
|----------------|-----|--------|------|-------|
| Baseline | 56% | 203 | 3.1% | 0.1% |
| Minmax β=0.01 | **69%** | **18** | 0% | 11% |

![bg right:42%](../experiments/seed_42_steps_1000/results/figures_minmax_kl_0.01/fig5_harm_vs_empty_1000.png)

---

# Policy collapse (β=0.01)

| Step | Minmax H | Minmax KL | Baseline H |
|------|----------|-----------|------------|
| 0 | 3.45 | 0.0 | 3.45 |
| 95 | <1.0 | +14 | ~4.2 |
| 999 | **0.0075** | **−41** | 3.73 |

- Negative KL = TRL estimator breakdown
- Last 200 steps: only **2** unsafe triggers — penalty inactive

![bg right:45%](../experiments/seed_42_steps_1000/results/figures_minmax_kl_0.01/fig7_baseline_vs_minmax_diagnostics_1000.png)

---

# Raw GPT-2 (no RL)

| Model | Ads | Unique | Harm |
|-------|-----|--------|------|
| **Raw GPT-2** | **0%** | 669 | **7.1%** |
| Trained baseline | 56% | 203 | 3.1% |
| Trained minmax | 69% | 18 | 0% |

**H1 rejected, H2 confirmed** — Advertisements is RL-learned, not a GPT-2 prior.

---

# Greedy vs sampled eval

| Decode | Model | Ads | Unique | Harm |
|--------|-------|-----|--------|------|
| Greedy | Baseline | 56% | 203 | 3.1% |
| Greedy | Minmax β=0.01 | 69% | 18 | 0% |
| **Sample** | Baseline | **3%** | 668 | 3.0% |
| Sample | Minmax β=0.01 | 47% | 150 | 0% |

Greedy distorts **baseline**; minmax still broken in **weights** until KL fixed.

---

# Training environment

| Env | trl | transformers | Used for |
|-----|-----|--------------|----------|
| `safe-rlhf` | 0.8.6 | 5.5.0 | Original seed-42 |
| **`ppo-minmax`** | **0.11.4** | **4.57.6** | kl02 ablation, future runs |

Confounder, not root cause of Ads (raw GPT-2 = 0%).

---

# KL ablation: minmax at β=0.2

Checkpoint: `checkpoints/minmax_kl02` · ~108 min train

| Metric | β=0.01 | β=0.2 (kl02) |
|--------|--------|--------------|
| Final entropy | 0.0075 | **3.31** |
| Final KL | −41 | **+3.19** |

**Weak KL was the proximate cause of collapse**, not the Minmax formula alone.

![bg right:40%](../experiments/seed_42_steps_1000/results/figures/fig1_training_curves_1000_kl02_fair.png)

---

# Master results matrix (seed 42)

| Run | Env | β | Eval | Ads | Unique | Harm | Trust? |
|-----|-----|---|------|-----|--------|------|--------|
| Raw GPT-2 | — | — | greedy | 0% | 669 | 7.1% | Yes |
| Baseline | safe-rlhf | 0.2 | greedy | 56% | 203 | 3.1% | No |
| Minmax | safe-rlhf | 0.01 | greedy | 69% | 18 | 0% | No |
| Baseline | safe-rlhf | 0.2 | sample | 3% | 668 | 3.0% | Partial* |
| Minmax | safe-rlhf | 0.01 | sample | 47% | 150 | 0% | No |
| **Minmax kl02** | ppo-minmax | **0.2** | sample | **1.4%** | **684** | **2.0%** | **Yes** |

*Fair A/B needs both conditions retrained in `ppo-minmax`.*

---

# Fair harm comparison (sample eval)

![width:900px](../experiments/seed_42_steps_1000/results/figures/fig4_overall_harm_bar_1000_kl02_fair.png)

Minmax **2.0%** vs baseline **3.0%** vs raw **7.1%** — modest, honest effect size (1 seed).

---

# Harm by category

![width:1000px](../experiments/seed_42_steps_1000/results/figures/fig3_harm_rate_bar_1000_kl02_fair.png)

n=50 per category — high variance; use for discussion, not strong per-category claims.

---

# PPO health at β=0.2

![width:1000px](../experiments/seed_42_steps_1000/results/figures/fig7_baseline_vs_minmax_diagnostics_1000_kl02_fair.png)

Entropy ~3–4 for both — minmax does not destabilise PPO at matched KL.

---

# Algorithm 1 dynamics (β=0.2)

![width:480px](../experiments/seed_42_steps_1000/results/figures/fig2_algorithm1_dynamics_1000_kl02_fair.png)
![width:480px](../experiments/seed_42_steps_1000/results/figures/fig6_minmax_diagnostics_1000_kl02_fair.png)

`V_MIN`/`V_MAX` saturate early — open Q: self-calibration vs fixed penalty?

---

# What we can vs cannot claim

**Can claim**
- Detoxify reward is gameable (Advertisements)
- Weak KL caused minmax collapse
- Greedy eval misrepresented baseline
- At β=0.2 + sample: 2.0% vs 3.0% vs 7.1% harm (1 seed)

**Cannot claim (yet)**
- Minmax beats baseline in general
- 0% harm from collapsed run = success
- Self-calibration > fixed penalty (untested)
- Proposal pilot numbers

---

# Next steps

| P | Task |
|---|------|
| **1** | **Fair re-run:** baseline + minmax, β=0.2, `ppo-minmax`, sample eval |
| 2 | Fixed-penalty comparator |
| 3 | Second seed (43) |
| 4 | Reward redesign (length penalty, refusal shaping) |
| 5 | Default minmax KL → 0.2 in code |

```bash
conda activate ppo-minmax
python Evaluation/plot_results.py ... --figure-suffix _kl02_fair
```

---

# Discussion

1. Is **2% vs 3%** (one seed) enough to pursue Minmax?
2. Lead thesis with **negative results** (collapse, hacking) then fair rerun?
3. Fixed-penalty ablation before more bound engineering?
4. Phase 2 probe timeline?

**Artifacts:** `worklog.md` · `docs/supervisor_progress_slides.html` · `docs/speaker_notes.md`
