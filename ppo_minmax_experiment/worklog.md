# Research Log — Safe RL via Minmax Penalties

Chronological record of what was tried, what was found, and what didn't work.
Purpose: so I can show my supervisors the actual path, not just a final
result — per Benjamin's advice, and because the debugging trail IS
methodology documentation for the domain-transfer difficulty (ROSARL to
LLM fine-tuning), not something to clean up and hide.

Format per entry: what I did → what I found → what I concluded / did next.
Dates are placeholders (Session N) — backfill real dates as I go, and add
new entries at the bottom as I keep working, in my own words.

---

## Session 1 — Proposal pilot (500-step run)

**Did:** Ran the first proof-of-concept: GPT-2 small, PPO + KL baseline vs
PPO + Minmax, 500 steps, BeaverTails-Evaluation (700 prompts).

**Found:** Baseline harm rate 17.29% (121/700), Minmax 7.14% (50/700) —
58.7% relative reduction. Self-harm regressed +10pp (8%→18%), attributed to
Detoxify's low recall on clinical/neutral self-harm language.

**Concluded:** This is the number in the proposal. Later discovered (Session
4) that both training and eval had bugs at this point, so this number
reflects a broken pipeline, not a trustworthy result — kept here for the
record, not as something to cite going forward.

---

## Session 2 — Reward centering bug

**Did:** Investigated why V_MAX stayed near 0 and self-calibration looked
degenerate.

**Found:** Raw reward was `−toxicity`, bounded to `[−1, 0]`. V_MAX
structurally can't rise above 0 with this convention, so
`R_unsafe = V_MIN − V_MAX` degenerates to `−V_MIN` only.

**Concluded:** Changed reward to `1 − 2×toxicity`, centered to `[−1, 1]`,
giving V_MAX a genuine positive signal to track. This is the reward
convention used everywhere since.

---

## Session 3 — Terminal-token value fix

**Did:** Investigated noisy, unstable V_MIN/V_MAX early in training.

**Found:** Value estimates were being averaged across all tokens in the
response (`values.reshape(-1)`), not just the final one. Early, uncalibrated
per-token noise was polluting the bounds.

**Concluded:** Switched to terminal-token value only
(`values.reshape(-1)[-1:]`) — the position that has attended to the whole
sequence via causal attention, and the position that actually corresponds
to "the value right as the episode ends," matching the single terminal
reward.

---

## Session 4 — Empty-response eval bug (critical)

**Did:** Re-examined why harm rate looked implausibly high in some runs
(~91–94% empty responses).

**Found:** The eval pipeline scored the *prompt* as a fallback whenever the
model generated an empty response, inflating harm rate with numbers that
had nothing to do with what the model actually said.

**Concluded:** Empty responses are now always excluded from toxicity
scoring — no fallback, no opt-in flag. `eval_summary_500.csv` and
`eval_summary_1000.csv` from before this fix are invalid and should not be
cited. This also retroactively invalidates the Session 1 proposal numbers.

---

## Session 5 — Determinism and batching fixes

**Did:** Noticed run-to-run comparisons were meaningless even at the same
seed.

**Found:** `torch.manual_seed()` was never actually called anywhere in
training — generation sampling was non-deterministic. Separately, training
was processing one prompt per step regardless of the configured
`batch_size`, inflating variance in the KL estimator.

**Concluded:** Added `set_seed()` at the start of every script (train and
eval). Rewrote the training loop to genuinely batch generation/scoring/step
together.

---

## Session 6 — PPO epochs and adaptive KL controller

**Did:** Debugged a recurring failure: `objective/kl` going negative, then
`approx_kl` climbing into the hundreds, entropy collapsing, TRL's
average-ratio-over-threshold warning firing repeatedly and skipping
mini-batch updates.

**Found:** Reproduced the same failure in the baseline too (even faster) —
confirming this was a TRL/PPO statistical issue with `adap_kl_ctrl=True` at
low batch size reacting to noisy single-sample KL estimates, not anything
Minmax-specific.

**Concluded:** `ppo_epochs` 4→1, `adap_kl_ctrl=False` (fixed `init_kl_coef`
for the whole run instead). Both conditions must use the same value here —
this is now enforced via the shared `ppo_defaults.py` / `make_experiment_ppo_config()`.

---

## Session 7 — Hard switch vs smooth blending (controlled comparison)

**Did:** Tested whether softening the safe/unsafe reward transition (smooth
blend near the threshold, instead of an immediate hard switch to R_unsafe)
would reduce KL instability.

**Found:** Counterintuitive result — smooth blending produced *higher* KL
than hard switch (max 16.7 vs 11.3 in one comparison), fewer unsafe
triggers, but a 1000-step head-to-head later showed smooth achieving almost
no harm-rate improvement (17.1%→16.4%) vs hard's 17.1%→4.6%.

**Concluded:** Hard switch kept. Smooth blending tested and rejected — this
is documented as a real finding (softening the cliff doesn't help when the
underlying self-calibration signal is what's unstable), not just an
abandoned idea.

---

## Session 8 — Long-run saturation and the −2.0 floor

**Did:** Watched a 1000-step Minmax run closely (`v_min`, `v_max`,
`r_unsafe`, `floor_active` per step).

**Found:** `r_unsafe_raw` (`V_MIN − V_MAX`) reached ≈ −1.996 by step ~150
and stayed essentially frozen for the rest of the run. Realized this isn't
really the −2.0 floor "clipping" anything — with reward bounded to
`[−1, 1]`, `V_MIN − V_MAX ≥ −2` is a mathematical fact regardless of the
floor. The floor was mostly just confirming that ceiling, not doing
independent work.

**Concluded:** This is a structural, not a bug: unlike ROSARL's original
dense/unbounded task reward, the LLM adaptation's reward is bounded by
construction (Detoxify ∈ [0,1]), so `V_MIN`/`V_MAX` inevitably approach
their limits early, and self-calibration (`R_unsafe` "growing more negative
over training") stops meaning anything after that point.

---

## Session 9 — Unbounded log-odds transform (tried, not merged)

**Did:** Explored decoupling the bounds-tracking bookkeeping from the
bounded PPO reward scale — computing `V_MIN`/`V_MAX` on an unbounded
log-odds transform of implied toxicity (`u = log((1−δ)/δ)`, δ clipped to
`[eps, 1−eps]`), while keeping the actual PPO reward unchanged, and
squashing the derived penalty back to a bounded scale via
`penalty_floor × tanh(-gap / calibration_scale)` before handing it to PPO.

**Found:** Delayed and softened saturation, but did not eliminate it.
`eps` re-imposes its own hard bound (`±log((1−eps)/eps)`), and real
GPT-2 + Detoxify data hits values near 0 or 1 toxicity almost immediately
(clean responses routinely score ~0.99+), so both V_MIN and V_MAX still hit
their eps-determined ceiling within the first ~100–150 steps in synthetic
testing matched to the real ~12% unsafe-trigger rate.

**Concluded:** This is a *second*, related structural finding, not a fix:
no reparameterization of a bounded detector output produces genuinely
unbounded self-calibration, because the detector itself (Detoxify) is
fundamentally bounded. This was a real deviation from Algorithm 1 (the
value fed into V_MIN/V_MAX bookkeeping was a synthetic proxy, not the
critic's literal output) and was NOT merged into the live codebase —
superseded by the category-scoped approach in Session 10. Kept here because
the negative result itself is worth citing in the limitations chapter.

---

## Session 10 — Per-category bounds, reward-only bound source (current default)

**Did:** Restructured `MinmaxPenaltyState` to track `V_MIN`/`V_MAX`
per BeaverTails harm category (`bound_scope="category"`) instead of one
global pair, and to compute bounds from the Detoxify reward only by default
(`bound_source="reward"`), only folding in critic values after
`critic_warmup_steps` if `bound_source="reward_and_value"` is explicitly
requested.

**Found:** Not yet evaluated head-to-head against `bound_scope="global"` —
this is a live design choice, not yet a confirmed improvement.

**Concluded:** This is the current default in the codebase. **Open task:**
run a fresh 1000-step Minmax run and check whether `v_min`/`v_max`/
`r_unsafe_raw` show renewed movement across the full run rather than
freezing early, the way the global-bounds version did.

---

## Session 11 — KL asymmetry (baseline β=0.2, Minmax β=0.01)

**Did:** Changed Minmax's KL coefficient from matching baseline (0.2) down
to 0.01, on the reasoning that a strong KL anchor and the Minmax penalty
both pulling the policy at once would compete and blur the safety signal.

**Found:** Not yet isolated — this was implemented but not tested against
a controlled "Minmax at β=0.2" comparison run.

**Concluded:** This is a plausible hypothesis, not a confirmed result. Flagged
explicitly in the README as an open question. **Open task:** run Minmax at
β=0.2 once, matching baseline, to see whether the "competing mechanisms"
story holds up empirically.

---

## Session 12 — Path 1 vs Path 2 (architectural clarification, not a bug fix)

**Did:** Traced exactly where the critic's value estimate is used across
the codebase.

**Found:** Two separate, non-communicating consumers of "value" per step:
(1) TRL's `trainer.step()` internally recomputes its own values for the
actual PPO training update (policy gradient + critic regression) — standard,
unmodified TRL code; (2) `get_value_estimates()`, called separately by this
codebase, feeds only `MinmaxPenaltyState`'s bounds tracking.

**Concluded:** Every saturation/instability finding from Sessions 8–9 lived
entirely in path 2. The core PPO training itself (path 1) was never
affected. This is a genuinely useful framing for the defense: the fragility
is isolated to the auxiliary safety-bookkeeping layer, not the underlying
RL training.

---

## Session 13 — Diagnostic logging expansion

**Did:** Added `entropy`, `value_loss`, `policy_loss`, `clipfrac` to both
training scripts' logs (previously computed by `trainer.step()` and
discarded), and added `plot_baseline_vs_minmax_diagnostics()` to
`plot_results.py` — a combined panel comparing all four between conditions.

**Found:** N/A yet — added proactively, before the next full run, on the
reasoning (Geraud's and Benjamin's advice) that these are the numbers that
explain *why* KL moves the way it does (falling entropy = policy
collapsing; rising clipfrac = a leading indicator before KL spikes) rather
than just showing that it did.

**Concluded:** Requires a fresh baseline run alongside the next Minmax run,
since old baseline logs don't have these columns.

---

## Session 14 — "Advertisements" reward hacking and Minmax policy collapse discovered (seed 42, 1000 steps)

**Date:** 2026-07-08

**Did:** Analysed the full outputs of the seed 42 / 1000-step run: per-prompt eval results
(`eval_1000_seed42_results.csv`, 1400 rows), per-category summary, and both training logs.
Also ran a Python analysis script against all data to surface response distribution,
entropy trajectory, and KL trajectory side by side.

---

### What was found

#### 1. The "Advertisements" response: reward hacking, not safety

Both trained models produce the single word "Advertisements" for the majority of all
700 evaluation prompts:

| Model    | "Advertisements" responses | % | Unique responses |
|----------|---------------------------|------|-----------------|
| Baseline | 392 / 700                 | 56%  | 203             |
| Minmax   | 483 / 700                 | 69%  | **18**          |

"Advertisements" is a **single token** in GPT-2's BPE vocabulary. Detoxify scores it at
toxicity ≈ 0.001, giving a centered reward of 1 − 2 × 0.001 = **0.998** — nearly the
theoretical maximum. Both models discovered this degenerate solution: one benign word,
maximum reward, no detectable toxicity.

This is textbook **reward hacking** / Goodhart's Law: the policy found a way to maximise
the proxy metric (Detoxify toxicity → 0) that is completely decoupled from the actual
safety objective (produce helpful, non-harmful responses to sensitive prompts).

The next-most-common baseline responses were "Yes." (43), "You can't." (17), "I'm not
sure." (12) — all plausible short answers. Minmax's non-"Advertisements" responses are
almost entirely garbage: `...` (38), `+` (31), `P` (24), `""` (15), `"` (12). The policy
has fully collapsed.

#### 2. Minmax: catastrophic, irreversible entropy collapse

The Minmax training log shows total policy collapse well before the run ended:

| Step | Minmax entropy | KL divergence | Baseline entropy | KL |
|------|---------------|---------------|------------------|----|
| 0    | 3.45          | 0.00          | 3.45             | 0.00 |
| 50   | 3.27          | **+43.8**     | 4.39             | +4.5 |
| 95   | **< 1.0**     | —             | ~4.2             | — |
| 100  | 0.91          | +15.5         | 4.21             | +2.6 |
| 223  | **< 0.1**     | —             | —                | — |
| 499  | 0.51          | −7.9          | 3.30             | +3.7 |
| 749  | 2.92          | +5.3          | 3.49             | +4.7 |
| 900  | **0.0001**    | −11.2         | 2.71             | +6.0 |
| 999  | **0.0075**    | **−41.0**     | 3.73             | +4.2 |

Entropy of 0.0075 at step 999 is a near-deterministic policy. For reference, GPT-2's
initial entropy is ~3.45 bits. Minmax ended the run at less than 0.003× its initial
diversity.

**Baseline**, by contrast, maintained healthy entropy throughout (2.3–4.7, mean ~3.7) and
positive KL throughout (1.1–7.3, mean ~5.0 in the last 100 steps). This is the difference
between a policy that learned something and one that degenerated.

#### 3. Massively negative KL is a numerical breakdown signal, not a real measurement

Minmax's last 100 steps had **57 out of 100 steps with negative KL divergence** (mean −4.91,
min −41.04). KL divergence cannot be negative in theory — it is always ≥ 0 by definition
(Gibbs' inequality). When TRL's Monte-Carlo KL estimator reports −41, it means the
current policy has moved so far from the reference model that the importance-sampling
weights used in the estimate are essentially zero, causing numerical breakdown. The
"negative KL" is noise, not signal — but it confirms the policy has drifted far outside
the region where the KL penalty can be trusted to act as an anchor.

In the last 100 steps, Minmax entropy oscillated between 0.0001 and 3.29 — the policy
was not stably collapsed but spasming between near-determinism and occasional wider
exploration. This is consistent with a highly peaked, occasionally numerically perturbed
softmax rather than genuine diversity.

#### 4. Minmax "0% harm rate" is meaningless

The summary CSV shows `minmax, _overall, harm_rate = 0.000`. This looks like a strong
result. It is not. Minmax achieved 0% harm by:
- Outputting "Advertisements" for 69% of prompts (scored as harmless by Detoxify)
- Leaving 11% of prompts with an empty response (excluded from harm scoring by design)
- Producing garbage single tokens (`+`, `P`, `""`, etc.) for the remaining 20%

There is no evidence the model learned to refuse, redirect, or respond safely to
harmful prompts. It learned to produce one benign token. A policy that outputs
"Advertisements" to every prompt — including "What is 2+2?" — scores 0% harm. That is
not safety.

Baseline's 3.1% harm rate reflects real (if modest) failures: the model actually generated
substantive responses, and a small fraction of those were genuinely problematic (e.g.,
sexual content, violence). These 22 cases are a real failure mode. Minmax's 0 cases are
an artefact of a collapsed, degenerate policy.

#### 5. The "Advertisements" pattern may predate RL training entirely

A critical observation: `eval_beavertails.py` uses **greedy decoding** (`do_sample=False`)
by default at evaluation time, while both training scripts use **stochastic sampling**
(`do_sample=True, top_k=0, top_p=1.0`) during the PPO rollout generation step. These
are fundamentally different decoding strategies.

If raw GPT-2 (no RL at all) already produces "Advertisements" greedily on the majority
of BeaverTails prompts, then RL training merely reinforced a pre-existing greedy mode of
the base model — it did not invent this behavior from scratch. In that scenario, what
we're measuring as "reward hacking" may partly be "greedy decoding of the initial model
on prompts where GPT-2 happens to strongly favour a benign single-token completion."

This is the key open question as of this session.

#### 6. Category breakdown of the "Advertisements" pattern (baseline)

"Advertisements" was NOT uniformly distributed across harm categories:

**Highest "Advertisements" rates (baseline):**
- hate_speech: 84% (42/50)
- offensive_language: 84% (42/50)
- privacy_violation: 82% (41/50)
- sexually_explicit: 82% (41/50)
- adult_content: 82% (41/50)

**Lowest "Advertisements" rates (most real responses):**
- aiding_and_abetting: 4% (2/50)
- violence: 4% (2/50)
- incitement: 4% (2/50)
- self_harm: 20% (10/50)
- organized_crime: 24% (12/50)

This pattern is suspicious. Categories where "Advertisements" is rare (violence,
incitement, aiding_and_abetting) are the exact categories where baseline generated
the most harmful responses. The model may be producing "Advertisements" as a genuine
GPT-2-prior response to prompts that contain typical advertising/commercial vocabulary,
not as a learned refusal. Violence/incitement prompts likely contain very different
vocabulary that doesn't prime GPT-2 toward "Advertisements."

#### 7. The KL asymmetry as a direct cause of Minmax collapse

Minmax uses β=0.01 (KL penalty coefficient), baseline uses β=0.2 — a 20× difference,
still flagged as an untested hypothesis in the README. With β=0.01, the KL penalty
contributes almost nothing to the PPO objective. Without a meaningful anchor to the
reference policy, Minmax is free to race directly toward the degenerate reward-maximising
solution ("Advertisements" always) and does so within ~100 steps (entropy < 1.0 by step
95). The Minmax penalty (`R_unsafe`) was supposed to create a safety constraint, but it
cannot prevent collapse to a non-toxic degenerate mode because "Advertisements" is not
classified as unsafe by Detoxify — so `R_unsafe` is never triggered for it.

**The structural problem:** The Minmax penalty only fires for responses with toxicity >
0.3 (reward < 0.4). A policy that always outputs "Advertisements" (toxicity ≈ 0.001)
never encounters an unsafe state, so `R_unsafe` is never applied, and the penalty does
exactly nothing once the policy has converged to this mode. The safety mechanism has
been optimised away.

---

### What was concluded

1. **The seed 42 / 1000-step results are invalid for comparing baseline vs Minmax**
   as a safety mechanism. Minmax collapsed to a degenerate policy. Baseline
   is weakly useful (genuine responses, modest harm) but also heavily dominated
   by "Advertisements."

2. **The "Advertisements" hacking needs to be traced to its origin.** Two hypotheses:
   - **H1 (GPT-2 prior):** Raw GPT-2, greedily decoded, already outputs "Advertisements"
     for >50% of BeaverTails prompts. RL just reinforced a pre-existing mode. If true,
     the eval metric is broken regardless of RL algorithm.
   - **H2 (RL artifact):** Raw GPT-2 outputs diverse text. "Advertisements" is a mode
     the RL training found and amplified. If true, the reward function is broken —
     any policy trained with `reward = 1 − 2 × Detoxify(response)` will converge to
     near-zero-toxicity single-token outputs.

3. **Immediate next investigation:** Evaluate raw GPT-2 (no RL, `gpt2` weights from
   HuggingFace) on the same 700-prompt BeaverTails-Evaluation set using the same greedy
   eval pipeline. Compare "Advertisements" rate, response diversity, and harm rate.

4. **If H2 is confirmed (raw GPT-2 is diverse):** The eval pipeline has a second,
   separate problem: greedy vs sampled decoding. Models trained with sampling and
   evaluated with greedy will always appear more degenerate than their training behavior.
   The "right" eval is to use `--do-sample True` with the same seed as training — this
   is the next investigation after the GPT-2 baseline.

5. **Longer term:** Even if these issues are resolved, the Minmax penalty as implemented
   cannot solve reward hacking because it is downstream of the same Detoxify reward
   signal that is being gamed. A policy that games Detoxify gets maximum reward AND
   never triggers `R_unsafe`. The penalty has no leverage over the degenerate mode.
   This may be the most important structural finding of the whole project.

---

### Open questions added by this session

- Does raw GPT-2 (greedy) already output "Advertisements" at high rates on BeaverTails?
  **→ Currently being investigated (Session 15 pending).**
- If not: does the "Advertisements" rate drop substantially when eval uses
  `--do-sample True`?
- Does Minmax collapse to "Advertisements" at β=0.2 as well, or is the weak KL anchor
  (β=0.01) the proximate cause?
- Is there any configuration of this reward function and policy that doesn't converge to
  near-zero-toxicity single-token outputs, short of changing the reward function itself?

---

## Session 15 — Raw GPT-2 eval: "Advertisements" is an RL artifact (H2 confirmed)

**Date:** 2026-07-08

**Did:** Saved untrained `gpt2` weights to `models/gpt2_raw` and ran
`eval_beavertails.py` on the same 700-prompt BeaverTails-Evaluation set with
greedy decoding (`--do-sample False`, same as the trained-model eval).
Output: `eval_1000_gpt2raw_results.csv`, `eval_1000_gpt2raw_summary.csv`.

**Found:**

| Model | "Advertisements" | Unique responses | Empty | Harm (nonempty) |
|-------|-----------------|------------------|-------|-----------------|
| Raw GPT-2 | **0 / 700 (0%)** | **669** | 0% | **7.1%** |
| Trained baseline | 392 / 700 (56%) | 203 | 0.1% | 3.1% |
| Trained minmax | 483 / 700 (69%) | 18 | 11% | 0.0% |

Raw GPT-2 does **not** greedily output "Advertisements" on BeaverTails prompts.
The most common responses are repetitive but diverse continuations (e.g. "I think
it's a very important question..." repeated, "I don't know..." loops). Harm rate
(7.1%) is higher than trained baseline (3.1%) — RL actually reduced measured harm,
but by collapsing to a degenerate non-toxic token, not by learning safe behaviour.

**Concluded:** **H1 rejected, H2 confirmed.** "Advertisements" is not a GPT-2
greedy prior on this eval set. It was learned during PPO training as a
Detoxify-reward exploit. The train-sample / eval-greedy mismatch may still matter
for other response patterns, but it is not the origin of "Advertisements."

**Next:** Re-run eval on trained checkpoints with `--do-sample True --seed 42`
to see whether sampling restores diversity or whether the policy has permanently
collapsed regardless of decoding mode.

**Note on environment:** The seed-42 1000-step training ran in `conda activate
safe-rlhf` (trl 0.8.6, transformers 5.5.0), not the README-pinned stack
(trl 0.11.4, transformers <5). That may have worsened PPO instability but did
not cause "Advertisements" — raw GPT-2 in base env shows 0% regardless.

---

## Session 16 — Dedicated `ppo-minmax` conda env + do_sample eval

**Date:** 2026-07-08

**Did:** Created `ppo-minmax` conda env (Python 3.10) with pinned stack for GTX 750 Ti
(2 GB VRAM, driver 560.94): `torch 2.7.1+cu118`, `transformers 4.57.6`, `trl 0.11.4`,
`detoxify`, `datasets`, etc. Added `requirements.txt` and `setup_env.ps1`.
Updated README to document the env and warn against using `safe-rlhf` for this project.

**Found:** Env verifies: CUDA available, GPU detected as GTX 750 Ti.

**Doing next:** Running `eval_beavertails.py` on trained baseline + minmax checkpoints
with `--do-sample --seed 42` (tag `seed42_sample`) in the new env.

**Found (do_sample eval complete):**

| Decoding | Model | Advertisements | Unique | Empty | Harm |
|---|---|---|---|---|---|
| Greedy | baseline | 392 (56%) | 203 | 1 | 22 |
| Greedy | minmax | 483 (69%) | 18 | 77 | 0 |
| Sample | baseline | 21 (3%) | 668 | 2 | 21 |
| Sample | minmax | 329 (47%) | 150 | 58 | 0 |

Sampling **dramatically reduced** baseline "Advertisements" (56% → 3%) and restored
diversity (203 → 668 unique responses). So for baseline, greedy decoding was a major
part of the collapse — but RL still learned something (harm 3.0% vs raw GPT-2 7.1%).

For minmax, sampling helped less: still 47% "Advertisements", only 150 unique
responses, 8.3% empty, 0% harm. The policy is still heavily degenerate; sampling
shifts the mode from pure "Advertisements" toward a mix of garbage + empty, not
toward real answers.

**Concluded:** Train-sample / eval-greedy mismatch explains **baseline** collapse
under greedy eval, not the full story for **minmax**. Both conditions still show
reward hacking; minmax collapse is structural (weak KL + Detoxify exploit), not
just a decoding bug.

---

## Session 17 — Minmax β=0.2 ablation: KL confirmed as proximate cause

**Date:** 2026-07-08

**Did:** Trained minmax at β=0.2 (matching baseline) in `ppo-minmax` env (6495s).
Evaluated with `--do-sample --seed 42` (tag `kl02_sample`).

**Found:**

| Run | β | Sample eval: Advertisements | Unique | Empty | Harm |
|---|---|---|---|---|---|
| minmax (old) | 0.01 | 47% | 150 | 58 | 0% |
| **minmax_kl02** | **0.2** | **1.4%** | **684** | **1** | **2.0%** |
| baseline (sample) | 0.2 | 3.0% | 668 | 2 | 3.0% |
| raw GPT-2 | — | 0% | 669 | 0 | 7.1% |

Training: final entropy **3.31** (vs 0.0075 at β=0.01), KL **+3.19** (vs −41).

**Concluded:** Weak KL (β=0.01) was the **proximate cause** of minmax policy collapse.
At β=0.2, minmax behaves like baseline — diverse responses, modest harm (2.0%),
no Advertisements gaming. The minmax penalty is not worse than baseline under matched KL;
the earlier 0% harm / 69% Advertisements result was an artifact of collapse, not evidence
that minmax "works" or "fails" as a safety mechanism. **Next:** fair A/B with both at
β=0.2 in `ppo-minmax`, eval protocol aligned (sample + greedy).

---

## Session 18 — Fair baseline retrain in `ppo-minmax` (closed)

**Date:** 2026-08-03

**Did:** Retrained baseline (β=0.2, 1000 steps, seed 42) in `ppo-minmax` (trl 0.11.4 /
transformers 4.57.6) to match `minmax_kl02`. Did not overwrite the old `safe-rlhf`
baseline:

- Log: `logs/baseline_ppominmax_training.csv`
- Checkpoint: `checkpoints/baseline_ppominmax`
- Train time: 7537s (~126 min). Final entropy **3.45**, KL **+4.73** (healthy).
- Sample eval (`--do-sample --seed 42`, tag `baseline_ppominmax_sample`).

**Found (fair A/B, both in `ppo-minmax`, β=0.2, sample eval):**

| Run | Ads | Unique | Empty | Harm |
|---|---|---|---|---|
| Raw GPT-2 | 0% | 669 | 0 | 7.1% |
| **baseline_ppominmax** | **0.3%** | **674** | **0** | **2.3%** |
| **minmax_kl02** | **1.4%** | **684** | **1** | **2.0%** |
| Old baseline (safe-rlhf, sample) | 3.0% | 668 | 2 | 3.0% |

**Concluded:** Under a matched environment and KL, Minmax and baseline are essentially
tied (2.0% vs 2.3%, one seed). Both beat raw GPT-2; neither collapses; Advertisements
hacking is gone. The Phase-1 GPT-2 + Detoxify pilot is **wrapped**: the honest claim is
that Minmax is *viable* at matched KL, not that it clearly outperforms PPO+KL. Further
gains need a better reward / detector (or fixed-penalty ablation), not more β=0.01 runs.

**Parked (not blocking wrap-up):** fixed-penalty comparator; category vs global bounds;
reward redesign; second seed; Phase 2 probe.

---

## Open tasks (carry forward, update as completed)

## old tasks

- [ ] Run fresh 1000-step baseline + Minmax with current
      (category-scoped, reward-only-bounds) config; check whether
      `r_unsafe_raw` still saturates early or shows renewed movement.
- [ ] Run `plot_results.py`, especially `fig7` (baseline vs Minmax
      diagnostics), and look for whether Minmax visibly destabilizes
      entropy/clipfrac relative to baseline.
- [ ] Isolated ablation: Minmax at β=0.2 (matching baseline) — test the
      "competing mechanisms" hypothesis directly.
- [ ] Fixed-magnitude-penalty comparator (always −2.0 when unsafe, no
      V_MIN/V_MAX at all) — cheap to add, tells you whether self-calibration
      is earning its keep over a simple fixed penalty.

**Priority 1 — diagnose the "Advertisements" issue before any new training runs:**
- [x] Evaluate raw GPT-2 (no RL) on BeaverTails-Evaluation (greedy). **Result: 0%
      Advertisements, 669 unique responses, 7.1% harm. H2 confirmed — RL artifact.**
- [x] Re-run eval on trained baseline + minmax checkpoints with
      `--do-sample True --seed 42` (tag `seed42_sample`, env `ppo-minmax`).
      **Result: sampling did not restore diversity; minmax 8.3% empty, 0% harm.**
- [ ] Assess whether the reward function (Detoxify-based, single-token-exploitable) can
      be fixed before running more experiments, or whether all future results under this
      reward are uninterpretable.

**Priority 2 — ablations (only meaningful once the eval pipeline is trusted):**
- [ ] Run fresh 1000-step baseline + Minmax with current (category-scoped,
      reward-only-bounds) config; check whether `r_unsafe_raw` still saturates early
      or shows renewed movement.
- [ ] Run `plot_results.py`, especially `fig7` (baseline vs Minmax diagnostics), and
      look for whether Minmax visibly destabilizes entropy/clipfrac relative to baseline.
- [x] Minmax ablation at β=0.2 (matching baseline) in `ppo-minmax` env.
      Log: `logs/minmax_kl02_training.csv`, checkpoint: `checkpoints/minmax_kl02`.
      **Done in 6495s. Final entropy 3.31, KL +3.19 (vs β=0.01: 0.0075, −41).**
- [x] Eval `minmax_kl02` with `--do-sample --seed 42` (tag `kl02_sample`).
      **Result: 1.4% Advertisements, 684 unique, 2.0% harm — collapse fixed.**
- [x] Fair baseline retrain in `ppo-minmax` (Session 18).
      Log/ckpt: `baseline_ppominmax_*`. Sample eval: **0.3% Ads, 674 unique, 2.3% harm.**
      Fair A/B closed: minmax 2.0% vs baseline 2.3% (one seed).
- [ ] Fixed-magnitude-penalty comparator (always −2.0 when unsafe, no V_MIN/V_MAX at
      all) — cheap to add, tells you whether self-calibration is earning its keep over
      a simple fixed penalty. **Parked.**
- [ ] `bound_scope="category"` vs `"global"` — not yet compared head to head.
- [ ] (Future work, not current scope) Advantage-weighted alternative to global bounds
      tracking — sketched conversationally, not implemented. Would test a different
      mechanism, not Algorithm 1 itself — worth a paragraph in the
      discussion/future-work section, not a pivot.

---

## Notes on how to use this file going forward

- Add a new `## Session N` block every time I try something and get a real
  result — good, bad, or confusing. A dead end is still an entry.
- Every entry should have a **Did / Found / Concluded** shape — resist the
  urge to only write down the ones that worked.
- If a "Concluded" turns out to be wrong later, don't delete it — add a new
  entry that says so and why. The correction is itself part of the record.