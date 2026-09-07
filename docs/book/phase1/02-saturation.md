# 2. Bounded reward and saturation

Sessions 8–9 answer a question that looks like “why is the floor clipping?” and turns out to be “why can’t self-calibration grow when the detector is bounded?” This is the first deep structural finding of Phase 1 — and the reason Phase 2 moved to a different safety signal.

```mermaid
flowchart TB
  D["Detoxify δ ∈ [0,1]"] --> R["Centered reward r = 1−2δ ∈ [−1,1]"]
  R --> Vmin["V_MIN tracks low values"]
  R --> Vmax["V_MAX tracks high values"]
  Vmin --> Gap["R_unsafe = V_MIN − V_MAX"]
  Vmax --> Gap
  Gap --> Bound["Gap ≥ −2 by arithmetic"]
  Bound --> Floor["Engineering floor −2 mostly confirms the math"]
```

## Session 8 — Long-run saturation and the $-2.0$ floor

A 1000-step MinMax run was watched step-by-step: $v_{\min}$, $v_{\max}$, $r_{\text{unsafe}}$, `floor_active`.

**Observation.** `r_unsafe_raw` ($V_{\MIN} - V_{\MAX}$) reached $\approx -1.996$ by step ~150 and stayed essentially frozen for the rest of the run.

At first glance that looks like the engineering floor of $-2$ “clipping” the penalty. It is not (mostly). With reward bounded to $[-1, 1]$,

$$
V_{\MIN} - V_{\MAX} \ge -2
$$

is a **mathematical fact**, independent of any clip. The floor mostly confirms that ceiling; it does little independent work once the bounds have approached their extremes.

<div class="finding">
<span class="label">Finding</span>
Unlike ROSARL’s original dense / unbounded task reward, the LLM adaptation’s reward is bounded by construction (Detoxify ∈ [0,1]). $V_{\MIN}$ / $V_{\MAX}$ inevitably approach their limits early, and self-calibration — “$R_{\text{unsafe}}$ growing more negative over training” — stops meaning anything after that point.
</div>

This is structural, not a coding bug. The bookkeeping is doing what the math allows; the math does not allow unbounded growth of the gap under a bounded reward.

## Session 9 — Unbounded log-odds (tried, not merged)

If the problem is the bounded scale, maybe track bounds on an unbounded transform while leaving the PPO reward alone.

**Attempt.** Compute $V_{\MIN}$ / $V_{\MAX}$ on a log-odds transform of implied toxicity

$$
u = \log\frac{1-\delta}{\delta}, \qquad \delta \text{ clipped to } [\varepsilon, 1-\varepsilon],
$$

keep the actual PPO reward unchanged, and squash the derived penalty back for PPO via

$$
\text{penalty} = \texttt{penalty\_floor} \times \tanh\!\left(\frac{-\text{gap}}{\texttt{calibration\_scale}}\right).
$$

**Result.** Saturation was delayed and softened, but not eliminated.

| Why it still fails | Detail |
|---|---|
| $\varepsilon$ re-bounds $u$ | Hard range $\pm\log((1-\varepsilon)/\varepsilon)$ |
| Real data hits extremes | Clean GPT-2 + Detoxify responses routinely score $\delta \approx 0.99+$ |
| Timing | Both $V_{\MIN}$ and $V_{\MAX}$ still hit the $\varepsilon$-ceiling within ~100–150 steps in synthetic tests matched to the real ~12% unsafe-trigger rate |

<div class="finding caution">
<span class="label">Caution — Algorithm 1 deviation</span>
The value fed into $V_{\MIN}$ / $V_{\MAX}$ bookkeeping was a synthetic proxy, not the critic’s literal output. That is a real deviation from Algorithm 1. The change was <strong>not</strong> merged into the live codebase; it was superseded by the category-scoped approach in <a href="/book/phase1/03-design-choices.md">Design choices</a>. Keep the negative result for the limitations chapter.
</div>

**Second structural finding.** No reparameterization of a bounded detector output produces genuinely unbounded self-calibration, because the detector itself (Detoxify) is fundamentally bounded.

## Why this matters for the rest of Phase 1

Saturation explains why “watch $R_{\text{unsafe}}$ grow more negative” stopped being a useful training diagnostic after ~150 steps. It does *not* yet explain the later Advertisements collapse — that is a separate failure mode (weak KL + gameable proxy) documented in [Advertisements collapse](/book/phase1/04-advertisements-collapse.md).

What Sessions 8–9 *do* motivate next: change how bounds are scoped and sourced (Session 10), and isolate whether fragility lives in PPO itself or only in the auxiliary bookkeeping (Session 12).
