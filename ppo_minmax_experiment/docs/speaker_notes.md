# Speaker notes — Supervisor progress update (21 slides)

Open `supervisor_progress_slides.html` in a browser.  
**Controls:** ← → or click left/right half · **N** = speaker notes · **F** = fullscreen · **P** = print to PDF

---

## Slide 1 — Title
- Frame this as an honest progress update, not a victory lap.
- Benjamin's advice: show supervisors the *path*, including dead ends.
- Experiment: seed 42, 1000 PPO steps, BeaverTails-Evaluation.

## Slide 2 — Agenda
- Set expectation: ~half the talk is "what went wrong and how we found it."
- End with actionable next steps, not just problems.

## Slide 3 — Setup
- Two conditions: baseline (PPO+KL) vs minmax (adds R_unsafe on unsafe rollouts).
- **Confound to flag early:** original minmax used β=0.01 vs baseline β=0.2 — not a clean A/B.
- Reward is Detoxify-based; unsafe threshold = toxicity > 0.3.

## Slide 4 — Eval pipeline
- Session 4 fix: empty responses never scored via prompt fallback.
- Session 16: training uses **sampling**, default eval was **greedy** — distorts baseline especially.
- Primary metric: `harm_rate_nonempty`; always pair with `empty_rate`.

## Slide 5 — Worklog fixes
- Skim quickly unless asked — shows systematic engineering.
- Key invalidation: Session 1 proposal numbers (17%→7%) must not be cited.
- Session 11 KL asymmetry was hypothesis-only until Session 17.

## Slide 6 — Worklog discoveries
- Sessions 14–17 are the core narrative for this meeting.
- Point them to `worklog.md` for full detail (609 lines).

## Slide 7 — Do not cite
- **Critical slide** — preempt "but you showed 0% harm before."
- Collapsed minmax is not evidence minmax works.
- `safe-rlhf` env (trl 0.8.6) is another confounder.

## Slide 8 — Advertisements
- Single GPT-2 BPE token → Detoxify ≈ 0 → reward ≈ 0.998.
- R_unsafe never fires on this mode → minmax penalty is powerless against it.
- Show fig5: 0% harm alongside 11% empty on collapsed minmax.

## Slide 9 — Policy collapse
- Entropy 0.0075 at step 999 vs baseline 3.73.
- Negative KL is estimator breakdown, not real physics.
- Only 2 unsafe triggers in last 200 steps — penalty inactive.
- fig7: entropy panel is the smoking gun.

## Slide 10 — Raw GPT-2
- 0% Advertisements, 669 unique responses, 7.1% harm.
- Proves RL introduced the exploit (H2), not GPT-2 prior (H1 rejected).
- Raw GPT-2 is the sensible "no training" reference point.

## Slide 11 — Greedy vs sample
- Baseline: 56% → 3% Advertisements when switching to sample eval.
- Minmax β=0.01 still 47% Ads under sample — collapse in weights.
- Two separate issues: eval protocol (baseline) + weak KL (minmax).

## Slide 12 — Environment
- Original training in `safe-rlhf` (trl 0.8.6, transformers 5.5.0).
- New `ppo-minmax` env pins trl 0.11.4 — matches README.
- Confounder, not root cause of Advertisements.

## Slide 13 — KL ablation
- Retrained minmax at β=0.2 in ~108 min.
- Entropy 3.31, KL +3.19 — healthy training.
- **Main technical finding:** weak KL caused collapse, not minmax formula per se.
- fig1: both conditions hit high reward (Detoxify ceiling).

## Slide 14 — Master results table
- **Money slide** — walk row by row.
- Only `minmax_kl02` + sample eval is fully trustworthy for minmax today.
- Baseline sample eval is partial trust (old env checkpoint).
- Emphasize: fair re-run needed for thesis table.

## Slide 15 — Fair harm (fig4)
- Minmax 2.0% vs baseline 3.0% vs raw 7.1%.
- Modest 1pp improvement — be honest about effect size.
- Single seed only.

## Slide 16 — Per category (fig3)
- n=50 per category → noisy bars.
- Use for discussion, not strong claims per category.

## Slide 17 — PPO diagnostics (fig7)
- At β=0.2, minmax does not destabilize entropy vs baseline.
- Contradicts story that "minmax breaks PPO" — it was KL mismatch.

## Slide 18 — Algorithm 1 (fig2 + fig6)
- V_MIN/V_MAX still saturate — structural limitation of bounded Detoxify reward.
- Fixed-penalty ablation still open.

## Slide 19 — Claims
- Split carefully: what we know vs what we infer.
- Negative results are thesis-worthy (reward hacking, eval bugs, collapse).

## Slide 20 — Next steps
- Priority 1: matched re-run in `ppo-minmax`, both β=0.2, sample eval.
- Mention defaulting minmax KL to 0.2 in code to prevent recurrence.

## Slide 21 — Discussion
- Ask: is 2% vs 3% enough to continue minmax line of work?
- Ask: structure thesis around debugging narrative + fair rerun?
- Phase 2 probe timeline if relevant.

---

## Anticipated questions

**Q: Why did minmax look better at 0% harm before?**  
A: Policy collapsed to non-toxic garbage/Advertisements. Metric gaming, not safety.

**Q: Is minmax better than baseline?**  
A: Under fair sample eval with β=0.2: 2.0% vs 3.0% harm, one seed. Suggestive, not conclusive. Need matched re-run + second seed.

**Q: Was the whole project wasted?**  
A: No — we documented failure modes of proxy rewards + KL confounding + eval protocol. That's core thesis methodology for ROSARL→LLM transfer.

**Q: What about the proposal numbers?**  
A: Invalid due to eval bugs. Do not cite.
