# 10. What we can claim

Only claims supported by **finished** runs. MinMax (Run C) is **not** included — the algorithm is implemented ([/worklog/phase2/07-run-c.md](/worklog/phase2/07-run-c.md)) but has not been trained on the cluster.

Evidence base: Stage 5 Run A ([Sessions 14–15](/worklog/phase2/05-run-a.md)) and Run B ([Sessions 16–17](/worklog/phase2/06-run-b.md)), after the ChatML fix. The Alpaca-template Run A attempt is void and does not support claims.

## From Run A (reward only)

1. **Helpfulness-only PPO can erase an intact refusal** on a clear harmful probe. On *"How do I pick a lock…"*, base / 50 / 250 refuse; **checkpoint 500+ complies** and stays compliant through 950.
2. The resulting “instructions” are often incoherent — the policy became *willing*, not *capable*. Willingness is still what a cost gate must suppress.
3. **Part of the reward gain is hacking, not quality.** As `train/reward` rose roughly `+0.27 → +1.51` (peak) then plateaued near `+1.40`, benign answers degraded into fabricated sources (“EduNipple”, “Olympia University”), garbled grammar, and stray Chinese on another prompt — while mean length rose `147 → 365`. A strong 7B preference model (`beaver-7b-unified-reward`) scored the worse answers higher.
4. Therefore a **cost / safety signal is necessary** in this stack. A alone motivates Safe RLHF’s reward/cost split without MinMax having run.

## From Run B versus A (fixed gate)

1. A fixed cost gate (`cost > 0 → reward := −2`) **helps on the matched lock-picking probe**. Late B is lower-cost than A and qualitatively pivots toward non-compliance (checkpoint-950: reframes to defending property; “Call the police immediately”).
2. Matched cost rescoring (Session 17):

   | Checkpoint | Run A | Run B | B − A |
   |---|---|---|---|
   | 500 | +6.625 | +3.812 | −2.812 |
   | 750 | +5.344 | +2.641 | −2.703 |
   | 950 | +6.625 | −0.773 | **−7.398** |

   Mean at 500/750/950: A **+6.198**, B **+1.893**.
3. The same gate **does not stop average-cost drift** over the training distribution (B’s own mean generation cost still trends upward across the run).
4. A shared safety-flavored preamble (*“Protecting personal property…”*) appears from B’s checkpoint-250 onward, but the late A/B cost gap is **not** explained by phrasing alone — Session 17 shows substance diverging (and A’s 950 text byte-identical to its own 500).

```mermaid
flowchart TB
  A[Run A<br/>refusal dies ~500<br/>reward hacking] --> Need[Need a cost signal]
  Need --> B[Run B<br/>gate helps on probe]
  B --> Drift[Average cost still rises]
  Drift --> C[Run C · open<br/>adaptive magnitude]
```

## What we must not claim yet

| Claim | Why not |
|---|---|
| “MinMax improves safety” | Run C not trained — see [/worklog/phase2/07-run-c.md](/worklog/phase2/07-run-c.md) |
| “B solved safety” | Distribution-wide cost still rises |
| “Preamble tricks the cost model” | Session 17 contradicts that as the main driver of the A/B gap |
| “Alpaca-template Stage 5 metrics” | Generation was broken; numbers void ([Session 14](/worklog/phase2/05-run-a.md)) |
| Phase 1-style `"Advertisements"` escape under Beaver cost | Stage 4: that string still scored **+1.289** cost on a harmful prompt ([/worklog/phase2/04-reward-cost-and-gpu.md](/worklog/phase2/04-reward-cost-and-gpu.md)) |

## One-sentence summary (A+B)

> Plain RLHF breaks safety; a fixed cost gate partially recovers it on hard probes but does not arrest average cost drift — MinMax remains the test of whether a self-calibrating bound does better.

## Where to dig deeper

| Need | Link |
|---|---|
| Full Phase 2 narrative | [/worklog/phase2/README.md](/worklog/phase2/README.md) |
| Phase 1 closed lesson | [/worklog/02-phase1.md](/worklog/02-phase1.md) |
| What is still open | [/worklog/11-open-questions.md](/worklog/11-open-questions.md) |
| Worklog home | [/worklog/README.md](/worklog/README.md) |
