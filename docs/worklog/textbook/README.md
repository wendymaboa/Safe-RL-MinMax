# Textbook — become fluent in this research

This track is **concept-first**. The [worklogs](/worklog/worklogs/phase2.md) are the lab notebook; these chapters are what an expert keeps in their head so the notebook makes sense.

## Who this is for

You already ran Stage 5. You do not need another “hello LLM” blog. You need to be able to:

- Explain **reward vs cost** without hesitation  
- Derive why gating on reward would punish refusals  
- Sketch the **PPO-RLHF** loop and where MinMax sits in it  
- Defend (or retract) MinMax as a **fair contrast to a fixed gate**  
- Read your own curves and probe text like a reviewer  

## Path (read in order)

| # | Chapter | What you should be able to do after |
|---|---|---|
| 1 | [Language models & transformers](/worklog/textbook/01-transformers.md) | Place Qwen/GPT-2 in the stack; say why ChatML mattered |
| 2 | [Preferences → scalar models](/worklog/textbook/02-preferences-and-scores.md) | Bradley–Terry; why only score *differences* are meaningful |
| 3 | [Helpfulness vs harmlessness](/worklog/textbook/03-reward-vs-cost.md) | Never confuse RM and CM again |
| 4 | [RLHF with PPO](/worklog/textbook/04-rlhf-ppo.md) | Actor, reference, critic, KL; where the scalar enters |
| 5 | [Safe RLHF (PKU)](/worklog/textbook/05-safe-rlhf.md) | What Dai et al. did; what you reused vs reinvented |
| 6 | [MinMax — mechanism & validity](/worklog/textbook/06-minmax.md) | State the claim, the null, and when MinMax is vacuous |
| 7 | [Your A/B/C instrument](/worklog/textbook/07-abc-design.md) | Argue isolation like a methods section |
| 8 | [Failure modes](/worklog/textbook/08-failure-modes.md) | Hacking, saturation, drift, length Goodhart |
| 9 | [Reading *your* results](/worklog/textbook/09-reading-results.md) | Expert narration of A, B, C |
| 10 | [Papers & further study](/worklog/textbook/10-papers.md) | What to extract from each paper |

**Orientation one-pager** (if you only have 10 minutes): [Foundations](/worklog/fundamentals.md).

## How to study

1. Read a chapter once without the codebase.  
2. Open the file it names (`ppo_cost_gate/trainer.py`, etc.) and find the 10–20 lines that match the diagram.  
3. Explain that chapter out loud in ≤3 minutes. If you can’t, re-read the “expert checklist” at the end.

After the textbook: [claims](/worklog/10-claims.md) → Phase 2 worklog Sessions 15–19.
