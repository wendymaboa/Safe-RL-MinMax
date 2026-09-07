<div class="cover-kicker">Research textbook · driven by the worklogs</div>

# Safe Reinforcement Learning with a MinMax Penalty

This site is built from two lab notebooks — mirrored in full, then retold as guided chapters with diagrams.

## Start with the worklogs

| Phase | Canonical file | On this site |
|---|---|---|
| **1** — GPT-2 + Detoxify + MinMax | `ppo_minmax_experiment/worklog.md` | [Read full Phase 1 worklog](/book/worklogs/phase1.md) |
| **2** — Qwen + LoRA + Safe RLHF | `safe-rlhf/worklog.md` | [Read full Phase 2 worklog](/book/worklogs/phase2.md) |

Every session (Did / Found / Concluded) is there. Guided chapters below are a pedagogical rewrite of **the same record**, not a second set of results.

```mermaid
flowchart TB
  W1[Phase 1 worklog] --> C1[Guided Phase 1 chapters]
  W2[Phase 2 worklog] --> C2[Guided Phase 2 chapters]
  C1 --> S[Claims & open questions]
  C2 --> S
```

## Status at a glance

| Stage / run | Status | Worklog | Guided chapter |
|---|---|---|---|
| Phase 1 closed | MinMax ≈ PPO+KL at matched β; Detoxify gameable | [worklog](/book/worklogs/phase1.md) | [phase1/](/book/phase1/README.md) |
| Phase 2 Stages 0–4 | Env, LoRA, PPO smoke, cost detector | [worklog](/book/worklogs/phase2.md) | [01](/book/phase2/01-reset-and-cluster.md)–[04](/book/phase2/04-reward-cost-and-gpu.md) |
| Run A | Complete — refusal eroded | Sessions 14–15 | [05](/book/phase2/05-run-a.md) |
| Run B | Complete — gate helps on probe; avg cost drifts | Sessions 16–17 | [06](/book/phase2/06-run-b.md) |
| Run C | Implemented; not trained | Session 18 | [07](/book/phase2/07-run-c.md) |
| Claims (A+B only) | Current verdict | — | [10](/book/10-claims.md) |

## Guided reading order

1. [How to read this book](/book/00-how-to-read.md)
2. [Phase 1 worklog](/book/worklogs/phase1.md) *or* [Phase 1 chapters](/book/phase1/README.md)
3. [Phase 2 worklog](/book/worklogs/phase2.md) *or* [Phase 2 chapters](/book/phase2/README.md)
4. [What we can claim](/book/10-claims.md)

After editing either worklog in the repo: `python scripts/sync_worklogs_to_docs.py` then commit the mirror under `docs/book/worklogs/`.
