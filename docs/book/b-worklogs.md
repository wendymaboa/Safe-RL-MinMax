# Appendix B — Raw worklogs

The textbook is curated. The **audit trail** is the worklog. On GitHub Pages, start with the Docsify mirrors (they are what `/book/…` links can actually load). Canonical files in the experiment folders remain the edit targets — sync with `python scripts/sync_worklogs_to_docs.py` (see [/book/a-updating.md](/book/a-updating.md)).

## Mirrored worklogs (read these first)

| Mirror (Docsify / Pages) | Scope | Textbook map |
|---|---|---|
| [/book/worklogs/phase1.md](/book/worklogs/phase1.md) | Phase 1 Sessions 1–18 | [/book/phase1/](/book/phase1/README.md) |
| [/book/worklogs/phase2.md](/book/worklogs/phase2.md) | Phase 2 Sessions 1–18+ | [/book/phase2/](/book/phase2/README.md) |

## Canonical sources (edit in the repo)

| File in repo | Same content as |
|---|---|
| `ppo_minmax_experiment/worklog.md` | [/book/worklogs/phase1.md](/book/worklogs/phase1.md) |
| `safe-rlhf/worklog.md` | [/book/worklogs/phase2.md](/book/worklogs/phase2.md) |

GitHub blob links (for browsing outside Docsify):

- [Phase 1 on GitHub](https://github.com/wendymaboa/Safe-RL-MinMax/blob/main/ppo_minmax_experiment/worklog.md)
- [Phase 2 on GitHub](https://github.com/wendymaboa/Safe-RL-MinMax/blob/main/safe-rlhf/worklog.md)

## Phase 1 session → chapter

| Sessions | Chapter |
|---|---|
| 1–7 | [/book/phase1/01-pilot-and-bugs.md](/book/phase1/01-pilot-and-bugs.md) |
| 8–9 | [/book/phase1/02-saturation.md](/book/phase1/02-saturation.md) |
| 10–13 | [/book/phase1/03-design-choices.md](/book/phase1/03-design-choices.md) |
| 14–18 | [/book/phase1/04-advertisements-collapse.md](/book/phase1/04-advertisements-collapse.md) |

## Phase 2 session → chapter

| Sessions | Chapter |
|---|---|
| 1–3 | [/book/phase2/01-reset-and-cluster.md](/book/phase2/01-reset-and-cluster.md) |
| 4–7 | [/book/phase2/02-lora-and-mkl.md](/book/phase2/02-lora-and-mkl.md) |
| 8–10 | [/book/phase2/03-template-and-ppo-loop.md](/book/phase2/03-template-and-ppo-loop.md) |
| 11–13 | [/book/phase2/04-reward-cost-and-gpu.md](/book/phase2/04-reward-cost-and-gpu.md) |
| 14–15 | [/book/phase2/05-run-a.md](/book/phase2/05-run-a.md) |
| 16–17 | [/book/phase2/06-run-b.md](/book/phase2/06-run-b.md) |
| 18 | [/book/phase2/07-run-c.md](/book/phase2/07-run-c.md) |

## Other durable artifacts

- `safe-rlhf/results/stage5_runA/` — archived Run A metrics & generations
- `docs/Slides/` — presentation decks
- `docs/phase_plan.pdf`, `docs/mscluster_commands.pdf` — planning / ops PDFs
