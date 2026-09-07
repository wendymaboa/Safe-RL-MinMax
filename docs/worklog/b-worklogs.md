# Appendix B — Raw worklogs

The worklog is curated. The **audit trail** is the worklog. On GitHub Pages, start with the Docsify mirrors (they are what `/worklog/…` links can actually load). Canonical files in the experiment folders remain the edit targets — sync with `python scripts/sync_worklogs_to_docs.py` (see [/worklog/a-updating.md](/worklog/a-updating.md)).

## Mirrored worklogs (read these first)

| Mirror (Docsify / Pages) | Scope | Worklog map |
|---|---|---|
| [/worklog/worklogs/phase1.md](/worklog/worklogs/phase1.md) | Phase 1 Sessions 1–18 | [/worklog/phase1/](/worklog/phase1/README.md) |
| [/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md) | Phase 2 Sessions 1–18+ | [/worklog/phase2/](/worklog/phase2/README.md) |

## Canonical sources (edit in the repo)

| File in repo | Same content as |
|---|---|
| `ppo_minmax_experiment/worklog.md` | [/worklog/worklogs/phase1.md](/worklog/worklogs/phase1.md) |
| `safe-rlhf/worklog.md` | [/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md) |

GitHub blob links (for browsing outside Docsify):

- [Phase 1 on GitHub](https://github.com/wendymaboa/Safe-RL-MinMax/blob/main/ppo_minmax_experiment/worklog.md)
- [Phase 2 on GitHub](https://github.com/wendymaboa/Safe-RL-MinMax/blob/main/safe-rlhf/worklog.md)

## Phase 1 session → chapter

| Sessions | Chapter |
|---|---|
| 1–7 | [/worklog/phase1/01-pilot-and-bugs.md](/worklog/phase1/01-pilot-and-bugs.md) |
| 8–9 | [/worklog/phase1/02-saturation.md](/worklog/phase1/02-saturation.md) |
| 10–13 | [/worklog/phase1/03-design-choices.md](/worklog/phase1/03-design-choices.md) |
| 14–18 | [/worklog/phase1/04-advertisements-collapse.md](/worklog/phase1/04-advertisements-collapse.md) |

## Phase 2 session → chapter

| Sessions | Chapter |
|---|---|
| 1–3 | [/worklog/phase2/01-reset-and-cluster.md](/worklog/phase2/01-reset-and-cluster.md) |
| 4–7 | [/worklog/phase2/02-lora-and-mkl.md](/worklog/phase2/02-lora-and-mkl.md) |
| 8–10 | [/worklog/phase2/03-template-and-ppo-loop.md](/worklog/phase2/03-template-and-ppo-loop.md) |
| 11–13 | [/worklog/phase2/04-reward-cost-and-gpu.md](/worklog/phase2/04-reward-cost-and-gpu.md) |
| 14–15 | [/worklog/phase2/05-run-a.md](/worklog/phase2/05-run-a.md) |
| 16–17 | [/worklog/phase2/06-run-b.md](/worklog/phase2/06-run-b.md) |
| 18 | [/worklog/phase2/07-run-c.md](/worklog/phase2/07-run-c.md) |

## Other durable artifacts

- `safe-rlhf/results/stage5_runA/` — archived Run A metrics & generations
- `docs/Slides/` — presentation decks
- `docs/phase_plan.pdf`, `docs/mscluster_commands.pdf` — planning / ops PDFs
