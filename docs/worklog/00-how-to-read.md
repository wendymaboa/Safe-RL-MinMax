# How to read this worklog

## Two layers, both from the worklogs

| Layer | What it is | Where |
|---|---|---|
| **Worklogs** | Full chronological Did / Found / Concluded — every dead end kept | [/worklog/worklogs/phase1.md](/worklog/worklogs/phase1.md), [/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md) |
| **Guided notes** | Same facts, worklog order, with diagrams | [/worklog/phase1/](/worklog/phase1/README.md), [/worklog/phase2/](/worklog/phase2/README.md) |

Canonical files in git:

- `ppo_minmax_experiment/worklog.md`
- `safe-rlhf/worklog.md`

The Pages site mirrors them under `docs/worklog/worklogs/` (GitHub Pages can only serve `docs/`). After editing a worklog, run:

```bash
python scripts/sync_worklogs_to_docs.py
```

If a guided note and the worklog disagree, **trust the worklog**, then fix the chapter.

## Suggested paths

**Supervisor (full story)**  
Worklog Phase 1 → Worklog Phase 2, or guided notes in sidebar order.

**Examiner (claims only)**  
[/worklog/10-claims.md](/worklog/10-claims.md) → skim [/worklog/phase1/04-advertisements-collapse.md](/worklog/phase1/04-advertisements-collapse.md) and [/worklog/phase2/05-run-a.md](/worklog/phase2/05-run-a.md)–[06](/worklog/phase2/06-run-b.md).

**Continuing Run C**  
[/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md) Session 18 → [/worklog/phase2/07-run-c.md](/worklog/phase2/07-run-c.md).

## Conventions in guided notes

| Marker | Meaning |
|---|---|
| Finding callout | Empirical result solid enough to cite |
| Caution callout | Easy misread or corrected earlier claim |
| Mermaid | Architecture / experimental logic |
| Session headers | Map 1:1 onto worklog sessions |

Update policy: [/worklog/a-updating.md](/worklog/a-updating.md).
