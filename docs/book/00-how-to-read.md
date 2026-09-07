# How to read this book

## Two layers, both from the worklogs

| Layer | What it is | Where |
|---|---|---|
| **Worklogs** | Full chronological Did / Found / Concluded — every dead end kept | [/book/worklogs/phase1.md](/book/worklogs/phase1.md), [/book/worklogs/phase2.md](/book/worklogs/phase2.md) |
| **Guided chapters** | Same facts, textbook order, with diagrams | [/book/phase1/](/book/phase1/README.md), [/book/phase2/](/book/phase2/README.md) |

Canonical files in git:

- `ppo_minmax_experiment/worklog.md`
- `safe-rlhf/worklog.md`

The Pages site mirrors them under `docs/book/worklogs/` (GitHub Pages can only serve `docs/`). After editing a worklog, run:

```bash
python scripts/sync_worklogs_to_docs.py
```

If a guided chapter and the worklog disagree, **trust the worklog**, then fix the chapter.

## Suggested paths

**Supervisor (full story)**  
Worklog Phase 1 → Worklog Phase 2, or guided chapters in sidebar order.

**Examiner (claims only)**  
[/book/10-claims.md](/book/10-claims.md) → skim [/book/phase1/04-advertisements-collapse.md](/book/phase1/04-advertisements-collapse.md) and [/book/phase2/05-run-a.md](/book/phase2/05-run-a.md)–[06](/book/phase2/06-run-b.md).

**Continuing Run C**  
[/book/worklogs/phase2.md](/book/worklogs/phase2.md) Session 18 → [/book/phase2/07-run-c.md](/book/phase2/07-run-c.md).

## Conventions in guided chapters

| Marker | Meaning |
|---|---|
| Finding callout | Empirical result solid enough to cite |
| Caution callout | Easy misread or corrected earlier claim |
| Mermaid | Architecture / experimental logic |
| Session headers | Map 1:1 onto worklog sessions |

Update policy: [/book/a-updating.md](/book/a-updating.md).
