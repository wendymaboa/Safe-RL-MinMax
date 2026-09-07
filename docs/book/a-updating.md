# Appendix A — Updating this book

This textbook should lag the worklog by **hours to days**, not months. Prefer occasional curated edits over auto-generating chapters from sessions.

## Sync worklogs into Docsify (required for GitHub Pages)

GitHub Pages serves only the `docs/` tree. Live Docsify links **cannot** fetch `ppo_minmax_experiment/worklog.md` or `safe-rlhf/worklog.md` at runtime. Mirrors live at:

| Canonical (edit here) | Docsify mirror |
|---|---|
| `ppo_minmax_experiment/worklog.md` | [/book/worklogs/phase1.md](/book/worklogs/phase1.md) |
| `safe-rlhf/worklog.md` | [/book/worklogs/phase2.md](/book/worklogs/phase2.md) |

After editing either canonical worklog, regenerate the mirrors from the repo root:

```bash
python scripts/sync_worklogs_to_docs.py
```

What the script does:

1. Reads each canonical worklog.
2. Strips the leading `#` title (the mirror supplies its own).
3. Prepends a short banner naming the source of truth and the sync command.
4. Writes `docs/book/worklogs/phase1.md` and `docs/book/worklogs/phase2.md` with LF newlines.

Do **not** hand-edit the mirrors as the primary source — they will be overwritten on the next sync. Chapter prose still needs a human pass (see below); sync only keeps the raw trail readable on Pages.

## When to update

| Event | Edit |
|---|---|
| New session in a worklog | Run `sync_worklogs_to_docs.py`; expand or correct the matching chapter |
| New session changes a claim | Rewrite the affected chapter; bump the home “Status at a glance” table |
| Run C finishes | Fill Chapter 9 / [/book/phase2/07-run-c.md](/book/phase2/07-run-c.md) results; refresh Chapters 10–11 |
| Corrected earlier conclusion | Update chapter in place; add a short “Correction” note pointing to the worklog session |
| New diagram-worthy architecture | Add/adjust Mermaid in the relevant chapter |

## What not to do

- Do **not** paste raw “Did / Found / Concluded” blocks into chapters.
- Do **not** invent numbers that are not in the worklog.
- Do **not** silently delete wrong claims from the **worklog**.
- Do **not** put multi-MB checkpoints or TensorBoard event files in `docs/`.
- Do **not** treat the Docsify mirrors as editable originals.

## Local preview

From the repo root:

```bash
# any static server — Docsify is client-side
npx --yes serve docs
# open http://localhost:3000 (or the port serve prints)
```

Or with Python:

```bash
python -m http.server 4173 --directory docs
```

## GitHub Pages

1. Repo **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `main`, folder: `/docs`
4. Wait for `https://wendymaboa.github.io/Safe-RL-MinMax/`

`.nojekyll` is already present so `_sidebar.md` / `_navbar.md` are not ignored.

## Suggested commit habit

Keep textbook updates in their own commit when possible:

```text
docs: update textbook Ch. 9–10 after Run C results
```

Worklog session commits stay separate from textbook polish. If you changed a worklog, include the synced `docs/book/worklogs/*.md` in the same commit (or an immediate follow-up) so Pages stays in sync.
