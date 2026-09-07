# Appendix A — Updating this book

This textbook should lag the worklog by **hours to days**, not months. Prefer occasional curated edits over auto-generating chapters from sessions.

## When to update

| Event | Edit |
|---|---|
| New session changes a claim | Rewrite the affected chapter; bump the home “Status at a glance” table |
| Run C finishes | Fill Chapter 9 results; refresh Chapters 10–11 |
| Corrected earlier conclusion | Update chapter in place; add a short “Correction” note pointing to the worklog session |
| New diagram-worthy architecture | Add/adjust Mermaid in the relevant chapter |

## What not to do

- Do **not** paste raw “Did / Found / Concluded” blocks into chapters.
- Do **not** silently delete wrong claims from the **worklog**.
- Do **not** put multi-MB checkpoints or TensorBoard event files in `docs/`.

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

Worklog session commits stay separate from textbook polish.
