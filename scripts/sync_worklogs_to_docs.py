#!/usr/bin/env python3
"""Sync repo worklogs into docs/book/worklogs/ for GitHub Pages / Docsify.

GitHub Pages only serves the docs/ tree, so the textbook cannot link to
../ppo_minmax_experiment/worklog.md at runtime. Edit the canonical worklogs
in the experiment folders, then run this script (or rely on CI) before push.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    {
        'src': ROOT / 'ppo_minmax_experiment' / 'worklog.md',
        'dst': ROOT / 'docs' / 'book' / 'worklogs' / 'phase1.md',
        'title': 'Phase 1 worklog — GPT-2 + Detoxify + MinMax',
        'canonical': 'ppo_minmax_experiment/worklog.md',
    },
    {
        'src': ROOT / 'safe-rlhf' / 'worklog.md',
        'dst': ROOT / 'docs' / 'book' / 'worklogs' / 'phase2.md',
        'title': 'Phase 2 worklog — Safe RLHF on Qwen + LoRA',
        'canonical': 'safe-rlhf/worklog.md',
    },
]


def strip_leading_h1(text: str) -> str:
    return re.sub(r'^# .+?\n\n', '', text, count=1, flags=re.M)


def mirror(src: Path, dst: Path, title: str, canonical: str) -> None:
    body = strip_leading_h1(src.read_text(encoding='utf-8'))
    header = (
        f'> **Source of truth:** `{canonical}` in the repo. This page is a Docsify '
        f'mirror for GitHub Pages (the live `docs/` tree cannot fetch files outside '
        f'itself). Edit the repo worklog, then run `python scripts/sync_worklogs_to_docs.py`.\n\n'
        f'# {title}\n\n'
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(header + body, encoding='utf-8', newline='\n')
    print(f'wrote {dst.relative_to(ROOT)} from {src.relative_to(ROOT)}')


def main() -> None:
    for item in TARGETS:
        mirror(item['src'], item['dst'], item['title'], item['canonical'])


if __name__ == '__main__':
    main()
