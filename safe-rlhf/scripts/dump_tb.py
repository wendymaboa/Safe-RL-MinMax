"""Print scalar metrics from TensorBoard event files as text.

The trainers log with `--log_type tensorboard`, so training metrics never appear in
stdout or the SLURM .out file. This dumps them to the terminal instead, which is what
you want over SSH.

    python scripts/dump_tb.py output/smoke
    python scripts/dump_tb.py output/smoke --full          # every step, not a summary
    python scripts/dump_tb.py output/smoke --tag train/reward
"""

from __future__ import annotations

import argparse
import glob
import math
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_dirs(root: str) -> list[str]:
    """Directories containing tfevents files, innermost first."""
    matches = glob.glob(os.path.join(root, '**', '*tfevents*'), recursive=True)
    return sorted({os.path.dirname(m) for m in matches})


def main() -> None:
    parser = argparse.ArgumentParser(description='Dump TensorBoard scalars as text.')
    parser.add_argument('logdir', type=str, help='Directory to search for event files.')
    parser.add_argument('--tag', type=str, default=None, help='Only this scalar tag.')
    parser.add_argument('--full', action='store_true', help='Print every step.')
    parser.add_argument('--trend', action='store_true',
                        help='Print 10 window means across the run. Distinguishes a steady '
                             'decline from an early crash followed by a plateau — which '
                             'imply different causes.')
    args = parser.parse_args()

    event_dirs = find_event_dirs(args.logdir)
    if not event_dirs:
        raise SystemExit(f'No tfevents files found under {args.logdir!r}')

    for event_dir in event_dirs:
        print(f'\n=== {event_dir} ===')
        acc = EventAccumulator(event_dir)
        acc.Reload()

        tags = acc.Tags().get('scalars', [])
        if args.tag:
            tags = [t for t in tags if t == args.tag]
        if not tags:
            print('  (no scalar tags)')
            continue

        for tag in sorted(tags):
            events = acc.Scalars(tag)
            values = [e.value for e in events]
            finite = [v for v in values if math.isfinite(v)]
            bad = len(values) - len(finite)

            if args.trend:
                # Ten equal windows across the run, so the SHAPE of the change is visible
                # rather than only its endpoints.
                n = len(finite)
                if n < 10:
                    print(f'\n  {tag}: too few points for a trend')
                    continue
                size = n // 10
                means = [
                    sum(finite[i * size:(i + 1) * size]) / size
                    for i in range(10)
                ]
                lo, hi = min(means), max(means)
                span = (hi - lo) or 1.0
                print(f'\n  {tag}')
                for i, m in enumerate(means):
                    bar = '#' * int(1 + 38 * (m - lo) / span)
                    print(f'    {i * 10:>3}-{(i + 1) * 10:<3}%  {m:+11.4g}  {bar}')
            elif args.full:
                print(f'\n  {tag}')
                for e in events:
                    print(f'    step {e.step:>6}  {e.value:+.6f}')
            else:
                if finite:
                    # Report windowed MEANS, not first/last. Single steps are noisy —
                    # per-step reward can swing several units at small batch sizes — so
                    # comparing two individual points reads noise as trend.
                    w = max(1, len(finite) // 10)
                    head = sum(finite[:w]) / w
                    tail = sum(finite[-w:]) / w
                    delta = tail - head
                    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
                    # %g, not %f: learning rates like 1e-5 round to 0.0000 in fixed point
                    # and look like a bug when they are perfectly correct.
                    summary = (
                        f'n={len(values):<4} '
                        f'first{w}={head:+.4g}  last{w}={tail:+.4g}  '
                        f'{arrow}{abs(delta):.4g}   '
                        f'[{min(finite):+.4g}, {max(finite):+.4g}]'
                    )
                else:
                    summary = f'n={len(values):<4} (no finite values)'
                flag = f'  ** {bad} non-finite **' if bad else ''
                print(f'  {tag:<34} {summary}{flag}')


if __name__ == '__main__':
    main()
