"""Plot Stage 5 comparison at the stage where Run C's R_unsafe stopped moving.

R_unsafe locks in the 40–50% decile (~steps 425–530) at ≈ −9.92.
Checkpoint-500 is the first A/B/C-matched save after that lock.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'worklog' / 'assets' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

COL_A = '#8C7753'
COL_B = '#5E4630'
COL_C = '#331B0D'
COL_MUTED = '#9C8A72'
COL_GRID = '#E8E0D4'

# Lock-picking probe cost at checkpoint-500 (Session 17). C filled after rescore.
PROBE_COST_500 = {
    'Run A': 6.625,
    'Run B': 3.812,
    'Run C': None,  # set after: python scripts/rescore_lockpicking.py (with RUN_C)
}

# Training decile 40–50% ≈ lock window
REWARD_LOCK = {'Run A': 1.447, 'Run C': 1.669}  # B not archived locally
PENALTY_LOCK = {'Run B': -2.0, 'Run C': -9.918}
COST_TRAIN_LOCK = {'Run C': 1.296}  # mean train/cost in 40–50%


def style(ax):
    ax.set_facecolor('white')
    ax.grid(True, axis='y', color=COL_GRID, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#5E4630', labelsize=9)
    ax.xaxis.label.set_color('#5E4630')
    ax.yaxis.label.set_color('#5E4630')
    ax.title.set_color('#331B0D')


def fig_lock_stage_comparison(probe_c: float | None = None) -> Path:
    """Main figure: A/B/C at R_unsafe lock (ckpt-500)."""
    costs = dict(PROBE_COST_500)
    if probe_c is not None:
        costs['Run C'] = probe_c

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    # --- Left: probe cost at checkpoint-500 ---
    ax = axes[0]
    style(ax)
    names = ['Run A', 'Run B', 'Run C']
    colors = [COL_A, COL_B, COL_C]
    vals = [costs[n] for n in names]
    xpos = np.arange(len(names))
    for i, (name, val, color) in enumerate(zip(names, vals, colors)):
        if val is None:
            ax.bar(i, 0.15, color='none', edgecolor=color, linewidth=1.6, linestyle='--', zorder=2)
            ax.text(i, 0.35, 'rescore\npending', ha='center', va='bottom', fontsize=8, color=color)
        else:
            ax.bar(i, val, color=color, zorder=2, width=0.65)
            ax.text(i, val + 0.15, f'{val:+.2f}', ha='center', va='bottom', fontsize=9, color='#331B0D')
    ax.axhline(0, color=COL_MUTED, linestyle='--', linewidth=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_ylabel('beaver-7b-unified-cost  (lower = safer)')
    ax.set_title('Lock-picking probe cost @ checkpoint-500')
    ymax = max(v for v in vals if v is not None) + 1.2
    ax.set_ylim(-0.5, ymax)

    # --- Right: gated penalty at lock ---
    ax = axes[1]
    style(ax)
    pen_names = ['Run B\nfixed', 'Run C\nMinMax']
    pen_vals = [PENALTY_LOCK['Run B'], PENALTY_LOCK['Run C']]
    pen_colors = [COL_B, COL_C]
    bars = ax.bar(pen_names, pen_vals, color=pen_colors, width=0.55, zorder=2)
    for bar, v in zip(bars, pen_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.35, f'{v:.2f}',
                ha='center', va='top', fontsize=9, color='white', fontweight='bold')
    ax.axhline(0, color=COL_MUTED, linestyle='--', linewidth=1)
    ax.set_ylabel('Gated replacement reward')
    ax.set_title(r'Penalty when cost $> 0$  (at C lock)')
    ax.set_ylim(-11, 0.5)

    fig.suptitle(
        r'At Run C $R_{\mathrm{unsafe}}$ lock  ·  ≈40–50% training  ·  checkpoint-500',
        color='#331B0D',
        fontsize=12,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()
    path = OUT / 'stage5_at_runC_lock.png'
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {path.relative_to(ROOT)}')
    return path


def fig_lock_stage_reward() -> Path:
    """Secondary: train/reward at the same decile (A vs C)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    style(ax)
    names = list(REWARD_LOCK.keys())
    vals = [REWARD_LOCK[n] for n in names]
    colors = [COL_A, COL_C]
    bars = ax.bar(names, vals, color=colors, width=0.55, zorder=2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.04, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9, color='#331B0D')
    ax.set_ylabel('train/reward (decile mean)')
    ax.set_title('Mean reward at 40–50%  (C lock window)')
    ax.set_ylim(0, max(vals) + 0.35)
    # annotate B missing
    ax.text(0.98, 0.02, 'Run B reward deciles not archived locally',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color=COL_MUTED)
    fig.tight_layout()
    path = OUT / 'stage5_reward_at_runC_lock.png'
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {path.relative_to(ROOT)}')
    return path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--probe-cost-c', type=float, default=None,
                   help='Run C lock-picking cost at checkpoint-500 from rescore')
    args = p.parse_args()
    fig_lock_stage_comparison(probe_c=args.probe_cost_c)
    fig_lock_stage_reward()


if __name__ == '__main__':
    main()
