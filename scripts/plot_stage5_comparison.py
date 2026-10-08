"""Analysis plots for Stage 5. Decile means, one metric per axes, runs overlaid.

Sources, not live TensorBoard:
  Run A reward: results/stage5_runA (copied into plot_stage5_results.py)
  Run C: dump_tb job 50802, 2026-09-08
  Run D: dump_tb of output/stage5_runD, 2026-10-05
  Run B cost deciles were never archived (endpoints only, about -2.6 to +1.7).
  Run D mean train/cost is missing the 20-50% deciles in the captured dump.
  Run E has not been plotted.

Writes PNGs under docs/worklog/assets/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'worklog' / 'assets' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

DECILES = [
    '0-10', '10-20', '20-30', '30-40', '40-50',
    '50-60', '60-70', '70-80', '80-90', '90-100',
]
X = np.arange(len(DECILES))

REWARD_A = [0.269, 1.094, 1.444, 1.447, 1.447, 1.508, 1.406, 1.355, 1.371, 1.404]
REWARD_C = [0.429, 1.326, 1.856, 1.824, 1.669, 1.427, 1.344, 1.214, 1.208, 1.214]
COST_C = [-2.658, -1.843, -1.097, -0.092, 1.296, 1.669, 1.938, 2.191, 2.181, 2.377]
UNSAFE_C = [0.140, 0.250, 0.314, 0.407, 0.544, 0.558, 0.587, 0.625, 0.619, 0.640]
R_UNSAFE_C = [-7.83, -8.895, -8.895, -9.534, -9.918, -9.918, -9.918, -9.918, -9.918, -9.918]

# Run D dump, 2026-10-05. Mean cost lines 20-50% were absent from the capture.
COST_D = [-2.768, -1.983, np.nan, np.nan, np.nan, 1.735, 1.714, 2.373, 2.373, 2.292]
COST_SAFE_D = [-3.608, -3.909, -3.832, -3.441, -3.060, -2.521, -2.412, -2.308, -2.322, -2.460]
COST_UNSAFE_D = [2.125, 3.737, 3.945, 4.621, 4.653, 5.017, 4.779, 5.120, 5.136, 5.142]
UNSAFE_D = [0.1191, 0.2406, 0.2854, 0.4198, 0.5177, 0.5613, 0.5778, 0.6344, 0.6344, 0.6297]
R_UNSAFE_D = [-8.403, -13.07, -14.25, -15.51, -15.47, -15.51, -15.23, -16.53, -16.78, -16.74]
R_BASE_D = [-8.076, -8.438, -8.773, -8.773, -8.773, -8.773, -8.773, -9.305, -9.414, -9.414]
SEVERITY_D = [0.4269, 0.6713, 0.6899, 0.7775, 0.7629, 0.7772, 0.7357, 0.7766, 0.7824, 0.7786]
C_SCALE_D = [3.479, 3.866, 4.046, 4.259, 4.384, 4.529, 4.659, 4.710, 4.772, 4.832]

CKPTS = ['base', '50', '250', '500', '750', '950']
PROBE_A = [-3.531, -2.422, 1.469, 6.625, 5.344, 6.625]
PROBE_B = [-3.531, -4.062, 1.695, 3.812, 2.641, -0.773]

COL_C = '#1f4e79'
COL_D = '#b85c38'
COL_A = '#5c5346'
COL_SAFE = '#2f6f4e'
COL_UNSAFE = '#8c2f39'
COL_BASE = '#6b6258'
COL_GRID = '#e6e1d8'


def style(ax):
    ax.set_facecolor('white')
    ax.grid(True, color=COL_GRID, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8, colors='#3d342c')
    ax.xaxis.label.set_color('#3d342c')
    ax.yaxis.label.set_color('#3d342c')
    ax.title.set_color('#1c1612')


def finish(fig, ax, name: str) -> None:
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    save(fig, name)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {path.relative_to(ROOT)}')


def decile_axis(ax):
    ax.set_xticks(X)
    ax.set_xticklabels(DECILES, rotation=30, ha='right')
    ax.set_xlabel('Training progress (decile of steps)')


def fig_unsafe():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    style(ax)
    ax.plot(X, UNSAFE_C, 'o-', color=COL_C, lw=2, ms=5, label='Run C  unsafe_rate')
    ax.plot(X, UNSAFE_D, 's-', color=COL_D, lw=2, ms=5, label='Run D  unsafe_rate')
    decile_axis(ax)
    ax.set_ylabel('Fraction of the batch with cost > 0')
    ax.set_ylim(0, 1)
    ax.set_title('Fraction gated — Run C (flat MinMax) vs Run D (cost-scaled)')
    finish(fig, ax, 'stage5_unsafe_rate_C_vs_D.png')


def fig_penalty():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    style(ax)
    ax.plot(X, R_UNSAFE_C, 'o-', color=COL_C, lw=2, ms=5, label='Run C  R_unsafe (flat)')
    ax.plot(X, R_BASE_D, '^-', color=COL_BASE, lw=1.6, ms=5, label='Run D  unscaled gap')
    ax.plot(X, R_UNSAFE_D, 's-', color=COL_D, lw=2, ms=5, label='Run D  applied penalty')
    ax.axhline(-2.0, color=COL_A, ls='--', lw=1.2, label='Run B fixed penalty (−2)')
    decile_axis(ax)
    ax.set_ylabel('Reward replacement on gated replies')
    ax.set_title('Penalty actually applied — B, C, and D')
    finish(fig, ax, 'stage5_penalty_B_C_D.png')


def fig_cost_split():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    style(ax)
    ax.plot(X, COST_C, 'o-', color=COL_C, lw=2, ms=5, label='Run C  mean cost')
    ax.plot(X, COST_D, 's-', color=COL_D, lw=2, ms=5, label='Run D  mean cost (20–50% not captured)')
    ax.plot(X, COST_SAFE_D, 'v-', color=COL_SAFE, lw=1.6, ms=5, label='Run D  cost | cost ≤ 0')
    ax.plot(X, COST_UNSAFE_D, 'D-', color=COL_UNSAFE, lw=1.6, ms=4, label='Run D  cost | cost > 0')
    ax.axhline(0, color='#9a9186', ls='--', lw=1)
    decile_axis(ax)
    ax.set_ylabel('Beaver cost (decile mean)')
    ax.set_title('Mean cost and the Run D split across the gate')
    finish(fig, ax, 'stage5_cost_split_C_D.png')


def fig_scale():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax in axes:
        style(ax)
        decile_axis(ax)
    axes[0].plot(X, COST_UNSAFE_D, 'D-', color=COL_UNSAFE, lw=1.6, ms=4, label='cost | cost > 0')
    axes[0].plot(X, C_SCALE_D, 'o-', color=COL_D, lw=2, ms=5, label='c_scale')
    axes[0].set_ylabel('Cost units')
    axes[0].set_title('Run D — scale chasing the unsafe mean')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    axes[1].plot(X, SEVERITY_D, 'o-', color=COL_D, lw=2, ms=5)
    axes[1].axhline(1.0, color='#9a9186', ls='--', lw=1, label='cap')
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel('severity = min(excess / c_scale, 1)')
    axes[1].set_title('Run D — severity of gated replies')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'stage5_runD_scale_severity.png')


def fig_reward():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    style(ax)
    ax.plot(X, REWARD_A, 'o-', color=COL_A, lw=2, ms=5, label='Run A  reward only')
    ax.plot(X, REWARD_C, 's-', color=COL_C, lw=2, ms=5, label='Run C  MinMax')
    decile_axis(ax)
    ax.set_ylabel('train/reward (decile mean)')
    ax.set_title('Helpfulness score — Run A vs Run C')
    finish(fig, ax, 'stage5_reward_A_vs_C_detail.png')


def fig_probe():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax)
    xpos = np.arange(len(CKPTS))
    w = 0.36
    ax.bar(xpos - w / 2, PROBE_A, width=w, color=COL_A, label='Run A')
    ax.bar(xpos + w / 2, PROBE_B, width=w, color=COL_C, label='Run B')
    ax.axhline(0, color='#9a9186', ls='--', lw=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(CKPTS)
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('beaver-7b-unified-cost')
    ax.set_title('Lock-picking probe — Run A vs Run B (Session 17 rescore)')
    finish(fig, ax, 'stage5_probe_cost_A_vs_B_detail.png')


def main():
    fig_unsafe()
    fig_penalty()
    fig_cost_split()
    fig_scale()
    fig_reward()
    fig_probe()


if __name__ == '__main__':
    main()
