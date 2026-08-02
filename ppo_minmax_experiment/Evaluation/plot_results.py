"""
Plot training curves and harm-rate bars from experiment outputs.

Usage:
    python Evaluation/plot_results.py --baseline-log experiments/seed_42_steps_1000/logs/baseline_training.csv \
        --minmax-log experiments/seed_42_steps_1000/logs/minmax_training.csv \
        --eval-summary experiments/seed_42_steps_1000/results/eval_1000_seed42_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Plot training and eval results.")
    p.add_argument("--experiment-dir", default="",
                   help="e.g. experiments/seed_42_steps_1000 — auto-fills log/summary paths")
    p.add_argument("--baseline-log", default="")
    p.add_argument("--minmax-log", default="")
    p.add_argument("--eval-summary", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--steps", type=int, default=1000, help="Used in output figure filenames.")
    p.add_argument("--figure-suffix", default="",
                   help="Suffix for figure filenames, e.g. _kl02_fair → fig1_training_curves_1000_kl02_fair.png")
    p.add_argument("--minmax-label", default="PPO + Minmax",
                   help="Legend label for the minmax condition.")
    p.add_argument("--eval-minmax-summary", default="",
                   help="Optional minmax-only eval summary; baseline rows come from --eval-summary.")
    args = p.parse_args()

    if args.experiment_dir:
        exp = Path(args.experiment_dir)
        parts = exp.name.split("_")
        if len(parts) >= 4 and parts[0] == "seed" and parts[2] == "steps":
            seed_tag = parts[1]
            args.steps = int(parts[3])
        else:
            seed_tag = "42"
        if not args.baseline_log:
            args.baseline_log = str(exp / "logs" / "baseline_training.csv")
        if not args.minmax_log:
            args.minmax_log = str(exp / "logs" / "minmax_training.csv")
        if not args.eval_summary:
            args.eval_summary = str(exp / "results" / f"eval_{args.steps}_seed{seed_tag}_summary.csv")
        if not args.output_dir:
            args.output_dir = str(exp / "results" / "figures")
    else:
        if not args.baseline_log:
            args.baseline_log = "results/baseline_training.csv"
        if not args.minmax_log:
            args.minmax_log = "results/minmax_training.csv"
        if not args.output_dir:
            args.output_dir = "results"

    return args


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_eval_summaries(baseline_summary: str, minmax_summary: str) -> list[dict]:
    """Baseline rows from the first file; minmax rows from the second."""
    baseline_rows = [r for r in load_csv(baseline_summary) if r["model"] == "baseline"]
    minmax_rows = [r for r in load_csv(minmax_summary) if r["model"] == "minmax"]
    if not baseline_rows:
        raise ValueError(f"No baseline rows in {baseline_summary}")
    if not minmax_rows:
        raise ValueError(f"No minmax rows in {minmax_summary}")
    return baseline_rows + minmax_rows


def _fig_name(stem: str, steps: int, suffix: str) -> str:
    return f"{stem}_{steps}{suffix}.png"


def plot_reward_curves(baseline_log, minmax_log, steps: int, out_dir: Path,
                       suffix: str = "", minmax_label: str = "PPO + Minmax") -> None:
    b = load_csv(baseline_log)
    m = load_csv(minmax_log)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        [int(r["step"]) for r in b],
        [float(r["mean_reward"]) for r in b],
        label="PPO + KL (baseline)",
        color="steelblue",
        linewidth=1.5,
    )
    ax.plot(
        [int(r["step"]) for r in m],
        [float(r["mean_reward"]) for r in m],
        label=minmax_label,
        color="darkorange",
        linewidth=1.5,
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Training Reward — {steps} steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / _fig_name("fig1_training_curves", steps, suffix)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_alg1_dynamics(minmax_log, steps: int, out_dir: Path, suffix: str = "") -> None:
    m = load_csv(minmax_log)
    fig, ax = plt.subplots(figsize=(9, 4))
    steps_x = [int(r["step"]) for r in m]
    ax.plot(steps_x, [float(r["v_max"]) for r in m], label="V_MAX", color="green")
    ax.plot(steps_x, [float(r["v_min"]) for r in m], label="V_MIN", color="red")
    ax.plot(
        steps_x,
        [float(r["r_unsafe"]) for r in m],
        label="R_unsafe",
        color="purple",
        linestyle="--",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Algorithm 1 Dynamics — {steps} steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / _fig_name("fig2_algorithm1_dynamics", steps, suffix)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def _rate_field(rows: list[dict]) -> str:
    if rows and "harm_rate_nonempty" in rows[0]:
        return "harm_rate_nonempty"
    return "harm_rate"


def plot_eval_bars(eval_rows: list[dict], steps: int, out_dir: Path,
                   suffix: str = "", minmax_label: str = "PPO + Minmax") -> None:
    rows = eval_rows
    rate_key = _rate_field(rows)
    spec = [
        ("baseline", "PPO + KL (baseline)", "steelblue"),
        ("minmax", minmax_label, "darkorange"),
    ]
    spec = [(m, label, color) for m, label, color in spec if m in {r["model"] for r in rows}]
    if not spec:
        print("No baseline/minmax rows in eval summary")
        return

    first_model = spec[0][0]
    categories = [
        r["category"]
        for r in rows
        if r["model"] == first_model and r["category"] != "_overall"
    ]
    x = np.arange(len(categories))
    bar_width = 0.8 / len(spec)

    fig, ax = plt.subplots(figsize=(15, 5.5))
    for i, (model, label, color) in enumerate(spec):
        by_cat = {
            r["category"]: float(r[rate_key])
            for r in rows
            if r["model"] == model and r["category"] != "_overall"
        }
        harm = [by_cat.get(c, 0.0) for c in categories]
        offset = (i - (len(spec) - 1) / 2) * bar_width
        ax.bar(x + offset, harm, bar_width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([c.split(",")[0][:22] for c in categories], rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Harm Rate (non-empty)")
    ax.set_title(f"Harm Rate by Category (non-empty) — {steps} steps")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = out_dir / _fig_name("fig3_harm_rate_bar", steps, suffix)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    overall_rows = [r for r in rows if r["category"] == "_overall"]
    overall_harm = {r["model"]: float(r[rate_key]) for r in overall_rows}
    overall_empty = {r["model"]: float(r.get("empty_rate", 0.0)) for r in overall_rows}

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [label for _, label, _ in spec]
    values = [overall_harm.get(m, 0.0) for m, _, _ in spec]
    colors = [color for _, _, color in spec]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f"{v * 100:.2f}%", ha="center", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Overall Harm Rate (non-empty)")
    ax.set_title(f"Overall Harm Rate (non-empty) — {steps} steps")
    ax.set_ylim(0, max(max(values) * 1.35, 0.05))
    plt.xticks(rotation=15, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = out_dir / _fig_name("fig4_overall_harm_bar", steps, suffix)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # Harm + empty side-by-side — exposes empty-response gaming
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_pos = np.arange(len(spec))
    width = 0.35
    harm_vals = [overall_harm.get(m, 0.0) for m, _, _ in spec]
    empty_vals = [overall_empty.get(m, 0.0) for m, _, _ in spec]
    b1 = ax.bar(x_pos - width / 2, harm_vals, width, label="Harm (non-empty)", color="indianred")
    b2 = ax.bar(x_pos + width / 2, empty_vals, width, label="Empty rate", color="gray")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Rate")
    ax.set_title(f"Harm vs Empty Responses — {steps} steps")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h * 100:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    out = out_dir / _fig_name("fig5_harm_vs_empty", steps, suffix)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_minmax_diagnostics(minmax_log, steps: int, out_dir: Path, suffix: str = "") -> None:
    m = load_csv(minmax_log)
    if not m or "n_unsafe" not in m[0]:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    steps_x = [int(r["step"]) for r in m]
    axes[0].plot(steps_x, [float(r["mean_toxicity"]) for r in m], color="crimson", linewidth=1.2)
    axes[0].set_ylabel("Mean Toxicity")
    axes[0].set_title(f"Minmax Training Diagnostics — {steps} steps")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps_x, [int(r["n_unsafe"]) for r in m], color="darkorange", label="n_unsafe", linewidth=1.2)
    if "floor_active" in m[0]:
        axes[1].plot(
            steps_x, [int(r["floor_active"]) for r in m], color="purple",
            linestyle="--", alpha=0.7, label="floor_active",
        )
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Count / flag")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / _fig_name("fig6_minmax_diagnostics", steps, suffix)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_baseline_vs_minmax_diagnostics(baseline_log, minmax_log, steps: int, out_dir: Path,
                                        suffix: str = "", minmax_label: str = "PPO + Minmax") -> None:
    """
    Side-by-side comparison of the four PPO internals that explain WHY kl
    moves the way it does, rather than just showing that it did:
      - entropy: falling entropy = policy collapsing toward one output
      - value_loss: the critic's own prediction error — spikes here often
        precede or accompany kl blowups, since a bad critic gives PPO a bad
        advantage estimate
      - policy_loss: the actual PPO objective being optimised
      - clipfrac: fraction of samples where PPO's clip mechanism engaged —
        rising clipfrac is often a LEADING indicator before kl itself spikes,
        since it means the policy is trying to move further than PPO's trust
        region allows

    Requires both logs to have gone through a training script with these
    four columns added (see train_baseline.py / train_minmax.py). Older logs
    without them are skipped with a printed note rather than a crash.
    """
    b = load_csv(baseline_log)
    m = load_csv(minmax_log)
    if not b or not m:
        return

    required = ("entropy", "value_loss", "policy_loss", "clipfrac")
    if not all(col in b[0] for col in required) or not all(col in m[0] for col in required):
        print(
            "Skipping combined diagnostics panel: one or both logs are missing "
            f"{required} — re-run training with the updated logging to get this plot."
        )
        return

    panels = [
        ("entropy", "Policy entropy", "Entropy"),
        ("value_loss", "Critic value loss", "Value loss"),
        ("policy_loss", "PPO policy loss", "Policy loss"),
        ("clipfrac", "PPO clip fraction", "Clip fraction"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes_flat = axes.flatten()

    b_steps = [int(r["step"]) for r in b]
    m_steps = [int(r["step"]) for r in m]

    for ax, (col, title, ylabel) in zip(axes_flat, panels):
        ax.plot(b_steps, [float(r[col]) for r in b], label="PPO + KL (baseline)",
                color="steelblue", linewidth=1.2)
        ax.plot(m_steps, [float(r[col]) for r in m], label=minmax_label,
                color="darkorange", linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes_flat[0].legend()
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 1].set_xlabel("Training Step")

    fig.suptitle(f"PPO internals: baseline vs Minmax — {steps} steps", y=1.00)
    plt.tight_layout()
    out = out_dir / _fig_name("fig7_baseline_vs_minmax_diagnostics", steps, suffix)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(args.baseline_log) and os.path.exists(args.minmax_log):
        plot_reward_curves(
            args.baseline_log, args.minmax_log, args.steps, out_dir,
            suffix=args.figure_suffix, minmax_label=args.minmax_label,
        )
        plot_alg1_dynamics(args.minmax_log, args.steps, out_dir, suffix=args.figure_suffix)
        plot_minmax_diagnostics(args.minmax_log, args.steps, out_dir, suffix=args.figure_suffix)
        plot_baseline_vs_minmax_diagnostics(
            args.baseline_log, args.minmax_log, args.steps, out_dir,
            suffix=args.figure_suffix, minmax_label=args.minmax_label,
        )
    else:
        print("Training logs not found — skipping curve plots.")

    eval_rows = None
    if args.eval_minmax_summary and args.eval_summary and os.path.exists(args.eval_summary):
        if os.path.exists(args.eval_minmax_summary):
            eval_rows = merge_eval_summaries(args.eval_summary, args.eval_minmax_summary)
        else:
            print(f"Eval minmax summary not found: {args.eval_minmax_summary}")
    elif args.eval_summary and os.path.exists(args.eval_summary):
        eval_rows = load_csv(args.eval_summary)

    if eval_rows:
        plot_eval_bars(
            eval_rows, args.steps, out_dir,
            suffix=args.figure_suffix, minmax_label=args.minmax_label,
        )
    elif args.eval_summary:
        print(f"Eval summary not found: {args.eval_summary}")


if __name__ == "__main__":
    main()