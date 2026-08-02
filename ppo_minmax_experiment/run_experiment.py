"""
run_experiment.py — train baseline + Minmax, then evaluate on BeaverTails.

Usage:
    python run_experiment.py --steps 1000 --seed 42
    python run_experiment.py --steps 10 --smoke-test
    python run_experiment.py --eval-only --steps 1000 --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from Train.ppo_defaults import (
    DEFAULT_KL_COEF_BASELINE,
    DEFAULT_KL_COEF_MINMAX,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SCORE_MODE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline + Minmax experiment")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--score-mode", type=str, default=DEFAULT_SCORE_MODE)
    p.add_argument("--baseline-kl-coef", type=float, default=DEFAULT_KL_COEF_BASELINE)
    p.add_argument("--minmax-kl-coef", type=float, default=DEFAULT_KL_COEF_MINMAX)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-minmax", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed (exit {result.returncode}, {time.time() - t0:.0f}s)")
        sys.exit(result.returncode)
    print(f"\n[OK] {label} ({time.time() - t0:.0f}s)")


def main() -> None:
    args = parse_args()
    steps = 10 if args.smoke_test else args.steps

    exp_dir = ROOT / f"experiments/seed_{args.seed}_steps_{steps}"
    log_dir = exp_dir / "logs"
    ckpt_dir = exp_dir / "checkpoints"
    results_dir = exp_dir / "results"
    for d in (log_dir, ckpt_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    baseline_log = log_dir / "baseline_training.csv"
    minmax_log = log_dir / "minmax_training.csv"
    baseline_ckpt = ckpt_dir / "baseline"
    minmax_ckpt = ckpt_dir / "minmax"
    python = sys.executable
    local_only = ["--local-files-only"] if args.local_files_only else []

    if not args.eval_only:
        if not args.skip_baseline:
            run(
                [
                    python,
                    "Train/train_baseline.py",
                    "--steps",
                    str(steps),
                    "--seed",
                    str(args.seed),
                    "--model-name",
                    args.model_name,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--score-mode",
                    args.score_mode,
                    "--kl-coef",
                    str(args.baseline_kl_coef),
                    "--log-path",
                    str(baseline_log),
                    "--save-dir",
                    str(baseline_ckpt),
                    *local_only,
                ],
                f"BASELINE ({steps} steps, seed={args.seed}, kl={args.baseline_kl_coef})",
            )

        if not args.skip_minmax:
            run(
                [
                    python,
                    "Train/train_minmax.py",
                    "--steps",
                    str(steps),
                    "--seed",
                    str(args.seed),
                    "--model-name",
                    args.model_name,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--score-mode",
                    args.score_mode,
                    "--kl-coef",
                    str(args.minmax_kl_coef),
                    "--log-path",
                    str(minmax_log),
                    "--save-dir",
                    str(minmax_ckpt),
                    *local_only,
                ],
                f"MINMAX ({steps} steps, seed={args.seed}, kl={args.minmax_kl_coef})",
            )

    run(
        [
            python,
            "Evaluation/eval_beavertails.py",
            "--steps",
            str(steps),
            "--run-tag",
            f"seed{args.seed}",
            "--output-dir",
            str(results_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--baseline-model",
            str(baseline_ckpt.relative_to(ROOT)),
            "--minmax-model",
            str(minmax_ckpt.relative_to(ROOT)),
        ],
        "EVALUATION (BeaverTails-Evaluation)",
    )

    summary = results_dir / f"eval_{steps}_seed{args.seed}_summary.csv"
    print(f"\nExperiment complete: {exp_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
