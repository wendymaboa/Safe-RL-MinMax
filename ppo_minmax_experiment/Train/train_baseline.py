"""
PPO + KL divergence baseline (control condition).

Usage:
    python Train/train_baseline.py --steps 1000 --seed 42
    python Train/train_baseline.py --steps 1000 --seed 42 --resume-from models/baseline --start-step 500
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from detoxify import Detoxify
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trl_compat import AutoModelForCausalLMWithValueHead, PPOTrainer
from Train.checkpoint_utils import (
    default_log_path,
    default_save_dir,
    resolve_resume_path,
    save_checkpoint,
)
from Train.generation_utils import generate_batch
from Train.ppo_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_KL_COEF_BASELINE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MINI_BATCH_SIZE,
    DEFAULT_SCORE_MODE,
    make_experiment_ppo_config,
    set_seed,
)
from Train.reward_utils import compute_rewards, load_training_prompts

CONDITION = "baseline"
# entropy, value_loss, policy_loss, clipfrac added: these come straight out of
# TRL's own trainer.step() stats dict and were previously computed and thrown
# away. They're the diagnostics that actually explain WHY kl_divergence moves
# (e.g. falling entropy = policy collapsing toward one output; rising
# clipfrac = PPO's clip mechanism engaging heavily, often a leading indicator
# before kl itself blows up) rather than just showing that it did.
LOG_COLUMNS = [
    "step",
    "condition",
    "mean_reward",
    "mean_raw_reward",
    "kl_divergence",
    "entropy",
    "value_loss",
    "policy_loss",
    "clipfrac",
    "mean_toxicity",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO + KL baseline training")
    p.add_argument("--steps", type=int, default=1000,
                   help="Stop after this step index (train steps 0 .. steps-1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-path", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--mini-batch", type=int, default=DEFAULT_MINI_BATCH_SIZE)
    p.add_argument("--kl-coef", type=float, default=DEFAULT_KL_COEF_BASELINE)
    p.add_argument("--score-mode", type=str, default=DEFAULT_SCORE_MODE, choices=["full", "response"])
    p.add_argument("--save-every", type=int, default=250,
                   help="Write a checkpoint every N steps (0 = final only).")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Checkpoint dir or run root to continue from.")
    p.add_argument("--start-step", type=int, default=None,
                   help="Required when resuming a legacy save without training_state.json.")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_path is None:
        args.log_path = str(default_log_path(args.steps, args.seed, CONDITION))
    if args.save_dir is None:
        args.save_dir = str(default_save_dir(args.steps, args.seed, CONDITION))

    set_seed(args.seed)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] Device: {device}")

    start_step = 0
    load_kw = {"local_files_only": args.local_files_only}

    if args.resume_from:
        ckpt_path, start_step = resolve_resume_path(args.resume_from)
        if args.start_step is not None:
            start_step = args.start_step
        elif start_step == 0 and not (ckpt_path / "training_state.json").is_file():
            raise SystemExit(
                "Legacy checkpoint: pass --start-step N (e.g. 500) to resume."
            )
        print(f"[baseline] Resuming from {ckpt_path} at step {start_step}")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, **load_kw)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(ckpt_path, **load_kw).to(device)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name, **load_kw
        ).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kw)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name, **load_kw
        ).to(device)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name, **load_kw
        ).to(device)
        model.v_head.summary.weight.data.zero_()
        model.v_head.summary.bias.data.zero_()

    tokenizer.pad_token = tokenizer.eos_token

    config = make_experiment_ppo_config(
        args.model_name,
        kl_coef=args.kl_coef,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch,
    )
    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    toxicity_clf = Detoxify("original")
    prompts = load_training_prompts(seed=args.seed)
    print(f"[baseline] Loaded {len(prompts)} prompts (seed={args.seed})")
    print(f"[baseline] Steps {start_step}->{args.steps}, save every {args.save_every or 'final'}")
    print(f"[baseline] Save dir: {args.save_dir}")

    append_log = start_step > 0 and Path(args.log_path).is_file()
    with open(args.log_path, "a" if append_log else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if not append_log:
            writer.writeheader()

        t0 = time.time()
        last_step = start_step
        for step in range(start_step, args.steps):
            last_step = step
            start = step * args.batch_size
            batch_prompts = prompts[start : start + args.batch_size]
            if len(batch_prompts) < args.batch_size:
                break

            query_tensors, response_tensors, batch_responses, _ = generate_batch(
                trainer, tokenizer, batch_prompts, device, args.max_new_tokens,
            )
            if not query_tensors:
                continue

            raw_rewards, toxicity_scores = compute_rewards(
                toxicity_clf, batch_prompts, batch_responses, score_mode=args.score_mode,
            )
            reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in raw_rewards]

            stats = trainer.step(query_tensors, response_tensors, reward_tensors)
            kl = float(stats.get("objective/kl", 0.0))
            entropy = float(stats.get("ppo/policy/entropy", 0.0))
            value_loss = float(stats.get("ppo/loss/value", 0.0))
            policy_loss = float(stats.get("ppo/loss/policy", 0.0))
            clipfrac = float(stats.get("ppo/policy/clipfrac", 0.0))
            mean_raw = sum(raw_rewards) / len(raw_rewards)
            mean_tox = sum(toxicity_scores) / len(toxicity_scores)

            writer.writerow({
                "step": step,
                "condition": CONDITION,
                "mean_reward": round(mean_raw, 4),
                "mean_raw_reward": round(mean_raw, 4),
                "kl_divergence": round(kl, 4),
                "entropy": round(entropy, 4),
                "value_loss": round(value_loss, 4),
                "policy_loss": round(policy_loss, 4),
                "clipfrac": round(clipfrac, 4),
                "mean_toxicity": round(mean_tox, 4),
            })
            fh.flush()

            if step % 10 == 0:
                print(
                    f"Step {step:4d} | reward={mean_raw:.4f} | "
                    f"tox={mean_tox:.4f} | kl={kl:.4f} | "
                    f"entropy={entropy:.4f} | vloss={value_loss:.4f} | clipfrac={clipfrac:.4f}"
                )

            if args.save_every and (step + 1) % args.save_every == 0:
                ckpt = save_checkpoint(
                    args.save_dir, step, trainer, tokenizer,
                    condition=CONDITION, seed=args.seed, steps_target=args.steps,
                )
                print(f"[baseline] Checkpoint saved: {ckpt}")

        save_checkpoint(
            args.save_dir, last_step, trainer, tokenizer,
            condition=CONDITION, seed=args.seed, steps_target=args.steps,
        )

    elapsed = time.time() - t0
    print(f"\n[baseline] Log: {args.log_path}")
    print(f"[baseline] Model: {args.save_dir}")
    print(f"[baseline] Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()