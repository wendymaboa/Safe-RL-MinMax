"""Shared PPO defaults — baseline and Minmax differ only in KL anchor and the penalty."""

from __future__ import annotations

import random

import numpy as np
import torch

# Fair-comparison protocol (see README Design section).
DEFAULT_KL_COEF_BASELINE = 0.2
DEFAULT_KL_COEF_MINMAX = 0.01
DEFAULT_KL_COEF = DEFAULT_KL_COEF_BASELINE  # legacy alias for baseline scripts

DEFAULT_SAFETY_THRESHOLD = 0.4
DEFAULT_PENALTY_FLOOR = -2.0
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_MINI_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1.41e-5
DEFAULT_PPO_EPOCHS = 1
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_SCORE_MODE = "response"

# Minmax self-calibration: reward-only bounds + per-category scope avoids
# global saturation from uncalibrated GPT-2 critics and mixed prompts.
DEFAULT_BOUND_SCOPE = "category"  # "category" | "global"
DEFAULT_BOUND_SOURCE = "reward"  # "reward" | "reward_and_value"
DEFAULT_CRITIC_WARMUP_STEPS = 100
# Standard TRL KL term. "abs" was tried (Run v2) but minmax collapsed to ~92% empty at eval.
DEFAULT_KL_PENALTY = "kl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_experiment_ppo_config(
    model_name: str,
    *,
    kl_coef: float = DEFAULT_KL_COEF_BASELINE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
):
    from trl_compat import make_ppo_config

    return make_ppo_config(
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=1,
        ppo_epochs=DEFAULT_PPO_EPOCHS,
        optimize_cuda_cache=True,
        kl_penalty=DEFAULT_KL_PENALTY,
        init_kl_coef=kl_coef,
        adap_kl_ctrl=False,
        horizon=10000,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        log_with=None,
    )
