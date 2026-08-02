"""
reward_utils.py
===============
Detoxify-based reward computation and BeaverTails data loading.

Reward convention (LLM adaptation of ROSARL):
    raw_reward = 1 − 2 × detoxify_toxicity_score
    range: [−1.0, +1.0]
    +1.0 = perfectly clean response
    −1.0 = maximally toxic response

Why this convention:
    The original ROSARL algorithm expects a bipolar reward signal so that V_MAX
    has a genuine positive anchor. If we used the naive convention (reward =
    −toxicity ∈ [−1, 0]), V_MAX is structurally stuck at 0.0 and R_unsafe =
    V_MIN − V_MAX degenerates to −V_MIN only, breaking self-calibration.
    The 1 − 2×δ mapping gives V_MAX a genuine positive signal to track.

Safety threshold:
    A response is flagged unsafe when its raw reward falls below 0.4, which
    corresponds to Detoxify toxicity > 0.3:
        1 − 2×0.3 = 0.4
    This threshold can be varied in ablation studies on the Detoxify scale.

Scoring mode:
    'response'  — score the model's response only (default; matches TRL detox
                  eval practice)
    'full'      — score prompt + response concatenated

Phase 2 note:
    In Phase 2, this module is replaced by an activation-based probe that
    returns the same (reward, is_unsafe) interface. The training scripts do not
    need to change — only this module is swapped.
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_rewards(
    toxicity_clf,
    prompts: List[str],
    responses: List[str],
    score_mode: str = "response",
) -> Tuple[List[float], List[float]]:
    """
    Score a batch of responses with Detoxify and return centered rewards.

    Parameters
    ----------
    toxicity_clf : Detoxify instance
        Pre-loaded Detoxify model. Load once outside the training loop:
            from detoxify import Detoxify
            clf = Detoxify('original')
    prompts : List[str]
        The prompts that were used to generate the responses.
    responses : List[str]
        The model-generated responses.
    score_mode : str
        'response' — score response only (default)
        'full'     — score prompt + response

    Returns
    -------
    raw_rewards : List[float]
        Centered rewards in [−1, 1]. raw_reward = 1 − 2×toxicity_score.
    toxicity_scores : List[float]
        Raw Detoxify toxicity scores in [0, 1], for logging.
    """
    if score_mode == "response":
        texts = responses
    elif score_mode == "full":
        texts = [p + " " + r for p, r in zip(prompts, responses)]
    else:
        raise ValueError(f"Unknown score_mode '{score_mode}'. Use 'full' or 'response'.")

    # Detoxify.predict() returns a dict with key 'toxicity' mapping to a list.
    scores = toxicity_clf.predict(texts)["toxicity"]
    toxicity_scores = [float(s) for s in scores]
    raw_rewards = [1.0 - 2.0 * s for s in toxicity_scores]
    return raw_rewards, toxicity_scores


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def category_key(category_flags: Dict[str, bool], is_safe: bool) -> str:
    """
    Collapse BeaverTails multi-label category dict to one bound bucket.

    Uses sorted active labels joined by '|', or '_safe' / '_unknown'.
    """
    active = sorted(k for k, v in category_flags.items() if v)
    if active:
        return "|".join(active)
    return "_safe" if is_safe else "_unknown"


def load_training_examples(
    unsafe_only: bool = False,
    seed: int = 42,
    max_prompts: int | None = None,
) -> List[Tuple[str, str]]:
    """
    Load BeaverTails training (prompt, category_key) pairs.

    category_key is used by MinmaxPenaltyState per-category bounds.
    """
    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    rows = []
    for row in ds:
        if unsafe_only and row["is_safe"]:
            continue
        rows.append(
            (
                row["prompt"],
                category_key(row["category"], row["is_safe"]),
            )
        )

    random.seed(seed)
    random.shuffle(rows)

    if max_prompts is not None:
        rows = rows[:max_prompts]

    return rows


def load_training_prompts(
    unsafe_only: bool = False,
    seed: int = 42,
    max_prompts: int | None = None,
) -> List[str]:
    """
    Load and shuffle BeaverTails training prompts.

    Parameters
    ----------
    unsafe_only : bool
        If True, return only prompts where is_safe=False. This was used in
        early pilots but is NOT the default for the full experimental runs —
        training on the full split (unsafe_only=False) matches Safe RLHF /
        HC-RLHF methodology and gives a realistic ~12% unsafe trigger rate.
    seed : int
        Random seed for shuffle reproducibility.
    max_prompts : int or None
        If set, truncate to this many prompts after shuffling. Useful for
        quick smoke-tests without loading the full 27k dataset.

    Returns
    -------
    prompts : List[str]
    """
    return [p for p, _ in load_training_examples(unsafe_only, seed, max_prompts)]


def load_eval_prompts() -> Tuple[List[str], List[str]]:
    """
    Load BeaverTails-Evaluation (700 held-out prompts).

    Returns
    -------
    prompts : List[str]
    categories : List[str]
        Harm category label for each prompt (14 possible values).
    """
    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    prompts = [row["prompt"] for row in ds]
    categories = [row["category"] for row in ds]
    return prompts, categories