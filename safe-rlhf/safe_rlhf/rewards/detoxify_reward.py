"""Detoxify-based verifiable reward for GPT-2 Minmax experiments."""

from __future__ import annotations

import torch
from detoxify import Detoxify

_classifier: Detoxify | None = None

# Centered scale: toxicity 0 -> +1, toxicity 0.5 -> 0, toxicity 1 -> -1
DEFAULT_SAFETY_THRESHOLD = 0.4  # toxicity > 0.3


def get_classifier() -> Detoxify:
    global _classifier
    if _classifier is None:
        _classifier = Detoxify('original')
    return _classifier


def compute_reward(texts: list[str]) -> torch.Tensor:
    """Return centered Detoxify scores in [-1, 1]: 1 - 2*toxicity."""
    scores = get_classifier().predict(texts)['toxicity']
    return torch.tensor([1.0 - 2.0 * float(s) for s in scores], dtype=torch.float32)


def is_unsafe(rewards: torch.Tensor, threshold: float = DEFAULT_SAFETY_THRESHOLD) -> torch.Tensor:
    return rewards < threshold
