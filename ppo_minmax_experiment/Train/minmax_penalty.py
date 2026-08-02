"""
minmax_penalty.py
=================
ROSARL Minmax penalty adapted for PPO-based LLM fine-tuning.

Algorithm 1 (from Nangue Tasse et al., RLC 2026):
    R_unsafe = V_MIN − V_MAX

V_MIN and V_MAX are running estimates that update every step and are never reset
between batches. When the unsafe-state detector fires, the raw reward is replaced
by R_unsafe, which is always non-positive and grows more negative as training
explores a wider range of values.

LLM-setting deviations from the original ROSARL (all documented):
  1. Reward is 1 − 2×δ (centered in [−1, 1]) rather than a dense environment
     reward. Required because Detoxify output is in [0, 1].
  2. Value estimate is taken from the terminal token of the response when
     bound_source includes the critic (after critic_warmup_steps).
  3. A floor of −2.0 is applied to R_unsafe as an engineering backstop against
     catastrophic PPO ratio violations during early training.
  4. Baseline uses β=0.2 (KL + Detoxify safety); Minmax uses β=0.01 (KL anchor
     only) so the comparison isolates the penalty mechanism.
  5. Hard switch only: when the detector fires, the full penalty is applied
     immediately (no smooth blending).
  6. bound_scope="category" tracks V_MIN/V_MAX per BeaverTails harm category
     instead of one global pair across all prompts (reduces contextual mixing).
  7. bound_source="reward" updates bounds from Detoxify rewards only until the
     critic warmup elapses; uncalibrated GPT-2 value heads otherwise saturate
     bounds to ±1 and collapse R_unsafe to the penalty floor.

Design invariants:
  - apply() computes R_unsafe from PRE-batch bounds, then updates bounds from
    the current batch (Algorithm 1 semantics).
  - This module has zero dependency on TRL or transformers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_FALLBACK_CATEGORY = "_global"


class MinmaxPenaltyState:
    """
    Maintains V_MIN and V_MAX for Minmax self-calibration.

    Parameters
    ----------
    safety_threshold : float
        Centered-reward threshold below which a response is flagged unsafe
        (Detoxify toxicity > 0.3 ↔ 1 − 2×0.3 = 0.4).
    penalty_floor : float
        Hard lower bound on R_unsafe (default −2.0).
    bound_scope : str
        ``"category"`` — separate bounds per harm category (default).
        ``"global"``   — single global pair (original adaptation).
    bound_source : str
        ``"reward"`` — use Detoxify rewards only (default, stable on GPT-2).
        ``"reward_and_value"`` — include critic after warmup.
    critic_warmup_steps : int
        Steps before critic values are folded into bound updates when
        bound_source is ``"reward_and_value"``.
    """

    def __init__(
        self,
        safety_threshold: float = 0.4,
        penalty_floor: float = -2.0,
        bound_scope: str = "category",
        bound_source: str = "reward",
        critic_warmup_steps: int = 100,
    ) -> None:
        if penalty_floor > 0:
            raise ValueError("penalty_floor must be non-positive.")
        if not (-1.0 <= safety_threshold <= 1.0):
            raise ValueError("safety_threshold must be in [−1, 1].")
        if bound_scope not in ("category", "global"):
            raise ValueError("bound_scope must be 'category' or 'global'.")
        if bound_source not in ("reward", "reward_and_value"):
            raise ValueError("bound_source must be 'reward' or 'reward_and_value'.")

        self.safety_threshold = safety_threshold
        self.penalty_floor = penalty_floor
        self.bound_scope = bound_scope
        self.bound_source = bound_source
        self.critic_warmup_steps = critic_warmup_steps

        # Global fallback bounds (also used when bound_scope="global").
        # v_max=1.0 is the best possible centered reward so R_unsafe starts at −1.0
        # rather than 0.0 before any batch is seen.
        self.v_min: float = 0.0
        self.v_max: float = 1.0
        self._category_bounds: Dict[str, Tuple[float, float]] = {}

        self.total_steps: int = 0
        self.total_unsafe_triggers: int = 0
        self.floor_active_steps: int = 0

    # ------------------------------------------------------------------
    # Bound helpers
    # ------------------------------------------------------------------

    def _bounds_for_category(self, category: Optional[str]) -> Tuple[float, float]:
        if self.bound_scope == "global":
            return self.v_min, self.v_max
        key = category or _FALLBACK_CATEGORY
        if key in self._category_bounds:
            return self._category_bounds[key]
        return self.v_min, self.v_max

    def _use_critic_in_bounds(self) -> bool:
        if self.bound_source != "reward_and_value":
            return False
        return self.total_steps >= self.critic_warmup_steps

    def _batch_value_bounds(self, value_estimates, batch_r_min: float, batch_r_max: float):
        if value_estimates is None or not self._use_critic_in_bounds():
            return batch_r_min, batch_r_max

        if _TORCH_AVAILABLE and hasattr(value_estimates, "numel"):
            ve = value_estimates
            if ve.numel() > 0:
                return float(ve.min().item()), float(ve.max().item())

        ve_list = list(value_estimates)
        if ve_list:
            return min(ve_list), max(ve_list)
        return batch_r_min, batch_r_max

    def _update_bounds(
        self,
        category: Optional[str],
        batch_r_min: float,
        batch_r_max: float,
        batch_v_min: float,
        batch_v_max: float,
    ) -> None:
        lo = min(batch_r_min, batch_v_min)
        hi = max(batch_r_max, batch_v_max)

        self.v_min = min(self.v_min, lo)
        self.v_max = max(self.v_max, hi)

        if self.bound_scope == "global":
            return

        key = category or _FALLBACK_CATEGORY
        if key not in self._category_bounds:
            self._category_bounds[key] = (lo, hi)
        else:
            cur_lo, cur_hi = self._category_bounds[key]
            self._category_bounds[key] = (min(cur_lo, lo), max(cur_hi, hi))

    def _r_unsafe_for_bounds(self, v_min: float, v_max: float) -> Tuple[float, float, bool]:
        r_unsafe_raw = v_min - v_max
        floor_active = r_unsafe_raw < self.penalty_floor
        r_unsafe = max(r_unsafe_raw, self.penalty_floor)
        return r_unsafe, r_unsafe_raw, floor_active

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def apply(
        self,
        raw_rewards: List[float],
        value_estimates=None,
        categories: Optional[List[str]] = None,
    ) -> Tuple[list, float, float, bool, float, float, int]:
        """
        Apply the Minmax penalty to a batch of rewards.

        Parameters
        ----------
        raw_rewards : List[float]
            Centered rewards in [−1, 1].
        value_estimates : optional
            Terminal-token critic values, shape (B,).
        categories : optional List[str]
            BeaverTails harm category per response (for bound_scope="category").

        Returns
        -------
        penalized_rewards, r_unsafe, r_unsafe_raw, floor_active,
        batch_v_min, batch_v_max, n_unsafe
        """
        if categories is not None and len(categories) != len(raw_rewards):
            raise ValueError("categories must match raw_rewards length.")

        penalized: list = []
        n_unsafe = 0
        step_r_unsafe = 0.0
        step_r_unsafe_raw = 0.0
        step_floor_active = False

        batch_v_mins: List[float] = []
        batch_v_maxs: List[float] = []

        for i, raw in enumerate(raw_rewards):
            cat = categories[i] if categories is not None else None
            v_min, v_max = self._bounds_for_category(cat)
            r_unsafe, r_unsafe_raw, floor_active = self._r_unsafe_for_bounds(v_min, v_max)

            if raw < self.safety_threshold:
                if _TORCH_AVAILABLE:
                    import torch as _torch

                    penalized.append(_torch.tensor(r_unsafe, dtype=_torch.float32))
                else:
                    penalized.append(float(r_unsafe))
                n_unsafe += 1
                step_r_unsafe = r_unsafe
                step_r_unsafe_raw = r_unsafe_raw
                step_floor_active = floor_active
            else:
                if _TORCH_AVAILABLE:
                    import torch as _torch

                    penalized.append(_torch.tensor(float(raw), dtype=_torch.float32))
                else:
                    penalized.append(float(raw))

        # Update bounds per category (or once globally).
        if categories is None or self.bound_scope == "global":
            batch_r_min = min(raw_rewards)
            batch_r_max = max(raw_rewards)
            bv_min, bv_max = self._batch_value_bounds(
                value_estimates, batch_r_min, batch_r_max
            )
            batch_v_mins.append(bv_min)
            batch_v_maxs.append(bv_max)
            self._update_bounds(None, batch_r_min, batch_r_max, bv_min, bv_max)
        else:
            by_cat: Dict[str, List[int]] = {}
            for i, cat in enumerate(categories):
                by_cat.setdefault(cat or _FALLBACK_CATEGORY, []).append(i)

            for cat, indices in by_cat.items():
                cat_rewards = [raw_rewards[j] for j in indices]
                batch_r_min = min(cat_rewards)
                batch_r_max = max(cat_rewards)
                if value_estimates is not None and _TORCH_AVAILABLE and hasattr(
                    value_estimates, "__getitem__"
                ) and hasattr(value_estimates, "numel"):
                    ve_slice = value_estimates[indices]
                elif value_estimates is not None:
                    ve_list = list(value_estimates)
                    ve_slice = [ve_list[j] for j in indices]
                else:
                    ve_slice = None
                bv_min, bv_max = self._batch_value_bounds(
                    ve_slice, batch_r_min, batch_r_max
                )
                batch_v_mins.append(bv_min)
                batch_v_maxs.append(bv_max)
                self._update_bounds(cat, batch_r_min, batch_r_max, bv_min, bv_max)

        batch_v_min = min(batch_v_mins) if batch_v_mins else 0.0
        batch_v_max = max(batch_v_maxs) if batch_v_maxs else 0.0

        self.total_steps += 1
        self.total_unsafe_triggers += n_unsafe
        if step_floor_active:
            self.floor_active_steps += 1

        return (
            penalized,
            step_r_unsafe,
            step_r_unsafe_raw,
            step_floor_active,
            batch_v_min,
            batch_v_max,
            n_unsafe,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "v_min": self.v_min,
            "v_max": self.v_max,
            "category_bounds": {
                k: [lo, hi] for k, (lo, hi) in self._category_bounds.items()
            },
            "safety_threshold": self.safety_threshold,
            "penalty_floor": self.penalty_floor,
            "bound_scope": self.bound_scope,
            "bound_source": self.bound_source,
            "critic_warmup_steps": self.critic_warmup_steps,
            "total_steps": self.total_steps,
            "total_unsafe_triggers": self.total_unsafe_triggers,
            "floor_active_steps": self.floor_active_steps,
        }

    def load_state_dict(self, d: dict) -> None:
        self.v_min = d["v_min"]
        self.v_max = d["v_max"]
        self.total_steps = d["total_steps"]
        self.total_unsafe_triggers = d["total_unsafe_triggers"]
        self.floor_active_steps = d["floor_active_steps"]
        self.bound_scope = d.get("bound_scope", self.bound_scope)
        self.bound_source = d.get("bound_source", self.bound_source)
        self.critic_warmup_steps = d.get("critic_warmup_steps", self.critic_warmup_steps)
        raw_cats = d.get("category_bounds", {})
        self._category_bounds = {
            k: (float(v[0]), float(v[1])) for k, v in raw_cats.items()
        }

    def __repr__(self) -> str:
        n_cat = len(self._category_bounds)
        return (
            f"MinmaxPenaltyState("
            f"v_min={self.v_min:.4f}, v_max={self.v_max:.4f}, "
            f"r_unsafe={self.v_min - self.v_max:.4f}, "
            f"scope={self.bound_scope}, source={self.bound_source}, "
            f"categories={n_cat})"
        )

    @property
    def num_categories(self) -> int:
        return len(self._category_bounds)
