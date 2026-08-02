"""Unit tests for MinmaxPenaltyState (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Train.minmax_penalty import MinmaxPenaltyState


def test_reward_only_category_bounds_avoid_immediate_floor():
  """Uncalibrated critic values must not pin R_unsafe to the floor on step 1."""
  state = MinmaxPenaltyState(
      bound_scope="category",
      bound_source="reward",
      penalty_floor=-2.0,
  )
  raw = [0.8, -0.2, 0.5, 0.9]
  cats = ["hate", "hate", "violence", "violence"]
  bad_critic = [-0.99, 0.99, -0.99, 0.99]

  _, r_unsafe, r_unsafe_raw, floor_active, _, _, n_unsafe = state.apply(
      raw, value_estimates=bad_critic, categories=cats,
  )

  assert n_unsafe == 1
  assert not floor_active, "reward-only bounds should not hit floor immediately"
  assert r_unsafe_raw > -2.0
  assert r_unsafe == r_unsafe_raw
  assert state.num_categories >= 1


def test_per_category_penalty_differs():
    state = MinmaxPenaltyState(bound_scope="category", bound_source="reward")
    state.apply([-0.5, -0.4], categories=["hate", "hate"])
    _, r_v, r_raw, _, _, _, n_unsafe = state.apply(
        [0.1], categories=["violence"],
    )
    assert n_unsafe == 1
    assert r_raw == state.v_min - state.v_max


if __name__ == "__main__":
  test_reward_only_category_bounds_avoid_immediate_floor()
  test_per_category_penalty_differs()
  print("OK: minmax_penalty tests passed")
