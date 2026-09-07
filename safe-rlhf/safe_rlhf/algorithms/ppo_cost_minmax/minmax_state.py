# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Running V_MIN / V_MAX state for Stage 5 Run C (cost-gated Minmax).

Algorithm 1 (Nangue Tasse et al., RLC 2026):
    R_unsafe = V_MIN − V_MAX

The detector itself lives outside this class — Run C fires on ``cost > threshold``,
the same condition as Run B. This module only (1) returns the current self-calibrated
penalty from pre-batch bounds and (2) expands those bounds from the current batch's
reward scores (optionally including critic values after warmup).

Deliberately simpler than Phase 1's ``MinmaxPenaltyState``: global bounds only (no
per-category scope), and the unsafe mask is supplied by the caller rather than
inferred from a Detoxify threshold on the reward itself.
"""

from __future__ import annotations

from typing import Any


class CostMinmaxState:
    """Track global V_MIN / V_MAX and produce R_unsafe = V_MIN − V_MAX."""

    def __init__(
        self,
        penalty_floor: float = -50.0,
        bound_source: str = 'reward',
        critic_warmup_steps: int = 100,
    ) -> None:
        if penalty_floor > 0:
            raise ValueError('penalty_floor must be non-positive.')
        if bound_source not in ('reward', 'reward_and_value'):
            raise ValueError("bound_source must be 'reward' or 'reward_and_value'.")

        self.penalty_floor = penalty_floor
        self.bound_source = bound_source
        self.critic_warmup_steps = critic_warmup_steps

        # Start equal so R_unsafe = 0 until the first batch expands the range.
        # Beaver rewards are unbounded, so seeding ±1 (Phase 1's centered Detoxify
        # init) would invent a scale that is not in the data.
        self.v_min: float = 0.0
        self.v_max: float = 0.0

        self.total_steps: int = 0
        self.total_unsafe_triggers: int = 0
        self.floor_active_steps: int = 0

    @property
    def r_unsafe_raw(self) -> float:
        return self.v_min - self.v_max

    @property
    def r_unsafe(self) -> float:
        return max(self.r_unsafe_raw, self.penalty_floor)

    def _use_critic(self) -> bool:
        return (
            self.bound_source == 'reward_and_value'
            and self.total_steps >= self.critic_warmup_steps
        )

    def current_penalty(self) -> tuple[float, float, bool]:
        """Return (r_unsafe, r_unsafe_raw, floor_active) from pre-update bounds."""
        raw = self.r_unsafe_raw
        floor_active = raw < self.penalty_floor
        return max(raw, self.penalty_floor), raw, floor_active

    def update(
        self,
        batch_r_min: float,
        batch_r_max: float,
        batch_v_min: float | None = None,
        batch_v_max: float | None = None,
        n_unsafe: int = 0,
    ) -> None:
        """Expand bounds from this batch. Call AFTER applying the pre-batch penalty."""
        lo = batch_r_min
        hi = batch_r_max
        if self._use_critic() and batch_v_min is not None and batch_v_max is not None:
            lo = min(lo, batch_v_min)
            hi = max(hi, batch_v_max)

        self.v_min = min(self.v_min, lo)
        self.v_max = max(self.v_max, hi)

        self.total_steps += 1
        self.total_unsafe_triggers += n_unsafe
        if self.r_unsafe_raw < self.penalty_floor:
            self.floor_active_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            'v_min': self.v_min,
            'v_max': self.v_max,
            'penalty_floor': self.penalty_floor,
            'bound_source': self.bound_source,
            'critic_warmup_steps': self.critic_warmup_steps,
            'total_steps': self.total_steps,
            'total_unsafe_triggers': self.total_unsafe_triggers,
            'floor_active_steps': self.floor_active_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.v_min = float(state['v_min'])
        self.v_max = float(state['v_max'])
        self.penalty_floor = float(state.get('penalty_floor', self.penalty_floor))
        self.bound_source = state.get('bound_source', self.bound_source)
        self.critic_warmup_steps = int(
            state.get('critic_warmup_steps', self.critic_warmup_steps),
        )
        self.total_steps = int(state['total_steps'])
        self.total_unsafe_triggers = int(state['total_unsafe_triggers'])
        self.floor_active_steps = int(state['floor_active_steps'])

    def __repr__(self) -> str:
        return (
            f'CostMinmaxState(v_min={self.v_min:.4f}, v_max={self.v_max:.4f}, '
            f'r_unsafe={self.r_unsafe:.4f}, source={self.bound_source}, '
            f'steps={self.total_steps})'
        )
