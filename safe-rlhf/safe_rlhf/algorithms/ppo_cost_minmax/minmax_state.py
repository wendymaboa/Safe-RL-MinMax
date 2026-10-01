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
"""Running V_MIN / V_MAX state for Stage 5 Run C / Run D (cost-gated Minmax).

Algorithm 1 (Nangue Tasse et al., RLC 2026):
    R_unsafe = V_MIN − V_MAX

Run D optionally scales that bound by how far the sample sits past the cost
threshold, without changing the units:

    excess    = max(cost − threshold, 0)
    severity  = min(excess / c_scale, severity_cap)
    R_unsafe  = −gap × (1 + severity)

``c_scale`` is the running mean of excess on gated samples, floored at
``cost_scale_floor`` (default 1.0). A barely unsafe reply still gets the full
MinMax gap; a worse reply can go as far as ``−2 · gap`` when ``severity_cap=1``.

The detector itself lives outside this class — both runs fire on
``cost > threshold``. This module (1) returns the current self-calibrated
penalty from pre-batch bounds, (2) expands those bounds from the current
batch's reward scores, and (3) optionally tracks the cost-excess scale.
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
        cost_scale_floor: float = 1.0,
        severity_cap: float = 1.0,
    ) -> None:
        if penalty_floor > 0:
            raise ValueError('penalty_floor must be non-positive.')
        if bound_source not in ('reward', 'reward_and_value'):
            raise ValueError("bound_source must be 'reward' or 'reward_and_value'.")
        if cost_scale_floor <= 0:
            raise ValueError('cost_scale_floor must be positive.')
        if severity_cap < 0:
            raise ValueError('severity_cap must be non-negative.')

        self.penalty_floor = penalty_floor
        self.bound_source = bound_source
        self.critic_warmup_steps = critic_warmup_steps
        self.cost_scale_floor = cost_scale_floor
        self.severity_cap = severity_cap

        # Start equal so R_unsafe = 0 until the first batch expands the range.
        # Beaver rewards are unbounded, so seeding ±1 (Phase 1's centered Detoxify
        # init) would invent a scale that is not in the data.
        self.v_min: float = 0.0
        self.v_max: float = 0.0

        self.total_steps: int = 0
        self.total_unsafe_triggers: int = 0
        self.floor_active_steps: int = 0

        # Running mean of (cost − threshold)+ on gated samples. Used only when
        # the trainer asks for a cost-scaled penalty (Run D). Starts at the
        # floor so the first batches do not divide by a tiny number.
        self.c_scale: float = cost_scale_floor
        self.c_scale_excess_sum: float = 0.0
        self.c_scale_count: int = 0

    @property
    def r_unsafe_raw(self) -> float:
        return self.v_min - self.v_max

    @property
    def r_unsafe(self) -> float:
        return max(self.r_unsafe_raw, self.penalty_floor)

    @property
    def gap(self) -> float:
        return self.v_max - self.v_min

    def effective_c_scale(self) -> float:
        return max(self.c_scale, self.cost_scale_floor)

    def severity(self, excess: float) -> float:
        """Map a non-negative cost excess onto [0, severity_cap]."""
        if excess <= 0:
            return 0.0
        return min(excess / self.effective_c_scale(), self.severity_cap)

    def scaled_penalty(self, excess: float) -> float:
        """Per-sample replacement: −gap × (1 + severity), floored.

        Uses the pre-update bounds and the current ``c_scale``. A sample just
        past the threshold (excess → 0) still receives the full MinMax gap.
        """
        raw = -self.gap * (1.0 + self.severity(excess))
        return max(raw, self.penalty_floor)

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
        excess_sum: float = 0.0,
        adapt_cost_scale: bool = False,
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

        if adapt_cost_scale and n_unsafe > 0:
            self.c_scale_excess_sum += float(excess_sum)
            self.c_scale_count += int(n_unsafe)
            self.c_scale = max(
                self.c_scale_excess_sum / self.c_scale_count,
                self.cost_scale_floor,
            )

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
            'cost_scale_floor': self.cost_scale_floor,
            'severity_cap': self.severity_cap,
            'c_scale': self.c_scale,
            'c_scale_excess_sum': self.c_scale_excess_sum,
            'c_scale_count': self.c_scale_count,
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
        self.cost_scale_floor = float(state.get('cost_scale_floor', self.cost_scale_floor))
        self.severity_cap = float(state.get('severity_cap', self.severity_cap))
        self.c_scale = float(state.get('c_scale', self.c_scale))
        self.c_scale_excess_sum = float(state.get('c_scale_excess_sum', self.c_scale_excess_sum))
        self.c_scale_count = int(state.get('c_scale_count', self.c_scale_count))

    def __repr__(self) -> str:
        return (
            f'CostMinmaxState(v_min={self.v_min:.4f}, v_max={self.v_max:.4f}, '
            f'r_unsafe={self.r_unsafe:.4f}, c_scale={self.c_scale:.4f}, '
            f'source={self.bound_source}, steps={self.total_steps})'
        )
