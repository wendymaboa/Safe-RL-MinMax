"""Unit tests for CostMinmaxState (no GPU / DeepSpeed required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_STATE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'safe_rlhf'
    / 'algorithms'
    / 'ppo_cost_minmax'
    / 'minmax_state.py'
)
_SPEC = importlib.util.spec_from_file_location('minmax_state', _STATE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules['minmax_state'] = _MOD
_SPEC.loader.exec_module(_MOD)
CostMinmaxState = _MOD.CostMinmaxState


def test_penalty_starts_at_zero_then_expands():
    state = CostMinmaxState(penalty_floor=-50.0, bound_source='reward')
    r_unsafe, r_raw, floor_active = state.current_penalty()
    assert r_unsafe == 0.0
    assert r_raw == 0.0
    assert not floor_active

    state.update(batch_r_min=-3.0, batch_r_max=2.0, n_unsafe=1)
    assert state.v_min == -3.0
    assert state.v_max == 2.0
    r_unsafe, r_raw, floor_active = state.current_penalty()
    assert r_raw == -5.0
    assert r_unsafe == -5.0
    assert not floor_active


def test_floor_binds_when_range_is_wide():
    state = CostMinmaxState(penalty_floor=-2.0, bound_source='reward')
    state.update(batch_r_min=-4.0, batch_r_max=4.0, n_unsafe=2)
    r_unsafe, r_raw, floor_active = state.current_penalty()
    assert r_raw == -8.0
    assert r_unsafe == -2.0
    assert floor_active


def test_critic_ignored_before_warmup():
    state = CostMinmaxState(
        penalty_floor=-50.0,
        bound_source='reward_and_value',
        critic_warmup_steps=2,
    )
    state.update(batch_r_min=0.0, batch_r_max=1.0, batch_v_min=-9.0, batch_v_max=9.0)
    assert state.v_min == 0.0
    assert state.v_max == 1.0
    state.update(batch_r_min=0.0, batch_r_max=1.0, batch_v_min=-9.0, batch_v_max=9.0)
    # total_steps is now 2, so the next update includes critic
    state.update(batch_r_min=0.0, batch_r_max=1.0, batch_v_min=-9.0, batch_v_max=9.0)
    assert state.v_min == -9.0
    assert state.v_max == 9.0


def test_state_dict_roundtrip():
    state = CostMinmaxState(penalty_floor=-50.0)
    state.update(batch_r_min=-1.5, batch_r_max=3.0, n_unsafe=4)
    restored = CostMinmaxState()
    restored.load_state_dict(state.state_dict())
    assert restored.v_min == state.v_min
    assert restored.v_max == state.v_max
    assert restored.total_steps == state.total_steps
    assert restored.total_unsafe_triggers == state.total_unsafe_triggers


if __name__ == '__main__':
    test_penalty_starts_at_zero_then_expands()
    test_floor_binds_when_range_is_wide()
    test_critic_ignored_before_warmup()
    test_state_dict_roundtrip()
    print('OK: cost_minmax state tests passed')
