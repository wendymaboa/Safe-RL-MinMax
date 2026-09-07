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
"""PPO with a threshold-gated Minmax cost penalty (Stage 5, Run C).

Shares Run B's cost-model plumbing and the identical gate condition
(``cost > threshold``). The only deliberate difference is the penalty magnitude:
Run B replaces an unsafe response's reward with a fixed constant; Run C replaces
it with the self-calibrated bound ``R_unsafe = V_MIN − V_MAX``, where the bounds
are running min/max over observed reward-model scores (optionally including
critic values after warmup).

That isolates the thesis question: does a self-calibrating bound buy anything
over a fixed −2.0 penalty, given the same detector and the same trigger?
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.distributed as dist

from safe_rlhf.algorithms.ppo_cost_gate.trainer import PPOCostGateTrainer
from safe_rlhf.algorithms.ppo_cost_minmax.minmax_state import CostMinmaxState
from safe_rlhf.utils import (
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    is_main_process,
    masked_mean,
)


class PPOCostMinmaxTrainer(PPOCostGateTrainer):
    TRAINING_TYPE = 'ppo_cost_minmax'

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        # Parent stores penalty_magnitude for Run B; unused here but required by
        # its __init__. A sentinel keeps AttributeError away without implying a
        # real fixed penalty.
        if not hasattr(args, 'penalty_magnitude'):
            args.penalty_magnitude = float('nan')
        super().__init__(args, ds_train_config, ds_eval_config)
        self.minmax = CostMinmaxState(
            penalty_floor=self.args.penalty_floor,
            bound_source=self.args.bound_source,
            critic_warmup_steps=self.args.critic_warmup_steps,
        )

    def _sync_scalar_minmax(self, local_min: torch.Tensor, local_max: torch.Tensor) -> tuple[float, float]:
        """All-reduce batch extrema so every rank updates V_MIN/V_MAX identically."""
        synced_min = local_min.clone()
        synced_max = local_max.clone()
        dist.all_reduce(synced_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(synced_max, op=dist.ReduceOp.MAX)
        return float(synced_min.item()), float(synced_max.item())

    def _maybe_persist_minmax_state(self) -> None:
        if not is_main_process() or not self.args.output_dir:
            return
        path = os.path.join(self.args.output_dir, 'minmax_state.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(self.minmax.state_dict(), handle, indent=2)

    # pylint: disable-next=too-many-locals
    def rl_step(self, rl_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        prompt = rl_batch['prompt']
        old_log_probs = rl_batch['log_probs']
        ref_log_probs = rl_batch['ref_log_probs']
        reward = rl_batch['reward']
        cost = rl_batch['cost']
        old_reward_values = rl_batch['reward_values']
        input_ids = rl_batch['input_ids']
        attention_mask = rl_batch['attention_mask']

        start = prompt.size(-1) - 1
        sequence_mask = attention_mask[:, 1:]

        # THE GATE — same condition as Run B. Only the replacement value differs:
        # V_MIN − V_MAX (floored) instead of a fixed --penalty_magnitude.
        unsafe_mask = cost > self.cost_threshold
        r_unsafe, r_unsafe_raw, floor_active = self.minmax.current_penalty()
        gated_reward = torch.where(
            unsafe_mask,
            torch.full_like(reward, r_unsafe),
            reward,
        )

        # Expand bounds from this batch AFTER applying the pre-batch penalty
        # (Algorithm 1). Sync extrema across ranks so multi-GPU stays consistent.
        batch_r_min, batch_r_max = self._sync_scalar_minmax(reward.min(), reward.max())
        # Terminal-token critic estimate per response (last non-masked position
        # in the generation span). Used only when bound_source includes values.
        with torch.no_grad():
            gen_mask = sequence_mask[:, start:]
            # gather last valid critic value per row; fall back to 0 if empty
            lengths = gen_mask.long().sum(dim=-1).clamp(min=1)
            idx = (lengths - 1).unsqueeze(-1)
            terminal_values = old_reward_values[:, start:].gather(1, idx).squeeze(-1)
            batch_v_min, batch_v_max = self._sync_scalar_minmax(
                terminal_values.min(),
                terminal_values.max(),
            )

        n_unsafe_local = int(unsafe_mask.sum().item())
        n_unsafe_tensor = torch.tensor(
            n_unsafe_local,
            device=reward.device,
            dtype=torch.long,
        )
        dist.all_reduce(n_unsafe_tensor, op=dist.ReduceOp.SUM)
        self.minmax.update(
            batch_r_min=batch_r_min,
            batch_r_max=batch_r_max,
            batch_v_min=batch_v_min,
            batch_v_max=batch_v_max,
            n_unsafe=int(n_unsafe_tensor.item()),
        )
        self._maybe_persist_minmax_state()

        with torch.no_grad():
            old_rewards = self.add_kl_divergence_regularization(
                gated_reward,
                prompt,
                old_log_probs,
                ref_log_probs,
                sequence_mask,
            )
            reward_advantages, reward_returns = self.get_advantages_and_returns(
                old_reward_values,
                old_rewards,
                sequence_mask,
                start,
            )

        logits = self.actor_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        actor_loss = self.actor_loss_fn(
            log_probs[:, start:],
            old_log_probs[:, start:],
            reward_advantages,
            sequence_mask[:, start:],
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        reward_values = self.reward_critic_model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).scores
        reward_values = reward_values.squeeze(dim=-1)[:, :-1]
        reward_critic_loss = self.critic_loss_fn(
            reward_values[:, start:],
            old_reward_values[:, start:],
            reward_returns,
            sequence_mask[:, start:],
        )
        self.reward_critic_model.backward(reward_critic_loss)
        self.reward_critic_model.step()

        with torch.no_grad():
            mask = sequence_mask[:, start:]
            kl_divergence = ((old_log_probs - ref_log_probs)[:, start:] * mask).sum(dim=-1).mean()
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            unsafe_rate = unsafe_mask.float().mean()
            reward_mean = reward.mean()
            cost_mean = cost.mean()
            gated_reward_mean = gated_reward.mean()
            reward_with_kl_penalty = (old_rewards[:, start:] * mask).sum(dim=-1).mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values[:, start:], mask)

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            unsafe_rate = get_all_reduce_mean(unsafe_rate)
            reward_mean = get_all_reduce_mean(reward_mean)
            cost_mean = get_all_reduce_mean(cost_mean)
            gated_reward_mean = get_all_reduce_mean(gated_reward_mean)
            reward_with_kl_penalty = get_all_reduce_mean(reward_with_kl_penalty)
            reward_advantage = get_all_reduce_mean(reward_advantage)
            reward_return = get_all_reduce_mean(reward_return)
            reward_value = get_all_reduce_mean(reward_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        dist.barrier()

        return {
            'train/actor_loss': actor_loss.item(),
            'train/reward_critic_loss': reward_critic_loss.item(),
            'train/reward': reward_mean.item(),
            'train/cost': cost_mean.item(),
            'train/gated_reward': gated_reward_mean.item(),
            'train/unsafe_rate': unsafe_rate.item(),
            'train/v_min': self.minmax.v_min,
            'train/v_max': self.minmax.v_max,
            'train/r_unsafe': r_unsafe,
            'train/r_unsafe_raw': r_unsafe_raw,
            'train/floor_active': float(floor_active),
            'train/reward_with_kl_penalty': reward_with_kl_penalty.item(),
            'train/reward_advantage': reward_advantage.item(),
            'train/reward_return': reward_return.item(),
            'train/reward_value': reward_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
        }
