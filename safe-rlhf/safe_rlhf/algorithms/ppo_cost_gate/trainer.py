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
"""PPO with a threshold-gated fixed cost penalty (Stage 5, Run B).

Reward stays untouched below the cost threshold. Above it, the reward for that whole
response is REPLACED with a fixed penalty — not blended, not scaled by how unsafe the
response is. This is the control for Run C (Minmax): both fire on the identical
condition (`cost > threshold`), so a difference between B and C isolates exactly one
thing — a fixed penalty magnitude versus a self-calibrated one — rather than also
differing in when the mechanism activates at all.

Deliberately not `ppo_reward_shaping`, which is already in this codebase: that trainer
computes `reward - lambda * cost` on every sample, continuously, whether or not the
response is actually unsafe. That mechanism does not share Run C's trigger condition, so
it would not isolate the same thing.

Structurally this is `PPOTrainer` (single reward critic, unchanged KL/advantage/actor-loss
machinery) with one frozen cost model added for scoring, following the same shape as
`ppo_reward_shaping`.
"""

from __future__ import annotations

import argparse
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizerBase

from safe_rlhf.algorithms.ppo.trainer import PPOTrainer
from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.utils import (
    batch_retokenize,
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    is_same_tokenizer,
    masked_mean,
)


class PPOCostGateTrainer(PPOTrainer):
    TRAINING_TYPE = 'ppo_cost_gate'

    cost_model: deepspeed.DeepSpeedEngine
    cost_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        super().__init__(args, ds_train_config, ds_eval_config)
        self.cost_threshold = self.args.cost_threshold
        self.penalty_magnitude = self.args.penalty_magnitude

    def init_models(self) -> None:
        # Builds the actor (+LoRA), reference, reward model, and reward critic exactly
        # as PPOTrainer does. The cost model is frozen and gets no LoRA config, same as
        # the reward model — it is a detector, never trained.
        super().init_models()
        self.cost_model, self.cost_tokenizer = load_pretrained_models(
            self.args.cost_model_name_or_path,
            model_max_length=self.args.max_length,
            auto_model_type=AutoModelForScore,
            padding_side='right',
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs={
                'score_type': 'cost',
                'do_normalize': self.args.normalize_cost,
            },
        )
        self.cost_model.set_normalize(self.args.normalize_cost)

        if is_same_tokenizer(self.tokenizer, self.cost_tokenizer):
            self.cost_tokenizer = self.tokenizer

    def init_engines(self) -> None:
        super().init_engines()
        self.cost_model = self._init_eval_engine(
            model=self.cost_model,
            ds_config=self.ds_eval_config,
        )
        self.cost_model.eval()

    @torch.no_grad()
    def post_rollout(
        self,
        prompt: torch.Tensor,
        sequence: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, Any]:
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_seq = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_seq = sequence
            reward_attention_mask = attention_mask

        if self.cost_tokenizer is not self.tokenizer:
            cost_tokenize_output = batch_retokenize(
                sequence,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.cost_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            cost_seq = cost_tokenize_output['input_ids']
            cost_attention_mask = cost_tokenize_output['attention_mask']
        else:
            cost_seq = sequence
            cost_attention_mask = attention_mask

        logits = self.actor_model(sequence, attention_mask=attention_mask).logits
        ref_logits = self.actor_reference_model(sequence, attention_mask=attention_mask).logits

        reward = self.reward_model(reward_seq, attention_mask=reward_attention_mask).end_scores
        cost = self.cost_model(cost_seq, attention_mask=cost_attention_mask).end_scores
        reward_values = self.reward_critic_model(sequence, attention_mask=attention_mask).scores

        reward = reward.squeeze(dim=-1)
        cost = cost.squeeze(dim=-1)
        reward_values = reward_values.squeeze(dim=-1)[:, :-1]

        log_probs = gather_log_probabilities(logits[:, :-1], sequence[:, 1:])
        ref_log_probs = gather_log_probabilities(ref_logits[:, :-1], sequence[:, 1:])

        return {
            'prompt': prompt,
            'log_probs': log_probs,
            'ref_log_probs': ref_log_probs,
            'reward': reward,
            'cost': cost,
            'reward_values': reward_values,
            'input_ids': sequence,
            'attention_mask': attention_mask,
        }

    @torch.no_grad()
    def eval_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        if self.reward_tokenizer is not self.tokenizer:
            reward_tokenize_output = batch_retokenize(
                input_ids,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.reward_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            reward_input_ids = reward_tokenize_output['input_ids']
            reward_attention_mask = reward_tokenize_output['attention_mask']
        else:
            reward_input_ids = input_ids
            reward_attention_mask = attention_mask

        if self.cost_tokenizer is not self.tokenizer:
            cost_tokenize_output = batch_retokenize(
                input_ids,
                src_tokenizer=self.tokenizer,
                dest_tokenizer=self.cost_tokenizer,
                skip_special_tokens=True,
                device=self.args.device,
            )
            cost_input_ids = cost_tokenize_output['input_ids']
            cost_attention_mask = cost_tokenize_output['attention_mask']
        else:
            cost_input_ids = input_ids
            cost_attention_mask = attention_mask

        reward = self.reward_model(
            reward_input_ids,
            attention_mask=reward_attention_mask,
        ).end_scores.squeeze(dim=-1)
        cost = self.cost_model(
            cost_input_ids,
            attention_mask=cost_attention_mask,
        ).end_scores.squeeze(dim=-1)
        return {
            'eval/reward': reward,
            'eval/cost': cost,
        }

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

        # THE GATE. Above threshold: the response's reward is replaced outright with a
        # fixed penalty, independent of how far past the threshold it is. Below: reward
        # is untouched. This is the whole difference from plain PPOTrainer (Run A) and
        # the whole difference from Run C (Minmax), where the penalty magnitude is
        # V_MIN - V_MAX instead of this fixed constant.
        unsafe_mask = cost > self.cost_threshold
        gated_reward = torch.where(
            unsafe_mask,
            torch.full_like(reward, self.penalty_magnitude),
            reward,
        )

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
            reward = reward.mean()
            cost = cost.mean()
            gated_reward_mean = gated_reward.mean()
            reward_with_kl_penalty = (old_rewards[:, start:] * mask).sum(dim=-1).mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values[:, start:], mask)

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            unsafe_rate = get_all_reduce_mean(unsafe_rate)
            reward = get_all_reduce_mean(reward)
            cost = get_all_reduce_mean(cost)
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
            'train/reward': reward.item(),
            'train/cost': cost.item(),
            'train/gated_reward': gated_reward_mean.item(),
            'train/unsafe_rate': unsafe_rate.item(),
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
