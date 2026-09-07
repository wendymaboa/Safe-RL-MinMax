#!/bin/bash
#SBATCH --job-name=s5-runB
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=12:00:00
#SBATCH --output=stage5_runB_%j.out
#
# Stage 5, RUN B — PPO + threshold-gated fixed cost penalty.
#
# Same actor/reward/critic config as Run A (scripts/stage5-runA-reward-only.sbatch), same
# seed, same step budget, same ChatML template. The ONLY difference from Run A is the new
# --cost_model_name_or_path plus the gate itself, implemented in
# safe_rlhf/algorithms/ppo_cost_gate/trainer.py: if a response's cost exceeds
# --cost_threshold, its reward is replaced outright with --penalty_magnitude.
#
# This is the control for Run C (Minmax): both fire on cost > threshold, the identical
# condition, and differ only in the penalty's magnitude — fixed here, self-calibrated
# (V_MIN - V_MAX) there. B vs A isolates "does having a safety signal help at all". C vs
# B isolates "does self-calibrating the penalty help over a fixed one" — the actual
# thesis question.
#
# Memory, single card: reward 7B + cost 7B + actor 1.5B + critic 1.5B ~= 34 GB of
# weights, plus logits/activations. Still fits a 48 GB Quadro RTX 8000 with headroom,
# but there is less margin than Run A had (~20 GB) — watch the first few steps for OOM.
#
# cost_threshold=0.0 matches the Stage 4 probe: beaver-7b-unified-cost scored every
# benign response negative and the harmful lock-picking response +4.34, so zero is a
# principled boundary from that data, not an arbitrary pick.
#
# penalty_magnitude=-2.0 is the trainer default; kept explicit here for the record since
# it is the number Run C's self-calibrated bound will be compared against.

set -euo pipefail

SAFE_RLHF_ENV="${SAFE_RLHF_ENV:-safe-rlhf-cu128}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$SAFE_RLHF_ENV"

# Qwen was trained on ChatML and only emits its eos token (<|im_end|>) in that
# format. Under the Alpaca template it never terminates — see constants.py.
export SAFE_RLHF_PROMPT_TEMPLATE=chatml
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=15
export HF_HUB_DOWNLOAD_TIMEOUT=30
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

REPO_DIR="$HOME/Safe-RL-MinMax/safe-rlhf"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="$HOME/Safe-RL-MinMax/output/stage5_runB"
mkdir -p "$OUTPUT_DIR"

echo "node    : $(hostname)"
echo "env     : ${CONDA_DEFAULT_ENV}"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

deepspeed --num_gpus 1 --master_port 29538 \
	--module safe_rlhf.algorithms.ppo_cost_gate \
	--train_datasets PKU-SafeRLHF/train:0.126 \
	--actor_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
	--reward_model_name_or_path PKU-Alignment/beaver-7b-unified-reward \
	--cost_model_name_or_path PKU-Alignment/beaver-7b-unified-cost \
	--reward_critic_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
	--use_lora True \
	--lora_r 16 \
	--lora_alpha 32 \
	--lora_dropout 0.05 \
	--max_length 512 \
	--temperature 1.0 \
	--num_return_sequences 1 \
	--repetition_penalty 1.0 \
	--trust_remote_code False \
	--epochs 1 \
	--update_iters 1 \
	--per_device_prompt_batch_size 8 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 1 \
	--actor_lr 1e-5 \
	--actor_weight_decay 0.01 \
	--actor_lr_scheduler_type cosine \
	--actor_lr_warmup_ratio 0.03 \
	--critic_lr 5e-6 \
	--critic_weight_decay 0.0 \
	--critic_lr_scheduler_type constant \
	--critic_lr_warmup_ratio 0.03 \
	--normalize_reward False \
	--normalize_cost False \
	--cost_threshold 0.0 \
	--penalty_magnitude -2.0 \
	--seed 42 \
	--use_torch_adam True \
	--kl_coeff 0.02 \
	--clip_range_ratio 0.2 \
	--clip_range_score 50.0 \
	--clip_range_value 5.0 \
	--save_interval 50 \
	--output_dir "$OUTPUT_DIR" \
	--log_type tensorboard \
	--zero_stage 2 \
	--fp16 True \
	--tf32 True

echo
echo "finished : $(date)"
echo "=== adapter snapshots ==="
ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | wc -l
du -sh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | tail -3
echo
echo "read the metrics with:"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/reward"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/unsafe_rate"
