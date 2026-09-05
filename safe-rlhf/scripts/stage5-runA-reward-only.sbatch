#!/bin/bash
#SBATCH --job-name=s5-runA
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=12:00:00
#SBATCH --output=stage5_runA_%j.out
#
# Stage 5, RUN A — the reference baseline: PPO with a helpfulness reward only.
#
# This is the control in a three-run design:
#
#   A (this)  reward only                      → what plain RLHF does, no safety signal
#   B         reward + fixed penalty, cost > 0 → what HAVING a safety signal buys
#   C         reward + Minmax,        cost > 0 → what the self-calibrating bound buys
#
# B vs A and C vs B are the two comparisons the thesis needs. A alone cannot support a
# claim about Minmax; it exists so that later claims are attributable.
#
# Deliberately NO cost model here — run A is defined by its absence. Adding one is
# run B's job and needs code that does not exist yet.
#
# --exclude=mscluster111: that node's GPU faults under both CUDA 11.8 and 13.0
# (nvidia-smi lists it, CUDA cannot initialise). Same fault as mscluster65 and 83.
# The remaining biggpu nodes are 2 x Quadro RTX 8000, 48 GB each.
#
# Memory, single card: reward 7B ~14 GB + actor 1.5B ~3 GB + critic 1.5B ~3 GB = ~20 GB
# of weights, plus logits tensors (batch x 512 x 151936 vocab) which are the real
# consumer. Batch 8 keeps that near ~2.5 GB per tensor. Comfortable inside 48 GB.
#
# Uses the ORIGINAL `safe-rlhf` env, not safe-rlhf-cu128. RTX 8000 is sm_75, which
# cu118 supports; that env passed Stages 3 and 4 and there is no reason to re-validate
# a working stack on torch 2.14.

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe-rlhf

export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=15
export HF_HUB_DOWNLOAD_TIMEOUT=30
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

REPO_DIR="$HOME/Safe-RL-MinMax/safe-rlhf"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="$HOME/Safe-RL-MinMax/output/stage5_runA"
mkdir -p "$OUTPUT_DIR"

echo "node    : $(hostname)"
echo "env     : ${CONDA_DEFAULT_ENV}"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# ~1000 PPO steps: 38,641 prompts x 0.21 = ~8,100, at 8 prompts per rollout batch.
# Matches Phase 1's 1000-step runs so the two phases stay loosely comparable.
deepspeed --num_gpus 1 --master_port 29537 \
	--module safe_rlhf.algorithms.ppo \
	--train_datasets PKU-SafeRLHF/train:0.21 \
	--actor_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
	--reward_model_name_or_path PKU-Alignment/beaver-7b-unified-reward \
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
	--bf16 True \
	--tf32 True

echo
echo "finished : $(date)"
echo "=== adapter snapshots ==="
ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | wc -l
du -sh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | tail -3
echo
echo "read the metrics with:"
echo "  python scripts/dump_tb.py $OUTPUT_DIR"
