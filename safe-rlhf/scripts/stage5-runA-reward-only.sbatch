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
# biggpu is HETEROGENEOUS: some nodes are 2 x Quadro RTX 8000 (48 GB, Turing sm_75),
# at least one is RTX PRO 6000 Blackwell (96 GB, sm_120). The cu118 env has no kernels
# for sm_120 and dies with "no kernel image is available for execution on the device",
# so this defaults to safe-rlhf-cu128, whose arch list spans sm_75..sm_120 and therefore
# runs on whichever node SLURM gives us. Override with SAFE_RLHF_ENV=safe-rlhf.
#
# mscluster111 is excluded separately: its GPU faults under BOTH CUDA toolkits
# (nvidia-smi lists it, CUDA cannot initialise) — same fault as mscluster65 and 83.
#
# Memory, single card: reward 7B ~14 GB + actor 1.5B ~3 GB + critic 1.5B ~3 GB = ~20 GB
# of weights, plus logits tensors (batch x 512 x 151936 vocab) which are the real
# consumer. Batch 8 keeps that near ~2.5 GB per tensor. Fits the smaller 48 GB cards.

set -euo pipefail

# Env is selectable: safe-rlhf is cu118 (sm_75..sm_90) and safe-rlhf-cu128 is
# torch 2.14+cu130 (sm_75..sm_120). biggpu is heterogeneous — some nodes are
# Blackwell sm_120, where cu118 fails with "no kernel image is available".
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

OUTPUT_DIR="$HOME/Safe-RL-MinMax/output/stage5_runA"
mkdir -p "$OUTPUT_DIR"

echo "node    : $(hostname)"
echo "env     : ${CONDA_DEFAULT_ENV}"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# ~1000 PPO steps. The first attempt used proportion 0.21 and produced 1671 steps, so
# the ratio is calibrated from that observation rather than from the dataset row count.
#
# fp16, not bf16: Quadro RTX 8000 is Turing (sm_75), which has native fp16 tensor cores
# but NO hardware bfloat16. The first attempt ran bf16 at 64 s/step — roughly 30 hours
# for the full run, against a 12-hour limit. Watch the first ~20 steps for NaN in
# train/reward or train/actor_loss; fp16 needs loss scaling and DeepSpeed handles it,
# but that is the failure mode to look for.
deepspeed --num_gpus 1 --master_port 29537 \
	--module safe_rlhf.algorithms.ppo \
	--train_datasets PKU-SafeRLHF/train:0.126 \
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
	--fp16 True \
	--tf32 True

echo
echo "finished : $(date)"
echo "=== adapter snapshots ==="
ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | wc -l
du -sh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | tail -3
echo
echo "read the metrics with:"
echo "  python scripts/dump_tb.py $OUTPUT_DIR"
