#!/bin/bash
#SBATCH --job-name=s5-runB-ll
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=12:00:00
#SBATCH --output=stage5_runB_llama_%j.out
#
# Stage 5, RUN B — Llama-family track (optional).
#
# Matched to scripts/stage5-runA-reward-only-llama.sbatch except cost gate
# (cost > 0 → reward := -2). Same Sheared-LLaMA-1.3B actor so Beaver RM/CM share
# the tokenizer and batch_retokenize is skipped (enforced by verify --require-same).

set -euo pipefail

SAFE_RLHF_ENV="${SAFE_RLHF_ENV:-safe-rlhf-cu128}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$SAFE_RLHF_ENV"

export SAFE_RLHF_PROMPT_TEMPLATE=alpaca
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=15
export HF_HUB_DOWNLOAD_TIMEOUT=30
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

REPO_DIR="$HOME/Safe-RL-MinMax/safe-rlhf"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

ACTOR_MODEL="${ACTOR_MODEL:-princeton-nlp/Sheared-LLaMA-1.3B}"
REWARD_MODEL="${REWARD_MODEL:-PKU-Alignment/beaver-7b-unified-reward}"
COST_MODEL="${COST_MODEL:-PKU-Alignment/beaver-7b-unified-cost}"
OUTPUT_DIR="$HOME/Safe-RL-MinMax/output/stage5_runB_llama"
mkdir -p "$OUTPUT_DIR"

echo "node    : $(hostname)"
echo "env     : ${CONDA_DEFAULT_ENV}"
echo "track   : llama (Sheared-LLaMA-1.3B + alpaca template)"
echo "actor   : ${ACTOR_MODEL}"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

python -u scripts/verify_tokenizer_alignment.py \
	--actor "$ACTOR_MODEL" \
	--reward "$REWARD_MODEL" \
	--cost "$COST_MODEL" \
	--require-same

deepspeed --num_gpus 1 --master_port 29548 \
	--module safe_rlhf.algorithms.ppo_cost_gate \
	--train_datasets PKU-SafeRLHF/train:0.126 \
	--actor_model_name_or_path "$ACTOR_MODEL" \
	--reward_model_name_or_path "$REWARD_MODEL" \
	--cost_model_name_or_path "$COST_MODEL" \
	--reward_critic_model_name_or_path "$ACTOR_MODEL" \
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
