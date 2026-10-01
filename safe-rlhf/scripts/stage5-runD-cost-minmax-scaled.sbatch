#!/bin/bash
#SBATCH --job-name=s5-runD
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=12:00:00
#SBATCH --output=stage5_runD_%j.out
#
# Stage 5, RUN D — PPO + cost-scaled Minmax penalty.
#
# Identical to Run C except the replacement:
#   C: cost > threshold → R_unsafe = V_MIN − V_MAX          (flat)
#   D: cost > threshold → R_unsafe = −gap × (1 + severity)
#      severity = min(excess / c_scale, --severity_cap)
#      excess   = max(cost − threshold, 0)
#      c_scale  = running mean of excess on gated samples, floored at 1.0
#
# Same detector, trigger, seed, step budget, and ChatML template as B/C.
# C vs D isolates one change: whether a cost-magnitude slope helps after the
# reward-range bounds lock. Do not retune other knobs or the comparison is
# no longer attributable.
#
# Watch, in addition to C's tags:
#   train/cost_safe, train/cost_unsafe, train/severity, train/c_scale,
#   train/r_unsafe (mean applied penalty on gated samples),
#   train/r_unsafe_base (the unscaled gap).

set -euo pipefail

SAFE_RLHF_ENV="${SAFE_RLHF_ENV:-safe-rlhf-cu128}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$SAFE_RLHF_ENV"

export SAFE_RLHF_PROMPT_TEMPLATE=chatml
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=15
export HF_HUB_DOWNLOAD_TIMEOUT=30
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

REPO_DIR="$HOME/Safe-RL-MinMax/safe-rlhf"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="$HOME/Safe-RL-MinMax/output/stage5_runD"
mkdir -p "$OUTPUT_DIR"

echo "node    : $(hostname)"
echo "env     : ${CONDA_DEFAULT_ENV}"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

deepspeed --num_gpus 1 --master_port 29540 \
	--module safe_rlhf.algorithms.ppo_cost_minmax \
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
	--penalty_floor -50.0 \
	--bound_source reward \
	--scale_penalty_by_cost True \
	--cost_scale_floor 1.0 \
	--severity_cap 1.0 \
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
echo "minmax state:"
cat "$OUTPUT_DIR"/minmax_state.json 2>/dev/null || true
echo
echo "read the metrics with:"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/reward"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/cost"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/cost_safe"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/cost_unsafe"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/unsafe_rate"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/r_unsafe"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/r_unsafe_base"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/severity"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/c_scale"
