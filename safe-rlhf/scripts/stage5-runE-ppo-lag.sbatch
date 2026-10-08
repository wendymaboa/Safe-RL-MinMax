#!/bin/bash
#SBATCH --job-name=s5-runE
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=12:00:00
#SBATCH --output=stage5_runE_%j.out
#
# Stage 5, RUN E — PKU PPO-Lag on the same stack as Runs A–D.
#
# Same actor, Beaver reward model, Beaver cost model, ChatML template, data
# fraction, seed, batch, KL, and clip ranges as Run B. The update is PKU's:
#
#   advantage = (A_reward - λ * A_cost) / (1 + λ)
#
# λ grows when the windowed mean cost is above --threshold and shrinks when
# it is below. Threshold 0 matches the gate's cost > 0 boundary as a constraint
# on the average, not as a per-reply switch.
#
# Lambda hyperparameters are PKU's script defaults (λ starts at 1, lr 0.1,
# capped at 5, window 128). ptx_coeff stays 0 and kl_coeff stays 0.02 so this
# is not their Alpaca-7B recipe; those two knobs would confound the comparison.
#
# The cost critic is Qwen + LoRA, same as the reward critic. Beaver-7B cannot
# be the critic: its tokenizer is not Qwen's, and a trainable 7B does not fit
# next to the two frozen Beaver scorers. Gradient checkpointing stays off, as
# in Runs A–D: with LoRA the frozen inputs do not require grad, and checkpointing
# made the first actor loss non-scalar (job 64646).

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

OUTPUT_DIR="$HOME/Safe-RL-MinMax/output/stage5_runE"
mkdir -p "$OUTPUT_DIR"

echo "node    : $(hostname)"
echo "env     : ${CONDA_DEFAULT_ENV}"
echo "started : $(date)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

deepspeed --num_gpus 1 --master_port 29541 \
	--module safe_rlhf.algorithms.ppo_lag \
	--train_datasets PKU-SafeRLHF/train:0.126 \
	--actor_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
	--reward_model_name_or_path PKU-Alignment/beaver-7b-unified-reward \
	--cost_model_name_or_path PKU-Alignment/beaver-7b-unified-cost \
	--reward_critic_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
	--cost_critic_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
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
	--threshold 0.0 \
	--lambda_init 1.0 \
	--lambda_lr 0.1 \
	--lambda_max 5.0 \
	--lambda_update_delay_steps 0 \
	--episode_cost_window_size 128 \
	--seed 42 \
	--use_torch_adam True \
	--kl_coeff 0.02 \
	--clip_range_ratio 0.2 \
	--clip_range_score 50.0 \
	--clip_range_value 5.0 \
	--ptx_coeff 0.0 \
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
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/cost"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/lambda"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/episode_cost"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/cost_advantage"
echo "  python scripts/dump_tb.py $OUTPUT_DIR --trend --tag train/reward_advantage"
