#!/bin/bash
#SBATCH --job-name=smoke-ppo
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=smoke_ppo_%j.out
#
# Stage 3 smoke test — does the PPO loop run end to end with LoRA?
#
# Deliberately tiny: the question is "does it run", not "does it learn".
#
#   * Actor and critic are both Qwen2.5-0.5B. The critic MUST share the actor's
#     tokenizer — rl_trainer.py raises ValueError otherwise, and its default is
#     the reward model path, which would be wrong here.
#   * The reward model is `gpt2`, chosen precisely because its tokenizer DIFFERS
#     from Qwen's. That forces post_rollout() down the batch_retokenize() branch,
#     which is the same path the real Beaver-7B reward model will take. A Qwen
#     stand-in would have skipped that branch entirely and tested nothing.
#   * Run length is capped with a dataset proportion (~38 prompts, ~19 steps),
#     because upstream has no --max_training_steps flag.
#   * save_interval=5 so the adapter-snapshot code actually gets exercised;
#     its default is 1000000, i.e. never.
#
# Fits comfortably on a `batch` node (RTX 3060, 12 GB): ~1.2 GB of weights.

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate safe-rlhf

export HF_HUB_DISABLE_XET=1                      # xet backend is very slow on compute nodes
export TRITON_CACHE_DIR="/tmp/${USER}/triton"    # node-local; NFS makes DeepSpeed hang on exit
mkdir -p "$TRITON_CACHE_DIR"

# Compute nodes fetch from HuggingFace at ~0.3 MB/s, and a stalled request hangs the job
# silently until the time limit. Offline mode is NOT usable here: TRANSFORMERS_OFFLINE is
# aliased to HF_HUB_OFFLINE in modern transformers, and `datasets` resolves a dataset
# module through the Hub API before reading its cache, with no offline fallback — so any
# offline flag breaks dataset loading outright.
#
# Instead, keep the network available but bound it. A stalled connection now raises after
# seconds rather than blocking for the whole job. All models must still be pre-fetched on
# the login node (`huggingface-cli download <repo> --include ...`); these timeouts are a
# safety net, not a substitute.
export HF_HUB_ETAG_TIMEOUT=15
export HF_HUB_DOWNLOAD_TIMEOUT=30

REPO_DIR="$HOME/Safe-RL-MinMax/safe-rlhf"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="$HOME/Safe-RL-MinMax/output/smoke"
mkdir -p "$OUTPUT_DIR"

echo "node : $(hostname)"
echo "env  : ${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=name,memory.total --format=csv

deepspeed --num_gpus 1 --master_port 29531 \
	--module safe_rlhf.algorithms.ppo \
	--train_datasets PKU-SafeRLHF/train:0.001 \
	--actor_model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
	--reward_model_name_or_path gpt2 \
	--reward_critic_model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
	--use_lora True \
	--lora_r 8 \
	--lora_alpha 16 \
	--lora_dropout 0.05 \
	--max_length 128 \
	--temperature 1.0 \
	--num_return_sequences 1 \
	--repetition_penalty 1.0 \
	--trust_remote_code False \
	--epochs 1 \
	--update_iters 1 \
	--per_device_prompt_batch_size 2 \
	--per_device_train_batch_size 2 \
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
	--save_interval 5 \
	--output_dir "$OUTPUT_DIR" \
	--log_type tensorboard \
	--zero_stage 2 \
	--bf16 True \
	--tf32 True

echo
echo "=== adapter snapshots written ==="
ls -la "$OUTPUT_DIR" | head -20
echo
echo "=== size of each snapshot (should be a few MB, not GB) ==="
du -sh "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "no checkpoint-* directories found"
