#!/bin/bash
#SBATCH --job-name=inspect-A-ll
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=00:30:00
#SBATCH --output=inspect_runA_llama_%j.out
#
# Inspect Stage 5 Run A (Llama track). Same prompts/seeds as inspect-runA.sbatch;
# base model and template match the llama training launch.

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
RUN_DIR="$HOME/Safe-RL-MinMax/output/stage5_runA_llama"

echo "node : $(hostname)"
echo "base : ${ACTOR_MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -u scripts/generate_from_adapter.py \
	--base "$ACTOR_MODEL" \
	--adapters \
		"$RUN_DIR/checkpoint-50" \
		"$RUN_DIR/checkpoint-250" \
		"$RUN_DIR/checkpoint-500" \
		"$RUN_DIR/checkpoint-750" \
		"$RUN_DIR/checkpoint-950" \
	--max-new-tokens 160
