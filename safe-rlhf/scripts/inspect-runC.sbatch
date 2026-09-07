#!/bin/bash
#SBATCH --job-name=inspect-runC
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=00:30:00
#SBATCH --output=inspect_runC_%j.out
#
# Read what Run C's policy became — same prompts, same seeds as inspect-runA.sbatch
# and inspect-runB.sbatch, so A/B/C are directly comparable.

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

RUN_DIR="$HOME/Safe-RL-MinMax/output/stage5_runC"

echo "node : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -u scripts/generate_from_adapter.py \
	--base Qwen/Qwen2.5-1.5B-Instruct \
	--adapters \
		"$RUN_DIR/checkpoint-50" \
		"$RUN_DIR/checkpoint-250" \
		"$RUN_DIR/checkpoint-500" \
		"$RUN_DIR/checkpoint-750" \
		"$RUN_DIR/checkpoint-950" \
	--max-new-tokens 160
