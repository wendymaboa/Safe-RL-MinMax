#!/bin/bash
#SBATCH --job-name=probe-rm
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=probe_rm_%j.out
#
# Stage 4 gate — does the Beaver-7B reward model rank responses sensibly?
#
# bigbatch, not batch: the model is ~14 GB in bf16, which fits an RTX 3090 (24 GB) but
# not an RTX 3060 (12 GB). An earlier CPU attempt hit the wall time — bfloat16 on CPU
# falls back to unoptimised kernels and a single 7B forward pass takes minutes.
# The script auto-detects the device, so it uses the GPU here without being told.

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

echo "node : $(hostname)"
echo "env  : ${CONDA_DEFAULT_ENV}"
free -g | head -2

nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u scripts/probe_reward_model.py \
	--model PKU-Alignment/beaver-7b-unified-reward \
	--device auto \
	--dtype bfloat16
