#!/bin/bash
#SBATCH --job-name=probe-rm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=probe_rm_%j.out
#
# Stage 4 gate — does the Beaver-7B reward model rank responses sensibly?
#
# Runs on CPU: the 7B model is ~14 GB in bf16 and a `batch` node's RTX 3060 has 12 GB.
# Only a dozen short sequences are scored, so CPU is fine and `batch` has ~88 idle nodes
# rather than the queue on bigbatch/biggpu.

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

python scripts/probe_reward_model.py \
	--model PKU-Alignment/beaver-7b-unified-reward \
	--device cpu \
	--dtype bfloat16
