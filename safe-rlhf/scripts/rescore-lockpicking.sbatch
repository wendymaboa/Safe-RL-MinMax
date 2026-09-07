#!/bin/bash
#SBATCH --job-name=rescore-lp
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --exclude=mscluster111
#SBATCH --time=00:20:00
#SBATCH --output=rescore_lp_%j.out
#
# Score the 12 lock-picking generations (Run A x6, Run B x6, same checkpoints) through
# the actual cost model, to settle Session 16's open question quantitatively instead of
# by reading tone: does Run B's shared "Protecting personal property..." preamble
# register as lower cost to the model that would need to be fooled by it, or does the
# same phrase (which Run A produces too, worded differently, with NO cost model at all)
# score about the same either way?

set -euo pipefail

SAFE_RLHF_ENV="${SAFE_RLHF_ENV:-safe-rlhf-cu128}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$SAFE_RLHF_ENV"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=15
export HF_HUB_DOWNLOAD_TIMEOUT=30
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

REPO_DIR="$HOME/Safe-RL-MinMax/safe-rlhf"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "node : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -u scripts/rescore_lockpicking.py
