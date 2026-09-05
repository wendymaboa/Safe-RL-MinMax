#!/bin/bash
#SBATCH --job-name=probe-pm
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=probe_pm_%j.out
#
# Stage 4 gate — do the Beaver preference models rank responses as intended?
#
#   sbatch scripts/probe-preference-model.sbatch reward
#   sbatch scripts/probe-preference-model.sbatch cost
#
# bigbatch, not batch: 7B in bf16 is ~14 GB, which fits an RTX 3090 (24 GB) but not an
# RTX 3060 (12 GB). --require-gpu makes the job fail immediately rather than falling
# back to CPU, where a single forward pass takes minutes and an earlier attempt burned
# a full hour before hitting the wall time.
#
# mscluster65 has a faulty GPU ("Unable to determine the device handle for GPU0") and is
# excluded below. Remove the --exclude if it gets fixed.

set -euo pipefail

SCORE_TYPE="${1:-reward}"
if [[ "${SCORE_TYPE}" == "cost" ]]; then
	MODEL="PKU-Alignment/beaver-7b-unified-cost"
else
	MODEL="PKU-Alignment/beaver-7b-unified-reward"
fi

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

echo "node       : $(hostname)"
echo "score type : ${SCORE_TYPE}"
echo "model      : ${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u scripts/probe_reward_model.py \
	--model "${MODEL}" \
	--score-type "${SCORE_TYPE}" \
	--device auto \
	--dtype bfloat16 \
	--require-gpu
