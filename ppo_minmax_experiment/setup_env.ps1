# Create and populate the ppo-minmax conda environment.
# Run from ppo_minmax_experiment/:  .\setup_env.ps1

$ErrorActionPreference = "Stop"
$EnvName = "ppo-minmax"

Write-Host "Creating conda env '$EnvName' (Python 3.10)..."
conda create -n $EnvName python=3.10 -y

Write-Host "Installing PyTorch (CUDA 11.8)..."
conda run -n $EnvName pip install torch --index-url https://download.pytorch.org/whl/cu118

Write-Host "Installing project requirements..."
conda run -n $EnvName pip install -r requirements.txt

Write-Host "Verifying installs..."
conda run -n $EnvName python -c @"
import torch
import transformers
import trl
import detoxify
import datasets
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
print('transformers', transformers.__version__)
print('trl', trl.__version__)
print('datasets', datasets.__version__)
print('OK')
"@

Write-Host ""
Write-Host "Done. Activate with:  conda activate $EnvName"
