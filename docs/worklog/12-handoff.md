# 12 — Handoff for a new machine

Written 2026-10-14, before moving the local workspace to a new PC. Read this before changing code, launching a job, or rewriting a claim. The chronological trail stays in the worklogs. This page is the state of the project and the failures that are already paid for.

## What this project is trying to do

Helpfulness-only RLHF can erase a refusal the base model already had. Safe RLHF splits a helpfulness reward from a harm cost. This project asks whether ROSARL’s MinMax replacement beats a **matched fixed penalty** that uses the same detector and the same trigger.

\[
R_{\text{unsafe}} = V_{\min} - V_{\max}
\]

When `cost > 0`, the response’s reward is replaced by that value. Success is beating Run B, not beating reward-only PPO. “A detector helps” is not a MinMax result.

| Run | What it is | State |
|---|---|---|
| A | Reward-only PPO | Done. Refusal of the lock-picking probe dies by about step 500 |
| B | `cost > 0` → fixed reward −2 | Done. Helps that probe. Mean training cost still rises (−2.6 → +1.7) |
| C | `cost > 0` → \(R_{\text{unsafe}}\), floor −50 | Trained and inspected. \(R_{\text{unsafe}}\) locks near **−9.92**. Mean cost still rises (−2.66 → +2.38). Unsafe rate 14% → 64%. Late probe text stays advice-shaped. **Cost rescore vs A/B is still open** |
| D | Cost-scaled MinMax, diagnostic | Trained. Harsher penalty (about −17) does not change the unsafe-rate curve |
| E | PKU PPO-Lag on this same stack | Trained (job 65532). Not the paper recipe. λ falls to ~0.02 while cost is negative, then pins at 5. Cost rises like C (−2.72 → +1.89) |

**Next action:** rescore Run C’s lock-picking generations (base, 50, 250, 500, 750, 950) through `beaver-7b-unified-cost` with the Session 17 protocol (`safe-rlhf/scripts/rescore-lockpicking.sbatch`). Then update chapters 9–10. Do not start a new model stack before that table exists.

One later training job is optional, and only after the rescore: rerun Run E with the same models and one change so λ cannot collapse while cost is still negative. If cost then stays near 0, the climb was the dead multiplier. If it still rises, the reward model beats the constraint on this stack even when the safety weight is on.

## Stack that is locked

| Piece | Choice |
|---|---|
| Actor | `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16, alpha=32, dropout=0.05 |
| Reference | Same actor, adapters disabled |
| Reward | `PKU-Alignment/beaver-7b-unified-reward`, frozen |
| Cost | `PKU-Alignment/beaver-7b-unified-cost`, frozen. Threshold `cost > 0` |
| Critics | Qwen2.5-1.5B-Instruct + LoRA, fresh score heads. Not Beaver |
| Data | `PKU-SafeRLHF/train` fraction 0.126, seed 42, 1 epoch, 1062 steps |
| Template | ChatML (`SAFE_RLHF_PROMPT_TEMPLATE=chatml`) |
| PPO | batch 8, KL 0.02, fp16, ZeRO-2, `--use_torch_adam True`, `clip_range_score` 50, `clip_range_value` 5, `ptx_coeff` 0 |
| Code | `safe-rlhf/` is vendored PKU Safe RLHF. MinMax lives in `algorithms/ppo_cost_minmax/`. PPO-Lag is their `algorithms/ppo_lag/` |

Both Beaver scorers fit on one 48 GB Quadro RTX 8000 together with the 1.5B actor and critic (about 34 GB of weights). Runs A–E finished on that card. A trainable Beaver-7B critic does not fit beside the two frozen scorers, and their trainer rejects it anyway: the critic must share the actor tokenizer. Beaver is Llama; Qwen is not. Leaving the critic flags unset crashes at startup because they default to the reward/cost model path.

## New PC

The cluster is not on the PC. Training, conda, checkpoints, and the Hugging Face cache stay on Wits `mscluster`. The new machine only needs the git repo and SSH.

```bash
git clone https://github.com/wendymaboa/Safe-RL-MinMax.git
cd Safe-RL-MinMax
```

Open that folder as the workspace. Checkpoints, TensorBoard events, and model weights are gitignored. They remain under `$HOME/Safe-RL-MinMax/output/` on the cluster. Do not copy multi-GB outputs onto the laptop.

Local preview of the worklog site:

```bash
python -m http.server 4173 --directory docs
```

Pages: https://wendymaboa.github.io/Safe-RL-MinMax/ from branch `main`, folder `/docs`.

SSH to the cluster with the existing Wits account. This repo does not store the hostname, password, or keys. After login, the project is already at `$HOME/Safe-RL-MinMax`. Pull there before `sbatch`. Do not recreate the conda env unless `conda activate safe-rlhf-cu128` fails.

## Cluster setup

| Item | Value |
|---|---|
| Login node | Has an RTX 2060 SUPER, 8 GB. Use it for downloads and light checks. Not for training |
| `batch` | RTX 3060, 12 GB. Dev only. Cannot hold the 7B scorers |
| `bigbatch` | RTX 3090, 24 GB. Smoke tests. Avoid `mscluster65` and `mscluster83` |
| `biggpu` | Most nodes: 2× Quadro RTX 8000, 48 GB each, 1 TB RAM. Stage 5 target. Exclude `mscluster111` |
| Env | `$HOME/miniconda3`, env `safe-rlhf-cu128` (clone of `safe-rlhf`; covers sm_75 through sm_120) |
| Jobs | `source "$HOME/miniconda3/etc/profile.d/conda.sh"` then `conda activate`. Non-interactive `sbatch` does not read `.bashrc` |
| Repo on cluster | `$HOME/Safe-RL-MinMax/safe-rlhf` |
| Outputs | `$HOME/Safe-RL-MinMax/output/stage5_run{A,B,C,D,E}` |
| HF cache | `~/.cache/huggingface` → `/datasets/wmaboa/hf`. Home quota is about 50 GB |
| GPU selection | This SLURM does **not** use `--gres`. Pick a partition. `sinfo -o "%N %G"` reports `GRES=(null)` |

Every job script starts with `set -euo pipefail`. Launch from `~/Safe-RL-MinMax/safe-rlhf`:

```bash
sbatch scripts/stage5-runA-reward-only.sbatch
sbatch scripts/stage5-runB-cost-gate.sbatch
sbatch scripts/stage5-runC-cost-minmax.sbatch
sbatch scripts/stage5-runD-cost-minmax-scaled.sbatch
sbatch scripts/stage5-runE-ppo-lag.sbatch
sbatch scripts/rescore-lockpicking.sbatch
```

`biggpu` is for debugged jobs. Smoke on `bigbatch` first. Pre-download weights on the login node. Compute nodes fetch Hugging Face at about 0.3 MB/s; the login node is 7–40 MB/s. Jobs already set `HF_HUB_DISABLE_XET=1`.

If the env must be rebuilt, do it on a compute node, not by rerunning a blind `conda env create` from the login node. Known pins: `transformers>=4.37.2,<4.47` (5.x moved `transformers.tokenization_utils`), and do not solve MKL in place. The working torch is a pip cu118 wheel, which does not link conda’s MKL. `safe-rlhf-cu128` was `conda create --clone safe-rlhf`. There is no CUDA JIT (CUDA 11.8 + host gcc 15), so jobs pass `--use_torch_adam True`. fp16, not bf16, on the Quadro RTX 8000 (Turing, `sm_75`, no bf16 tensor cores).

## Problems already faced

### Phase 1 — closed

GPT-2 + Detoxify. Do not cite the early harm rates.

| Problem | What it actually was |
|---|---|
| Reward centered on the wrong side of zero | \(V_{\max}\) had nothing positive to track. Centered reward is \(r = 1 - 2\delta\) |
| Empty generations scored as the prompt | Harm mixed prompt toxicity into the policy. Empty strings are excluded |
| Seed and batch were fake | `torch.manual_seed` was never called, and the loop trained one prompt per step |
| Adaptive KL at tiny batches | Noisy KL, negative KL, skipped PPO updates. Not MinMax-specific. Epochs 4→1, adaptive KL off, β matched |
| MinMax looked perfectly safe | Bounded Detoxify froze \(R_{\text{unsafe}}\) at −2. Later, both policies collapsed to the token `Advertisements` (Detoxify ≈ 0, so the gate never fired). Weak KL (β 0.01 vs 0.2) made it worse |
| Fair close | Matched β 0.2, one seed: MinMax harm about 2.0%, PPO+KL about 2.3%. Viable, not a win |

### Phase 2 — environment and cluster

| Problem | What it actually was |
|---|---|
| Vendored `safe-rlhf/` was not pristine | Phase 1 had wired Detoxify and MinMax into it. Wiped and re-cloned. Commit `3b21444` is the clean baseline. Later diffs are the Phase 2 deviation set |
| Conda missing inside `sbatch` | Batch shells are non-interactive. Source `conda.sh` by absolute path |
| `CondaToSNonInteractiveError`, then pip into system Python | A failed `conda env create` continued because `set -e` was absent |
| `iJIT_NotifyEvent` / MKL | MKL ≥ 2024.1 dropped symbols torch links. In-place `mkl=2024.0.0` did not solve. Pip cu118 torch sidesteps it |
| `transformers` 5.x | Upstream recipe has no upper bound. Pin `<4.47` |
| Session 12 “biggpu is all Blackwell and unusable” | Wrong. The partition is mixed. `mscluster111` is a faulty 96 GB Blackwell (`nvidia-smi` works, CUDA does not, under both cu118 and cu130). Other biggpu nodes are Quadro RTX 8000 and always worked |
| 64 s/step | Turing has no bf16 tensor cores. fp16 is the right mode |
| Home quota | 14 GB Beaver weights do not belong in `$HOME`. Cache is on `/datasets/wmaboa` |

### Phase 2 — training, already invalid or already explained

| Problem | What it actually was |
|---|---|
| Alpaca prompt template on Qwen | Qwen never emits `<|im_end|>`, so generation runs to `max_length`. Any Stage 5 number from before ChatML is void |
| Critic defaults to the Beaver reward model | Tokenizer mismatch raises `ValueError`. Pass a Qwen critic explicitly |
| Reward model “failed” the first probe | It ranks helpfulness, not safety. A refusal scores about −3.8 reward and about −3.1 cost. Harmful instructions score about +4.3 cost. Threshold 0 is PKU’s boundary |
| Run C cost keeps rising | The gate does not touch samples with `cost ≤ 0`. Those are trained as reward-only PPO, and the reward model scores refusals as unhelpful, so the policy walks across 0. Logged `train/cost` is the Beaver score of the rollout, not the replaced label. The value clip is 5 and \(R_{\text{unsafe}}\) is about −10, so the critic cannot represent the penalty |
| Run D did not fix that | Scaling the penalty by how far cost is above 0 still left unsafe rate on C’s curve. `c_scale` chases the unsafe mean |
| PPO-Lag “should have worked because it is their code” | The update is theirs. The recipe is not. Their script uses Alpaca-7B, critics initialized from the cost and reward models, `kl_coeff` 0.01, and `ptx_coeff` 16. Run E kept λ’s hyperparameters and dropped PTX, used Qwen LoRA, and used fresh Qwen critics so the job would match A–D and fit. Qwen-Instruct starts near cost −2.7, so their \(\log\lambda\) update drives λ from 1 to ~0.02. The step size is proportional to λ, so it cannot recover until late, then it hits the cap of 5 with ~200 steps left. Cost never returns to 0 |
| Shared preamble on the lock-picking probe | Real, but Session 17 rescoring says the late A/B gap is not only surface form |

## Do not redo

- Do not claim MinMax beats B. The rescore is not done, and late C text does not show B’s police pivot.
- Do not claim Run E reproduced Safe RLHF, or that PPO-Lag is broken upstream.
- Do not load Beaver as a critic on the Qwen track.
- Do not set `SAFE_RLHF_PROMPT_TEMPLATE=alpaca` for Qwen.
- Do not turn on Alpaca PTX for Qwen. That corpus is the template that broke generation.
- Do not pass `--gres`. Do not schedule `mscluster65`, `83`, or `111`.
- Do not treat a rising reward curve as quality. Read the generations.
- Do not put checkpoints or TensorBoard events in `docs/`.

## Where the detail lives

| Need | File |
|---|---|
| Phase 1 notebook | `ppo_minmax_experiment/worklog.md` |
| Phase 2 notebook | `safe-rlhf/worklog.md` |
| Claims that are allowed | `docs/worklog/10-claims.md` |
| Still open | `docs/worklog/11-open-questions.md` |
| Cluster chapter | `docs/worklog/05-cluster.md` |
| Run C | `docs/worklog/phase2/07-run-c.md` |
| Decile numbers for the plots | `scripts/plot_stage5_comparison.py` |
