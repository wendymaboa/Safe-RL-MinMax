# 5. Cluster and environment

This chapter is operational, not theoretical — but Phase 2’s validity depends on it.

## Hardware map (as learned the hard way)

Wits `mscluster` partitions matter:

| Partition | Typical GPU | Notes |
|---|---|---|
| `batch` | RTX 3060 12 GB | Often constrained |
| `bigbatch` | RTX 3090 24 GB | Usable; avoid known faulty nodes |
| `biggpu` | Quadro RTX 8000 48 GB *or* Blackwell 96 GB | Mixed estate; exclude `mscluster111` |

Session 12 wrongly concluded biggpu was unusable under CUDA 11.8 after hitting Blackwell nodes. Session 13 corrected this: Quadro RTX 8000 nodes work; one node is faulty.

Stage 5 A/B/C target: **biggpu**, Quadro RTX 8000, exclude `mscluster111`.

## Environment constraints that shaped code

- **No CUDA extension JIT** on the login/compute path we use (CUDA 11.8 + new host gcc) → `--use_torch_adam True` to avoid FusedAdam JIT.
- Prefer **fp16** on Turing (Quadro RTX 8000): no bf16 tensor cores.
- HF cache on `/datasets/wmaboa` (home quota is tight).
- Pre-download models on the login node; set `HF_HUB_DISABLE_XET=1`.

## Launch pattern

```bash
cd ~/Safe-RL-MinMax/safe-rlhf
sbatch scripts/stage5-runA-reward-only.sbatch
sbatch scripts/stage5-runB-cost-gate.sbatch
sbatch scripts/stage5-runC-cost-minmax.sbatch
```

Inspect matched checkpoints:

```bash
sbatch scripts/inspect-runA.sbatch
sbatch scripts/inspect-runB.sbatch
sbatch scripts/inspect-runC.sbatch
```
