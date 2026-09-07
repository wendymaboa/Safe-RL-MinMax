# Appendix B — Raw worklogs

The textbook is curated. Audit trails live here:

| Document | Scope |
|---|---|
| [`safe-rlhf/worklog.md`](https://github.com/wendymaboa/Safe-RL-MinMax/blob/main/safe-rlhf/worklog.md) | Phase 2 — Safe RLHF on Qwen + LoRA (Sessions 1–18+) |
| [`ppo_minmax_experiment/worklog.md`](https://github.com/wendymaboa/Safe-RL-MinMax/blob/main/ppo_minmax_experiment/worklog.md) | Phase 1 — GPT-2 + Detoxify + MinMax |

## Phase 2 session index (for cross-reference)

| Session | Topic |
|---|---|
| 1 | Reset vendored framework to upstream |
| 2–3 | Cluster inventory; environment failures |
| 4–5 | Trainer reading; LoRA plumbing |
| 6–7 | MKL wall; LoRA verified |
| 8–10 | Template decision; CUDA JIT; PPO loop closed |
| 11 | Reward vs cost probe |
| 12–13 | biggpu / Blackwell confusion and correction |
| 14–15 | Run A bugs + valid results |
| 16–17 | Run B inspect + cost rescoring |
| 18 | Run C implementation |

## Other durable artifacts

- `safe-rlhf/results/stage5_runA/` — archived Run A metrics & generations
- `docs/Slides/` — presentation decks
- `docs/phase_plan.pdf`, `docs/mscluster_commands.pdf` — planning / ops PDFs
