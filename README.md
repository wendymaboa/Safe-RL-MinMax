# Safe-RL-MinMax

Research workspace for safe reinforcement learning with a MinMax penalty: Phase 1 (GPT-2 + Detoxify), Phase 2 (Qwen + LoRA + PKU Safe RLHF), proposal materials, and a Docsify worklog site.

## Layout

| Path | Role |
|------|------|
| [`ppo_minmax_experiment/`](ppo_minmax_experiment/) | **Phase 1** — GPT-2 + Detoxify + MinMax (`worklog.md`) |
| [`safe-rlhf/`](safe-rlhf/) | **Phase 2** — vendored PKU Safe RLHF + Qwen/LoRA (`worklog.md`, Stage 5 A/B/C) |
| [`docs/worklog/`](docs/worklog/) | Docsify site — full worklog mirrors + guided notes |
| [`docs/materials/`](docs/materials/) | Proposals, decks, notebooks, ops PDFs |
| [`docs/Slides/`](docs/Slides/) | Supervisor / progress slide decks |
| [`scripts/`](scripts/) | Repo helpers (e.g. sync worklogs → Pages) |
| [`csam-template/`](csam-template/) | Active Wits CSAM LaTeX proposal |
| [`latex-proposal/`](latex-proposal/) | Earlier / archived proposal materials |

Root stays minimal: `README.md`, `.gitignore`, and the folders above.

## Worklog site (GitHub Pages)

| On site | Source file |
|---|---|
| `#/worklog/worklogs/phase1` | `ppo_minmax_experiment/worklog.md` |
| `#/worklog/worklogs/phase2` | `safe-rlhf/worklog.md` |

```bash
python scripts/sync_worklogs_to_docs.py   # after editing either worklog
python -m http.server 4173 --directory docs
```

- Pages: **Settings → Pages → `main` / `/docs`**
- URL: https://wendymaboa.github.io/Safe-RL-MinMax/

## Quick start (Phase 1 experiment)

```bash
cd ppo_minmax_experiment
# Install PyTorch for your CUDA setup first, then:
pip install -r requirements.txt
python run_experiment.py --smoke-test --seed 42
```

Phase 2 cluster launches live under `safe-rlhf/scripts/` (e.g. `stage5-runA-*.sbatch`).

## Notes

- Checkpoints, local weights, and large run outputs are gitignored.
- `.cursor/` is not tracked.
