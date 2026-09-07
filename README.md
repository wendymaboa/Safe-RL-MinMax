# Safe-RL-MinMax

Research workspace for safe reinforcement learning with a MinMax penalty, including proposal materials, experiments, and the PKU Safe RLHF framework.

## Repository layout

| Path | Description |
|------|-------------|
| `csam-template/` | Active Wits CSAM LaTeX proposal (source of truth for the written proposal) |
| `latex-proposal/` | Earlier / archived proposal materials |
| `ppo_minmax_experiment/` | PPO + MinMax penalty experiment on GPT-2 with Detoxify / BeaverTails |
| `safe-rlhf/` | PKU-Alignment Safe RLHF (Beaver) framework — vendored reference |

## Quick start (experiment)

```bash
cd ppo_minmax_experiment
# Install PyTorch for your CUDA setup first, then:
pip install -r requirements.txt
python run_experiment.py --smoke-test --seed 42
```

See `ppo_minmax_experiment/README.md` for full training, evaluation, and design notes.

## Research worklog (Docsify + GitHub Pages)

Guided notes + **full mirrors of both worklogs** live under `docs/`:

| On site | Source file |
|---|---|
| `#/worklog/worklogs/phase1` | `ppo_minmax_experiment/worklog.md` |
| `#/worklog/worklogs/phase2` | `safe-rlhf/worklog.md` |

After editing either worklog:

```bash
python scripts/sync_worklogs_to_docs.py
```

- Preview: `python -m http.server 4173 --directory docs`
- Pages: **Settings → Pages → `main` / `/docs`**
- URL: https://wendymaboa.github.io/Safe-RL-MinMax/

## Notes

- Large experiment artifacts (checkpoints, local model weights, run outputs) are gitignored.
- IDE folders such as `.cursor/` are not tracked.
