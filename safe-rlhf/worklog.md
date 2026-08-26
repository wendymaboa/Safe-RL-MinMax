# Research Log — Phase 2: Safe RLHF on Qwen + LoRA

Chronological record of what was tried, what was found, and what didn't work,
for the Phase 2 reference implementation: reproducing the official Safe RLHF
algorithm shape (reward model + actor-critic PPO, **no** cost model, **no**
Lagrangian) on Qwen2.5 + LoRA.

Phase 1 (GPT-2 + Detoxify + PPO/Minmax) is closed and lives in
`../ppo_minmax_experiment/worklog.md`. Its final honest result: under matched KL,
Minmax and PPO+KL are essentially tied (2.0% vs 2.3% harm, one seed), and the
binding constraint was the *reward*, not the algorithm — Detoxify is bounded and
gameable, and both policies learned to emit the single token "Advertisements"
rather than learning safety.

Same format as Phase 1: **Did / Found / Concluded**. A dead end is still an entry.
If a "Concluded" turns out to be wrong later, add a new entry saying so rather than
editing the old one.

---

## Session 1 — Resetting the vendored framework to pristine upstream

**Date:** 2026-08-18

**Did:** Before building anything new, checked whether `safe-rlhf/` was actually an
unmodified copy of PKU-Alignment's framework. Cloned upstream fresh into a temp
directory and diffed it against the local tree.

**Found:** It was **not** untouched, contradicting the claim in
`ppo_minmax_experiment/README.md` that "`safe-rlhf/` at the repo root is the PKU
Safe RLHF framework — left untouched." Phase 1 had reached into it:

- Added files: `algorithms/ppo/trainer_minmax.py`, `algorithms/ppo/trainer_detoxify.py`,
  `datasets/raw/beavertails.py`, `safe_rlhf/rewards/` (Detoxify wrapper),
  `notebooks/colab_gpt2_minmax_500steps.ipynb`, and four GPT-2 launch scripts.
- Modified files: `pyproject.toml` / `requirements.txt` (added `detoxify`, `trl==0.11.4`,
  `peft`; tightened pins), `algorithms/ppo/__init__.py` and `main.py` (wired the two custom
  trainers plus `--use_detoxify_reward` / `--use_minmax` / `--max_training_steps`),
  `datasets/raw/__init__.py`, `trainers/rl_trainer.py` (a max-steps early-stop hook),
  and import-compatibility shims in `datasets/base.py`, `utils.py`, and
  `models/score_model/gpt2/modeling_gpt2.py`.

Also confirmed while diffing: **`Qwen2ForScore` is native to upstream**, not a Phase 1
addition. Qwen support for score models comes for free.

**Concluded:** Wiped the directory and replaced it with a fresh upstream clone
(`.git` excluded so it stays a normal tracked directory, not a submodule). Verified
byte-identical against a second independent clone. Committed as `3b21444 reset safe rlhf`,
so from here every `git diff` against that commit is *exactly* our Phase 2 deviation
set — which is also what the thesis needs for "deviations from the official implementation."

**Noted for later:** the removed import shims (`transformers.tokenization_utils` →
`tokenization_utils_base`) were a *legitimate* fix, unlike the Minmax/Detoxify code.
They will be needed again if we land on a `transformers` version where that import moved.

---

## Session 2 — Cluster inventory

**Date:** 2026-08-18

**Did:** Established what hardware is actually reachable on the Wits `mscluster`,
rather than assuming. Ran `nvidia-smi`, `sinfo`, `scontrol show partition`, and
cross-checked against the MSS Community Guidelines (Feb 2024).

**Found:**

| Partition | Nodes | GPU / node | VRAM | System RAM |
|---|---|---|---|---|
| stampede | 40 | 2 × GTX 1060 | 6 GB each | 32 GB |
| bigbatch | 48 | 1 × RTX 3090 | 24 GB | 128 GB |
| **biggpu** | 4–7 | 2 × Quadro RTX 8000 | **48 GB each** | 1 TB |

- `sinfo -o "%N %G"` reports `GRES=(null)` on every node — this cluster does **not**
  tag GPUs as SLURM generic resources. You never pass `--gres`; you select a GPU by
  selecting a partition.
- The login node has its own RTX 2060 SUPER (8 GB). It is not a training resource.
- At time of checking, biggpu was 4/7 `alloc` and 3/7 `down*`. MSS guidance is explicit
  that biggpu is for mature debugged code only, and that September–November has
  historically near-zero headroom.

**Concluded:** Real runs go on **biggpu**; all development and smoke-testing goes on
**bigbatch** first, per the cluster's own escalation etiquette. Memory budget for the
target architecture (see Session 3) is ~20 GB of weights, which fits one 48 GB RTX 8000
comfortably and does *not* fit bigbatch's 24 GB with the 7B reward model resident.

Findings written up as a reference doc (`mscluster Field Guide`) so this does not have
to be rediscovered.

---

## Session 3 — Environment build, and five ways it failed

**Date:** 2026-08-18

**Did:** Built the `safe-rlhf` conda env on a compute node from upstream's own
`conda-recipe.yaml`, plus `peft` (which upstream does not list anywhere).

**Found:** Five distinct failures, in order:

1. **No conda at all.** `which conda` was empty — installed Miniconda into `$HOME`,
   per MSS's recommendation to manage a personal install.
2. **`conda: command not found` inside `sbatch`.** Batch jobs run a *non-interactive*
   shell, which skips the conda-init block in `.bashrc`. Fixed by sourcing
   `$HOME/miniconda3/etc/profile.d/conda.sh` by absolute path at the top of every job.
3. **`CondaToSNonInteractiveError`.** Recent conda refuses non-interactive env creation
   until the `pkgs/main` and `pkgs/r` Terms of Service are accepted. One-time fix.
4. **Silent cascade into the base env.** The job script had no `set -e`, so after
   `conda env create` failed at (3), execution continued: `conda activate` failed,
   and `pip install peft` ran against the node's *system* Python — producing an
   `externally-managed-environment` error and, on a later attempt, installing ~3 GB of
   unpinned CUDA wheels into the base env. **`set -euo pipefail` is now mandatory in
   every job script.**
5. **`ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`.** MKL ≥ 2024.1
   removed the ittnotify symbols PyTorch links against. Fix: pin `mkl=2024.0.0`.

**Also found:** the solver installed `transformers 5.15.0`, because upstream's recipe
says only `transformers >= 4.37` with no upper bound. safe-rlhf imports
`from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy`,
which 5.x moved.

**Concluded:** Two pins are required and both are **deliberate, documented deviations**,
not leftovers: `mkl=2024.0.0` and `transformers>=4.37.2,<4.47`. Note this independently
re-derives the same `transformers` constraint Phase 1 had applied — the pin was correct
then and correct now; only the Detoxify/Minmax code alongside it was Phase-1-specific.

**Status:** environment fix job submitted; verification (`torch.cuda.is_available()`,
device count, the `tokenization_utils` import) **not yet confirmed**.

---

## Session 4 — Reading the trainer before designing around it

**Date:** 2026-08-25

**Did:** Before committing to an architecture, traced how `PPOTrainer` actually moves
tensors between the actor, reward model, and critic — specifically whether a
LLaMA-family reward model can score a Qwen actor's outputs.

**Found:** The two paths are **asymmetric**, and this is decisive:

- **Reward model — bridge exists.** `post_rollout()` (`algorithms/ppo/trainer.py:43-55`)
  checks `if self.reward_tokenizer is not self.tokenizer` and calls `batch_retokenize()`
  to decode and re-encode the sequence. A LLaMA-family reward model scoring a Qwen actor
  is therefore *supported upstream*, not a hack.
- **Critic — no bridge, and a hard failure.** The critic is fed the actor's raw
  `sequence` with no re-tokenization (`trainer.py:61`), and `rl_trainer.py:177-193`
  raises `ValueError` outright if the critic's tokenizer differs from the actor's.
  Worse, `--reward_critic_model_name_or_path` **defaults to the reward model path**, so
  leaving it unset with a Qwen actor and a LLaMA RM crashes at startup.

Also found: the actor is generated from via `self.actor_model.module.generate(...)`
(`rl_trainer.py:411`), i.e. through the DeepSpeed engine's inner module.

**Concluded:** Architecture settled, with one decision *forced* rather than chosen:

| Role | Model | Trains? |
|---|---|---|
| Actor | Qwen2.5-1.5B-Instruct + LoRA | adapters only |
| Reference | same actor, adapters disabled | no (free) |
| Reward | `PKU-Alignment/beaver-7b-unified-reward` | frozen |
| Critic | `Qwen2ForScore` on Qwen base + LoRA | adapters + score head |

The critic **must** be Qwen-family — not a design preference but a constraint imposed by
the trainer. This is also cheaper than PKU's RM-initialised 7B critic. Budget: ~20 GB of
weights in bf16, which fits one RTX 8000.

Two further scope decisions recorded: **skip SFT** (Qwen2.5-Instruct already ships
instruction-tuned; PKU needed SFT only because raw LLaMA-7B cannot follow instructions),
and **use PKU's released reward model** rather than training one, which removes an entire
training stage and is more faithful to "official shape" than a home-trained RM.

---

## Session 5 — LoRA plumbing implemented

**Date:** 2026-08-25

**Did:** Implemented LoRA support, which upstream lacks entirely — `peft` appears in no
import anywhere in the framework, and `load_pretrained_models()` loads full weights
straight into DeepSpeed. Four changes, each gated so that `--use_lora False` reproduces
upstream byte-for-byte:

1. **`models/pretrained.py`** — `load_pretrained_models()` takes an optional
   `lora_config: LoraConfig | None` and, when given, wraps the model via
   `get_peft_model()`. Placed *after* `resize_tokenizer_embedding()`, since resizing
   embeddings is cleaner on a raw HF model than through the PEFT wrapper.
2. **`algorithms/ppo/main.py`** — a new `lora` argument group: `--use_lora`, `--lora_r`
   (default 16), `--lora_alpha` (32), `--lora_dropout` (0.05), `--lora_target_modules`
   (default `None`).
3. **`trainers/rl_trainer.py`** — builds two *different* configs. The actor gets
   `task_type=TaskType.CAUSAL_LM` so `get_peft_model` returns a `PeftModelForCausalLM`
   with a proper `generate()`. The critic gets **`modules_to_save=['score_head']`** and
   no task type.
4. **`trainers/rl_trainer.py`** — new `AdapterDisabledReference` class replacing the
   second full model copy (see below).

**Found / reasoned:**

- **The `score_head` trap.** `get_peft_model()` freezes every parameter that is not an
  adapter. The critic's `score_head` is a freshly-initialised `nn.Linear` created by
  `ScoreModelMixin.init_score_head()`. Without `modules_to_save`, it would stay frozen at
  random initialisation for the entire run: the critic never learns, advantages become
  noise, and PPO trains against garbage **while appearing to run perfectly**. This is the
  single highest-risk line in the change.
- **π_ref is free under LoRA.** LoRA leaves base weights untouched, so the reference
  policy is just the actor with adapters disabled. `AdapterDisabledReference` wraps the
  actor engine and calls it inside `with ...disable_adapter():`, saving a full model copy
  (~3 GB at 1.5B, ~6 GB at 3B) and one DeepSpeed engine. Implemented as a callable proxy
  specifically so `self.actor_reference_model(...)` keeps working unchanged in `ppo`,
  `ppo_lag`, and `ppo_reward_shaping` — no algorithm trainer was touched.
- **Guarded with `getattr(self.args, 'use_lora', False)`**, because `rl_trainer.py` is
  shared with `ppo_lag` and `ppo_reward_shaping`, whose parsers have no LoRA flags. A
  direct attribute access would break those algorithms.
- **PEFT's Qwen2 defaults are narrower than assumed.** Checked the v0.20.0 source:
  `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` contains
  `"qwen2": ["q_proj", "v_proj"]` — query and value projections only, following the
  original LoRA paper. Not `k_proj`, not `o_proj`, nothing in the MLP. So
  `--lora_target_modules None` resolves without error but is conservative. Whether to
  widen it to all attention projections plus MLP is an open question to settle with
  measurements, not assumption.

**Concluded:** Diff against pristine upstream is **3 files, +108/−13** (the deletions are
re-indentation into `else:` branches, not removals), plus one new file
`scripts/verify_lora.py`.

**This is verified as syntax only. Nothing has been executed.** `peft` and `torch` are not
installed locally, so no import check, no model load, no runtime behaviour has been
observed. Specifically unproven: that `modules_to_save=['score_head']` resolves against
`Qwen2ForScore`'s actual module naming; that DeepSpeed accepts a `PeftModel` where it
expects a `PreTrainedModel`; and that `AdapterDisabledReference` behaves correctly under
ZeRO-3 parameter partitioning.

**Also written:** `scripts/verify_lora.py`, which exercises the real code path on a small
Qwen2 checkpoint and asserts (a) the actor wraps as `PeftModelForCausalLM` and can still
generate, (b) the critic's `score_head` survives as trainable, (c) omitting `lora_config`
leaves the model completely untouched, and (d) `disable_adapter()` genuinely restores base
behaviour. Note (d) has a subtlety: LoRA initialises the `B` matrix to zeros, so an
untrained adapter is a no-op and a naive enabled-vs-disabled comparison passes trivially.
The script perturbs `B` first so the test means something.

---

## Open tasks

**Blocking the first real run:**

- [ ] Confirm the environment fix job succeeded (`torch.cuda.is_available()`, device
      count, `tokenization_utils` import) — Session 3 left this unverified.
- [ ] Run `scripts/verify_lora.py` on the cluster. Expect the `score_head` assertion to be
      the one that fails, if any does.
- [ ] **Adapter save / resume is not implemented.** Upstream calls `save_checkpoint` /
      `save_pretrained` on the actor expecting a plain HF model; with a `PeftModel` this
      either saves the wrong object or writes a full merged checkpoint instead of a small
      adapter. This fails *after* a long run rather than at startup — the worst time.
- [ ] Decide the prompt template: upstream's Alpaca-style `configs/constants.py` prompt
      vs Qwen's native ChatML. Qwen2.5-Instruct was tuned on ChatML, so keeping Alpaca
      framing imposes avoidable distribution shift, but switching means editing a
      constants file and deviating further from upstream.

**Before trusting any result:**

- [ ] Verify `batch_retokenize` round-trips Qwen text through the LLaMA tokenizer without
      meaningful drift — it decodes with `skip_special_tokens=True` and re-encodes.
      If scores look wrong at smoke-test time, the fallback is training a small Qwen RM.
- [ ] Sanity-check the Beaver reward model separates obviously-harmful from
      obviously-helpful text *before* spending biggpu hours on it.
- [ ] Assert the critic's `score_head` has `requires_grad=True` after wrapping, as an
      in-run check and not just a one-off test.

**Deferred by decision, not oversight:**

- [ ] Widening `--lora_target_modules` beyond PEFT's `["q_proj", "v_proj"]` default —
      revisit with measurements at smoke-test stage.
- [ ] A separate flag for a full-weight critic with a LoRA actor. Currently `--use_lora`
      turns both on together; a small critic with a full-precision value head is a
      legitimate configuration if the critic underfits.
- [ ] PPO-Lag, cost models, reward shaping, GRPO, and the Minmax penalty. All explicitly
      out of scope until this reference build produces a trustworthy number.
