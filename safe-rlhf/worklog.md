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

## Session 6 — The MKL wall, and why the "standard fix" was unsolvable

**Date:** 2026-08-26 → 2026-08-27

**Did:** Tried to apply the two pins from Session 3 (`mkl=2024.0.0`,
`transformers<4.47`) to the built env.

**Found:** Three failures in sequence, each with a distinct cause.

1. **`PackagesNotFoundInChannelsError: cuda-toolkit11.8.*.*`.** `conda env create`
   built the env from the recipe's five channels, but a subsequent `conda install`
   only searches whatever is in `.condarc` — which is `defaults` alone. The solver
   then could not re-satisfy the already-installed `cuda-toolkit 11.8` because that
   package lives in `nvidia/label/cuda-11.8.0`.
2. **Solver hang.** Re-running with explicit channels made the solve run for 45+
   minutes without terminating, on both the classic solver and (nominally) libmamba.
3. **Root cause, found via `conda list | grep -i mkl`:** the env has `mkl 2025.0.0`,
   and also `blas 1.0 mkl`, `mkl-service`, `mkl_fft`, `mkl_random` — i.e. **numpy is
   built against MKL**. Downgrading `mkl` to 2024.0.0 therefore requires simultaneously
   re-solving numpy, blas and three MKL bindings against a pinned CUDA toolkit. That is
   not a slow solve; it is a combinatorial problem that does not finish.

**Concluded:** The fix cited in every bug report for `undefined symbol: iJIT_NotifyEvent`
— downgrade MKL — is **not applicable to this environment**. Sidestepped it instead:
replaced conda's PyTorch with a pip cu118 wheel, which bundles its own math libraries and
does not link conda's MKL at all. MKL 2025 stays in place for numpy; torch stops caring.

    pip install --force-reinstall --no-deps torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu118

`--no-deps` then caused a second, smaller failure — `libcudnn.so.9: cannot open shared
object file` — because torch 2.5.1 needs cuDNN 9 and the CUDA runtime packages had been
skipped. Re-running the same command *without* `--no-deps` installed only the missing
`nvidia-*` wheels (pip saw `2.5.1+cu118` as already satisfying `==2.5.1`, so no
re-download).

**Stage 0 gate passed:** `torch 2.5.1+cu118`, `transformers 4.46.3` (inside the
`<4.47` pin), `peft 0.20.0`, `deepspeed` imports cleanly.

**Process lesson, and the expensive one:** several days were lost debugging a package
install *through the batch queue* — submit, wait hours, read one error line, repeat.
Environment work belongs in an interactive shell where the feedback loop is seconds.
Only jobs that genuinely need a GPU should be queued.

---

## Session 7 — LoRA plumbing verified at runtime

**Date:** 2026-08-27

**Did:** Ran `scripts/verify_lora.py` on the `batch` partition against
`Qwen/Qwen2.5-0.5B-Instruct`.

**Found: 17/18 checks passed.** The substantive results:

| Check | Result |
|---|---|
| Actor wraps as `PeftModelForCausalLM` | pass |
| Trainable fraction | **540,672 / 494,330,624 = 0.109%** |
| Adapters injected into | **`['q_proj', 'v_proj']`** |
| `generate()` through the wrapper | pass |
| `disable_adapter()` restores base output | pass |
| **`score_head` is trainable** | **pass — 2 of 4 tensors** |
| Critic forward returns scores | pass, shape (1, 4, 1) |
| No-LoRA control untouched | pass, 100% trainable |

Three numbers corroborate each other and are worth recording:

- **48 `lora_B` tensors** = 24 layers × 2 target modules. Qwen2.5-0.5B has 24 layers.
- **Critic trainable − actor trainable = 541,569 − 540,672 = 897** = `score_head`
  weight (896, the hidden size) + bias (1). The value head is trainable down to the
  parameter, confirming `modules_to_save=['score_head']` resolved correctly. This was
  the highest-risk line in the whole change and it is now verified, not assumed.
- `['q_proj', 'v_proj']` confirms PEFT v0.20.0's Qwen2 default is the original LoRA
  paper's narrow choice — query and value only, no `k_proj`, no `o_proj`, no MLP.

**The one failure was a bug in the test, not the code.** The check
`adapters are re-enabled after the proxy call` used
`getattr(module, 'disable_adapters', False)` across all modules, which picks up bound
methods and properties on PEFT wrappers — truthy regardless of actual state. The
contradiction is visible in the output: `proxy output differs from the adapter-enabled
actor` passed, and that comparison uses a forward pass taken *after* the proxy call, so
adapters must have been re-enabled. Replaced the introspection with a behavioural
comparison of logits. **Lesson: do not assert on a library's internal attribute names;
assert on observable behaviour.**

**Also found:**

- `batch` partition nodes have GPUs (`cuda: True 1`) — undocumented in the MSS guide.
- HuggingFace downloads on a `batch` node ran at ~0.3 MB/s (514 MB in 31 minutes) versus
  7–40 MB/s on the login node. Suspected cause is the `hf-xet` chunked-transfer backend.
  Workaround: pre-download on the login node, or `export HF_HUB_DISABLE_XET=1`. Relevant
  for Stage 4, where the reward model is ~14 GB.
- A benign warning: Qwen's tokenizer vocab (151,665) is smaller than its embedding matrix
  (151,936). That is normal padding for Qwen, not a misconfiguration.

**Concluded:** **Stage 1 is functionally verified**, except for adapter save/resume,
which has no runtime coverage yet. The remaining unknowns are DeepSpeed-specific — whether
a `PeftModel` survives ZeRO wrapping, and whether `AdapterDisabledReference` behaves under
parameter partitioning — and neither can be tested without a real distributed launch.

---

## Session 8 — Stage 2 closed: template decision made and verified

**Date:** 2026-08-27

**Did:** Settled the two decisions left open at the end of Stage 1, then verified the
one that could be verified.

**Decision 1 — interval checkpoints → adapter snapshots.** `rl_trainer.py` now writes
a LoRA adapter into `output_dir/checkpoint-{step}/` at each `--save_interval` when the
actor is PEFT-wrapped, instead of a DeepSpeed engine checkpoint. Rationale is specific
to this project: Phase 1's most valuable findings were trajectory-shaped (entropy < 1.0
by step 95, KL negative across the final 100 steps), but only the *final* policy was ever
loadable, so "what was the policy actually saying at step 250 vs 750?" could not be
answered. At ~2 MB per adapter, every snapshot of a full run can be kept and evaluated.
The cost is optimizer state, which upstream cannot restore anyway (Session 5: there is no
`load_checkpoint` call anywhere in the framework).

**Decision 2 — prompt template → keep upstream's Alpaca-style, do not switch to ChatML.**
This reverses the recommendation in the original plan document. The deciding fact is that
`post_rollout()` re-tokenizes for the reward model with `skip_special_tokens=True`.
ChatML's structure lives entirely in special tokens (`<|im_start|>`, `<|im_end|>`), so it
would be stripped before Beaver-7B saw anything, leaving it to score bare `system` /
`user` / `assistant` words with no delimiters — a format it was never trained on. The
Alpaca template is plain text and survives intact, and is the format PKU trained the
reward model on. Given that Phase 1's central finding was that a corrupted reward signal
invalidates everything downstream, reward-path fidelity outranks actor-path fidelity here.

**Found (`scripts/verify_dataset.py`, 10/10 passed):**

- `PKU-SafeRLHF/train` loads unchanged: **38,641 prompts**.
- Prompts render exactly as intended, e.g.
  `'BEGINNING OF CONVERSATION: USER: How can I increase my online presence...? ASSISTANT:'`
- Exactly one USER turn and one ASSISTANT turn per prompt; prompt terminates at the
  assistant marker; no doubled BOS.
- **The decisive check:** after `skip_special_tokens=True`, the decoded string is
  byte-identical to the original including both markers. This is the empirical form of the
  argument above — the reward model really will see a well-formed prompt.

**Also found — a silent model modification worth knowing about.** Qwen ships with no
`bos_token`, and `resize_tokenizer_embedding()` injects `DEFAULT_BOS_TOKEN = '<s>'`
(a LLaMA convention). Two effects: (a) a `<s>` token is added to Qwen's vocabulary, though
it appears 0 times in actual sequences, so it is registered but never emitted; (b) because
a token was added, `resize_token_embeddings()` fires and **shrinks** the embedding matrix
from 151,936 to 151,666. Those 270 rows are unused alignment padding (151,936 = 128 × 1187,
sized for tensor cores), so nothing breaks, and under LoRA the embedding is frozen
regardless. Recorded so it is not mistaken for a bug later.

**Concluded: Stage 2 is complete.** Template locked, dataset path verified, no code change
required for the template itself. **Conditional to revisit:** if Stage 4's probe shows the
Beaver reward model behaving badly and we fall back to training our own Qwen RM, this
decision flips — a self-trained RM would be trained on whatever format we choose, making
ChatML correct and letting the actor be on-distribution too.

---

## Session 9 — Stage 3 attempt 1: the cluster cannot JIT-compile CUDA extensions

**Date:** 2026-08-27

**Did:** First real `deepspeed --module safe_rlhf.algorithms.ppo` launch
(`scripts/smoke-ppo-qwen-lora.sbatch`). Deliberately tiny: Qwen2.5-0.5B actor and
critic, `gpt2` as a stand-in reward model, `max_length 128`, batch size 2,
~19 steps via a dataset proportion, `save_interval 5`.

**Note on the reward-model choice.** The first draft of this smoke test used a Qwen
stand-in reward model. That was wrong, and Wendy caught it: `rl_trainer.py` collapses
`reward_tokenizer` onto `tokenizer` when `is_same_tokenizer()` is true, and
`post_rollout()` only calls `batch_retokenize()` when they differ. A Qwen reward model
would therefore have **skipped the re-tokenization branch entirely** — testing a code
path we will never use. `gpt2` was chosen precisely because its tokenizer differs from
Qwen's, forcing the same bridge the real Beaver-7B reward model will take, at 124M
parameters instead of 7B.

**Found:** The run failed during DeepSpeed's JIT compilation of `FusedAdam`:

    error: #error -- unsupported GNU version! gcc versions later than 11 are not supported!

`gcc --version` on the nodes reports **15.2.0**, and `/usr/bin/` has only `gcc-15`.
CUDA 11.8's `nvcc` supports gcc ≤ 11. There is no older system compiler available.

**This is not specific to `FusedAdam`.** It is a property of the environment: with
CUDA 11.8 and gcc 15, **no CUDA extension can be JIT-compiled on this cluster**.
Anything DeepSpeed tries to build at runtime will fail the same way. `DeepSpeedCPUAdam`
is not an escape — it compiles too.

Worth noting what *did* work before the failure: `load_pretrained_models()` with LoRA
completed under a real distributed launch, and DeepSpeed accepted a `PeftModel` through
model initialisation. Also, build step `[2/3]` — plain `c++` compiling the frontend —
**succeeded**. gcc 15 handles the C++ fine; only `nvcc` refuses. So avoiding CUDA kernels
is sufficient, and avoiding C++ entirely is not necessary.

**Concluded:** Added `--use_torch_adam` (default `False`, so upstream behaviour is
unchanged) which selects `torch.optim.AdamW` instead of either DeepSpeed Adam. It takes
the same parameter groups and the same `ADAM_BETAS`, and `deepspeed.initialize()` accepts
any `torch.optim.Optimizer`. The only cost is kernel fusion speed, which is irrelevant
against not running at all. Checked *before* the offload branch so it short-circuits both
compiled paths.

**Options considered and rejected:**

- `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler` — a four-major-version gap between
  gcc 11 and gcc 15 means the compile would very likely fail anyway on libstdc++ changes.
- `export CC=/usr/bin/gcc-11` — no gcc 11 exists on the nodes.
- `conda install -c conda-forge gxx_linux-64=11` — would probably work, but Session 6
  showed conda solves in this environment are unreliable and can hang indefinitely.

**Carry forward — this will resurface.** If a future stage needs a DeepSpeed op with no
pure-torch fallback (some ZeRO-3 paths, fused kernels, sparse attention), `--use_torch_adam`
will not save us and the real fix becomes installing gcc 11 into the conda env. Better to
know that now than to discover it while queuing for biggpu at Stage 5.

---

## Open tasks

**Blocking the first real run:**

- [x] Confirm the environment imports cleanly. **Done (Session 6):** torch 2.5.1+cu118,
      transformers 4.46.3, peft 0.20.0, deepspeed. Required replacing conda's torch with a
      pip cu118 wheel rather than downgrading MKL.
- [x] Run `scripts/verify_lora.py` on the cluster. **Done (Session 7): 17/18 passed.**
      The `score_head` assertion — predicted as most likely to fail — passed. The single
      failure was a bug in the test's use of PEFT internals, since fixed.
- [x] **Adapter saving implemented** (Sessions 5 and 8): `TrainerBase.save()` writes an
      adapter for the final model, and `save_interval` writes per-step adapter snapshots.
      Both gather ZeRO-3 shards for the trainable subset before writing. **Still has no
      runtime coverage** — needs the Stage 3 smoke run to exercise it.
- [ ] **Resume is still not implemented** and does not exist upstream either. A run killed
      by load shedding restarts from zero. Adapter snapshots make this survivable
      (restart from the latest adapter, losing optimizer state) but there is no code for it.
- [x] Prompt template decided and verified (Session 8): **keep upstream Alpaca-style**.
      Reverses the plan document's recommendation; see Session 8 for the reasoning.

**Before trusting any result:**

- [ ] Verify `batch_retokenize` round-trips Qwen text through the LLaMA tokenizer without
      meaningful drift — it decodes with `skip_special_tokens=True` and re-encodes.
      If scores look wrong at smoke-test time, the fallback is training a small Qwen RM.
- [ ] Sanity-check the Beaver reward model separates obviously-harmful from
      obviously-helpful text *before* spending biggpu hours on it.
- [ ] Assert the critic's `score_head` has `requires_grad=True` after wrapping, as an
      in-run check and not just a one-off test.

**Environment constraints to design around (Session 9):**

- [ ] **No CUDA extension can be JIT-compiled on mscluster** (CUDA 11.8 + gcc 15, no older
      host compiler present). Currently sidestepped for the optimizer via
      `--use_torch_adam`. If a later stage needs a DeepSpeed op with no pure-torch
      fallback, the fix is `conda install -c conda-forge gxx_linux-64=11` — schedule that
      deliberately rather than discovering it mid-run.
- [ ] `/datasets/wmaboa` is confirmed writable from compute nodes; the HuggingFace cache
      is now symlinked there (`~/.cache/huggingface -> /datasets/wmaboa/hf`) to keep the
      14 GB reward model out of the 50 GB home quota.
- [ ] HuggingFace downloads on compute nodes run ~0.3 MB/s versus 7–40 MB/s on the login
      node. Pre-download on the login node, and set `HF_HUB_DISABLE_XET=1`.
- [ ] `batch` partition nodes have an **RTX 3060 (12 GB)** and there are usually ~88 idle.
      That is the development resource; `bigbatch` and `biggpu` are routinely fully
      allocated. 12 GB fits a 0.5B or 1.5B actor+critic, but not the 7B reward model.

**Deferred by decision, not oversight:**

- [ ] Widening `--lora_target_modules` beyond PEFT's `["q_proj", "v_proj"]` default —
      revisit with measurements at smoke-test stage.
- [ ] A separate flag for a full-weight critic with a LoRA actor. Currently `--use_lora`
      turns both on together; a small critic with a full-precision value head is a
      legitimate configuration if the critic underfits.
- [ ] PPO-Lag, cost models, reward shaping, GRPO, and the Minmax penalty. All explicitly
      out of scope until this reference build produces a trustworthy number.
