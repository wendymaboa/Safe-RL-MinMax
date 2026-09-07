# 03 — Template and PPO loop

Source sessions: 8–10 · Full text: [/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md)

*Sessions 8–10 · 2026-08-27*

Stage 2 locks the data path and (temporarily) the prompt template. Stage 3 gets a real DeepSpeed PPO launch through LoRA end to end — after fighting the cluster’s inability to JIT CUDA extensions. Canonical trail: [/worklog/worklogs/phase2.md](/worklog/worklogs/phase2.md).

## Session 8 — Stage 2 closed: template decision made and verified

Two leftover Stage 1 decisions were settled, then the one that could be verified was checked with `scripts/verify_dataset.py`.

### Decision 1 — Interval checkpoints → adapter snapshots

When the actor is PEFT-wrapped, `rl_trainer.py` writes a LoRA adapter into `output_dir/checkpoint-{step}/` at each `--save_interval` instead of a DeepSpeed engine checkpoint.

**Why:** Phase 1’s most valuable findings were trajectory-shaped (entropy $< 1.0$ by step 95, KL negative across the final 100 steps), but only the *final* policy was ever loadable — so “what was the policy saying at step 250 vs 750?” could not be answered. At ~2 MB per adapter (before the embedding-save bug of Session 10), every snapshot of a full run can be kept and evaluated. Upstream cannot restore optimizer state anyway — there is no `load_checkpoint` call in the framework.

### Decision 2 — Prompt template → keep Alpaca-style *(Session 8 decision)*

This reversed the original plan’s ChatML recommendation. The deciding fact: `post_rollout()` re-tokenizes for the RM with `skip_special_tokens=True`. ChatML’s structure lives in special tokens (`<|im_start|>`, `<|im_end|>`), which would be stripped before Beaver-7B sees anything — leaving bare `system` / `user` / `assistant` words, a format the RM was never trained on. The Alpaca template is plain text, survives intact, and matches how PKU trained the RM. Given Phase 1’s lesson that a corrupted reward signal invalidates everything downstream, **reward-path fidelity outranked actor-path fidelity** here.

<div class="finding caution">
<span class="label">Later reversal — Session 14</span>
This decision is reversed on evidence in <a href="/worklog/phase2/05-run-a.md">Run A</a>: under Alpaca, Qwen never emits its eos token, so every generation runs to the length cap and decays into noise. ChatML becomes mandatory for Stage 5. The Session 8 argument about what the RM <em>receives</em> was incomplete about what the actor can <em>produce</em>.
</div>

**`scripts/verify_dataset.py` — 10/10 passed:**

| Check | Result |
|---|---|
| `PKU-SafeRLHF/train` loads | **38,641 prompts** |
| Prompt render | `BEGINNING OF CONVERSATION: USER: … ASSISTANT:` |
| Turn structure | Exactly one USER and one ASSISTANT; terminates at assistant marker; no doubled BOS |
| After `skip_special_tokens=True` | Decoded string **byte-identical** to the original including both markers |

The last row is the empirical form of the Session 8 argument — the reward model really will see a well-formed Alpaca prompt.

**Silent model modification worth recording.** Qwen ships with no `bos_token`; `resize_tokenizer_embedding()` injects `DEFAULT_BOS_TOKEN = '<s>'` (a LLaMA convention). Effects:

1. `<s>` is registered but never emitted (appears 0 times in actual sequences).
2. Because a token was added, `resize_token_embeddings()` fires and **shrinks** the embedding matrix from **151,936 → 151,666**. Those 270 rows are unused alignment padding ($151{,}936 = 128 \times 1187$, sized for tensor cores), so nothing breaks; under LoRA the embedding is frozen regardless.

Recorded so it is not mistaken for a bug — and so Session 10’s 520 MB snapshot failure has a clear cause.

**Stage 2 complete** at the time — with a conditional: if Stage 4 forced a home-trained Qwen RM, ChatML would flip back to correct. Stage 4 kept Beaver; Session 14 flipped ChatML for the *actor* anyway.

## Session 9 — Stage 3 attempt 1: no CUDA JIT on this cluster

First real launch: `scripts/smoke-ppo-qwen-lora.sbatch` — Qwen2.5-0.5B actor/critic, **`gpt2` as stand-in RM**, `max_length 128`, batch 2, ~19 steps via a dataset proportion, `save_interval 5`.

**Why gpt2, not a Qwen stand-in RM:** Wendy caught the mistake — if reward and actor share a tokenizer, `rl_trainer.py` collapses `reward_tokenizer` onto `tokenizer` and **skips** `batch_retokenize()`. A Qwen RM would have tested a code path Stage 5 never uses. `gpt2` forces the same bridge Beaver-7B will take, at 124M parameters instead of 7B.

Failure during DeepSpeed JIT of `FusedAdam`:

```text
error: #error -- unsupported GNU version! gcc versions later than 11 are not supported!
```

Nodes report **gcc 15.2.0**; `/usr/bin/` has only `gcc-15`. CUDA 11.8’s `nvcc` supports gcc ≤ 11. **No CUDA extension can be JIT-compiled on this cluster** — not just `FusedAdam`; `DeepSpeedCPUAdam` compiles too.

What *did* work before the failure:

- LoRA `load_pretrained_models()` under a real distributed launch.
- DeepSpeed accepted a `PeftModel` through model initialisation.
- Build step `[2/3]` (plain C++ frontend) succeeded — gcc 15 is fine; only `nvcc` refuses.

So avoiding CUDA kernels is sufficient; avoiding C++ entirely is not necessary.

**Fix:** `--use_torch_adam` (default `False`, upstream unchanged) selects `torch.optim.AdamW` instead of either DeepSpeed Adam. Same param groups / `ADAM_BETAS`; `deepspeed.initialize()` accepts any `torch.optim.Optimizer`. Checked *before* the offload branch so it short-circuits both compiled paths. Cost is fusion speed, irrelevant against not running at all.

| Option | Verdict |
|---|---|
| `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler` | Rejected — four-major-version gap (gcc 11 vs 15); libstdc++ likely breaks anyway |
| `export CC=/usr/bin/gcc-11` | Rejected — no gcc 11 on the nodes |
| `conda install … gxx_linux-64=11` | Deferred — Session 6 showed conda solves here hang indefinitely |
| `--use_torch_adam` | **Chosen** |

```mermaid
flowchart TD
  Launch[deepspeed PPO smoke] --> JIT[DeepSpeed JIT FusedAdam]
  JIT --> Fail[nvcc refuses gcc 15]
  Fail --> Opt{Options}
  Opt -->|chosen| Torch[--use_torch_adam · AdamW]
  Opt -->|rejected| Allow[allow-unsupported-compiler]
  Opt -->|rejected| Gcc11[system gcc-11 missing]
  Opt -->|deferred| CondaGcc[conda gxx=11]
  Torch --> Cont[Continue Stage 3]
```

**Carry forward:** if a later stage needs a DeepSpeed op with no pure-torch fallback (some ZeRO-3 paths, fused kernels, sparse attention), `--use_torch_adam` will not save you — install gcc 11 deliberately, not mid-biggpu queue.

## Session 10 — Stage 3 closed: PPO loop end to end with LoRA

Smoke completed: **exit 0:0, 37 PPO steps, ~3 minutes.** Four failures on the way, each one layer deeper:

| # | Failure | Fix |
|---|---|---|
| 1 | nvcc vs gcc 15 | `--use_torch_adam` (Session 9) |
| 2 | 30-minute silent hang in `poll_schedule_timeout` (1.9% CPU, CUDA context, 0% GPU) | `gpt2` not prefetched while Beaver download saturated the link — diagnose via `ps -o stat,wchan`; `py-spy` blocked by ptrace |
| 3 | `HF_HUB_OFFLINE=1` breaks `datasets` Hub module resolution (also via `TRANSFORMERS_OFFLINE` alias) | Bounded timeouts instead: `HF_HUB_ETAG_TIMEOUT=15`, `HF_HUB_DOWNLOAD_TIMEOUT=30` |
| 4 | Adapter snapshots **520 MB**, not ~1 MB | PEFT `save_embedding_layers='auto'` saw vocab resize (Session 8) as “trained embeddings” and saved 151,666 × 896 fp32 ≈ 518 MB — force `False`; snapshots drop to **1.1 MB** (= 540,672 × 2 bytes bf16) |

**Evidence the LoRA design is correct, from a real run:**

| Metric | Observed | Proves |
|---|---|---|
| `train/kl_divergence` | **first = 0.0000**, then ±0.5 | `AdapterDisabledReference` works |
| `train/reward_value` | −0.02 → **+2.94** | `modules_to_save=['score_head']` works |
| `train/reward` | 2.4609 → 2.5703 | Qwen→GPT-2 bridge delivers real text |
| snapshot size | **1.1 MB** | only adapter weights saved |

KL at step 0 is the strongest write-up point: LoRA `B` matrices initialise to **zero**, so the actor *is* the base model and KL vs reference is exactly 0. Had the proxy returned adapter-enabled output, KL would stay pinned at zero forever — silent failure with no error and no warning. The observed trajectory rules it out.

`reward_value` climbing toward true reward (~2.5) is the critic head learning; frozen, it would have stayed at its random value (Session 5’s silent-failure mode — the highest-risk line in the LoRA change).

**Two non-issues (do not re-investigate):**

- `actor_lr` / `reward_critic_lr` printing `+0.0000` was `%f` formatting of `1e-5` in `dump_tb.py` (switched to `%g`).
- Mildly negative KL (min −0.63) is Monte-Carlo noise on a 2-sequence batch — not Phase 1’s −41 numerical breakdown.

**Unresolved oddity:** `mean_generated_length` equals `max_generated_length` every step (plausible left-padding artefact on batch size 2; recheck at larger batches before trusting either number).

Also added `scripts/dump_tb.py` — TensorBoard metrics never reach the SLURM `.out` file; this is how trajectories get inspected over SSH.

**Stage 3 passes:** DeepSpeed accepts `PeftModel`, reference proxy works under a live engine, re-tokenization bridge fires, snapshots are the right size. Iteration time ~3 minutes per attempt — contrast Session 6’s 45-minute conda hangs. Fast failure is what made this stage tractable.

```mermaid
flowchart LR
  S8[S8 · Template + dataset] --> S9[S9 · CUDA JIT wall]
  S9 --> Adam[--use_torch_adam]
  Adam --> S10[S10 · Four deeper failures]
  S10 --> OK[37 steps · exit 0]
  OK --> Evidence[KL=0 then grows · score_head learns · 1.1 MB adapters]
```

---

**Prev:** [02](/worklog/phase2/02-lora-and-mkl.md) · **Next:** [04 — Reward, cost, and GPU](/worklog/phase2/04-reward-cost-and-gpu.md)
