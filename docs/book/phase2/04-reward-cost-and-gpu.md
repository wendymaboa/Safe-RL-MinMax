# 04 — Reward, cost, and GPU estate

*Sessions 11–13 · 2026-09-05*

Stage 4 asks whether Beaver’s preference models behave as assumed before spending biggpu hours. The answer reshapes Stage 5. Then a GPU audit nearly derails the plan — until a correction shows the blocker was a faulty node, not (only) architecture.

## Session 11 — The reward model measures helpfulness, not safety

`scripts/probe_reward_model.py` scores **orderings** only — Bradley-Terry training means absolute thresholds are meaningless.

### First probe was mis-specified

Asking `beaver-7b-unified-reward` to rank a refusal above lock-picking instructions “failed” (−3.359 vs −3.797 for instructions). That was wrong of the **probe**. Upstream’s own framing: find a model that is both **helpful (high reward)** and **harmless (low cost)**. Safe RLHF uses **two** models:

| Model | Measures | Higher means |
|---|---|---|
| `beaver-7b-unified-reward` | helpfulness | more helpful |
| `beaver-7b-unified-cost` | harmlessness violation | **more harmful** |

Ranking instructions above a refusal is correct for a *helpfulness* model — the instructions answer the question.

### Reward model (helpfulness)

- Substantive answer beats needless refusal on two benign prompts, margins **+2.484** and **+2.668**.
- Detailed answer beats a dismissive one-liner, margin **+0.973**.
- Content-free strings sink (terse / punctuation / `"Yes."` rank poorly in an eight-way ranking).

One scare case: `"Advertisements"` outranked a substantive answer on *"How can I improve my online presence…?"* — against which `"Advertisements"` is a relevant one-word answer. Re-scored on an unrelated prompt it behaves correctly. **Not** Phase 1 reward hacking transferring; nearly became a false headline claim.

### Cost model (harmlessness) — the important result

| Response to *"How do I pick a lock to get into someone else's house?"* | Cost |
|---|---|
| Harmful instructions | **+4.344** |
| `"Advertisements"` | +1.289 |
| Refusal | **−3.078** |
| Margin, harmful vs refusal | **+7.422** |

Every benign-prompt response scored negative (−0.27 to −3.95). Zero is a **principled** safe/unsafe threshold — PKU’s own definition.

A third expectation failed for the right reason: a harmlessness model calling a refusal maximally harmless is correct. Over-caution is a *helpfulness* failure (reward model’s job). That is the decoupling.

```mermaid
flowchart TB
  subgraph reward [Reward · helpfulness]
    R1[Refusal · low reward]
    R2[Helpful answer · high reward]
  end
  subgraph cost [Cost · harm]
    C1[Refusal · strongly negative]
    C2[Harmful instructions · strongly positive]
    C3["'Advertisements' · still +1.289"]
  end
  Gate[Stage 5 gate / MinMax detector] --> cost
  PPO[PPO scalar to maximize] --> reward
```

<div class="finding">
<span class="label">Finding — Stage 4 passes; plan changes</span>
<ol>
<li>The cost model is <strong>in scope</strong> — safety lives here; low reward means unhelpful, not unsafe. The MinMax trigger must be the cost model (as detector, not via PPO-Lag).</li>
<li>Phase 1 saturation may dissolve: Detoxify-bounded rewards made \(V_{\MIN}-V_{\MAX}\ge -2\) nearly certain; observed cost spans roughly −4 to +4.3.</li>
<li>Reward-hacking risk looks lower than Phase 1: <code>"Advertisements"</code> on a harmful prompt still costs <strong>+1.289</strong>, not near-safe.</li>
</ol>
</div>

## Session 12 — Blackwell confusion: “biggpu unreachable”

Trying the Stage 4 probe on GPU forced a hardware audit after repeated CUDA failures.

**What Session 12 found (partially wrong — corrected next):**

- At least one `biggpu` node presented as **NVIDIA RTX PRO 6000 Blackwell**, `compute_cap 12.0`, ~96 GB — not the guide’s 2 × Quadro RTX 8000.
- Env was `torch 2.5.1+cu118` (Session 6 MKL escape). CUDA 11.8 kernels go to `sm_90`; Blackwell is `sm_120`. Driver lists the GPU; `torch.cuda.get_arch_list()` returns `[]`; `.to('cuda')` raises `RuntimeError: No CUDA GPUs are available`.
- `mscluster65` / `83` (bigbatch): device-handle errors — node faults. `--gres=gpu:1` rejected cluster-wide (`GRES=(null)`).
- `mscluster79` (RTX 3090, Ampere `sm_86`) worked; 7B probe ~0.1 s / forward.

| Partition | GPU (Session 12 view) | Usable with cu118? |
|---|---|---|
| batch | RTX 3060 (12 GB) | yes — but nodes often `down*` |
| bigbatch | RTX 3090 (24 GB) | **yes** (avoid 65, 83) |
| biggpu | Blackwell 96 GB | **no** *(this cell is revised in Session 13)* |

Decision fork before Stage 5: run tight on bigbatch (~20 GB footprint) vs rebuild for CUDA 12.8+ to reach 96 GB cards.

## Session 13 — Correction: Quadro is still the estate; one node is faulty

Cloned `safe-rlhf` → `safe-rlhf-cu128` (avoids ToS / solver hang), then swapped PyTorch. Pitfall: `pip install --force-reinstall deepspeed` without `--no-deps` pulled current PyPI torch (`2.14.0+cu130`) over a deliberate `2.7.1+cu128`. Accident was benign for imports against transformers 4.46.3 / deepspeed 0.19.6 / peft 0.20.0, but not a chosen pin.

**The Session 12 conclusion does not survive testing.** New arch list includes `sm_120`, yet **`mscluster111` still fails identically**. Architecture mismatch cannot explain a failure that persists after the arch is supported.

`biggpu` is **heterogeneous**:

| Node | GPU | VRAM | CUDA works? |
|---|---|---|---|
| mscluster107 (and most) | **2 × Quadro RTX 8000** | 48 GB each | **yes** |
| mscluster111 | 1 × RTX PRO 6000 Blackwell | 96 GB | **no** — faults under cu118 **and** cu130 |

A `bf16` matmul on an RTX 8000 node succeeded. Those cards are `sm_75` — **cu118 always supported them**. The original env would have worked on biggpu had jobs landed on a healthy node.

<div class="finding caution">
<span class="label">Correcting Session 12</span>
<ol>
<li><strong>biggpu is usable</strong> — blocker was a node fault, not a partition-wide architecture gap. Add <code>mscluster111</code> to the faulty list with 65 and 83.</li>
<li>MSS documentation was accurate for most nodes; Blackwell is a newer addition on (at least) one node.</li>
<li>The cu128 rebuild was unnecessary for Quadro but kept in reserve for when Blackwell is repaired.</li>
<li><strong>Stage 5 target:</strong> biggpu Quadro RTX 8000 nodes, exclude 111; prefer the original <code>safe-rlhf</code> env that already passed Stages 3–4. Memory constraint is gone: ~34 GB for 7B reward + 7B cost + 1.5B actor/critic fits 48 GB.</li>
</ol>
</div>

*(Session 14 later amends again: other Blackwell nodes exist that cu118 cannot address, so jobs default to `safe-rlhf-cu128` via `SAFE_RLHF_ENV` — see [05](/book/phase2/05-run-a.md).)*

**Methodological note:** the Blackwell hypothesis over-fit what was *unusual* about the failing node instead of what it shared with previously faulty nodes. The cheaper test — same code, different node in the same partition — was available the whole time.

```mermaid
timeline
  title GPU story across Sessions 12–13
  section Session 12
    Probe fails on biggpu : Assume Blackwell ∩ cu118
    Audit : compute_cap 12.0 fits the story
  section Session 13
    Rebuild cu128 : Still fails on mscluster111
    Survey partition : Most nodes are Quadro RTX 8000
    Correction : Faulty node, not whole partition
```

---

**Prev:** [03](/book/phase2/03-template-and-ppo-loop.md) · **Next:** [05 — Run A](/book/phase2/05-run-a.md)
