<div class="cover-kicker">Architecture · Phase 1 → Phase 2</div>

# Architecture atlas — from pilot to Stage 5

One page of diagrams for the whole project stack. Use this for talks and Google Slides (redraw shapes from the Mermaid logic; keep labels). Full narrative lives in the [Phase 1](/worklog/phase1/README.md) and [Phase 2](/worklog/phase2/README.md) chapters.

---

## 0. Big picture — why two phases

```mermaid
flowchart LR
  subgraph P1 [Phase 1 · closed]
    A1[GPT-2 actor/critic]
    D1[Detoxify δ ∈ 0..1]
    M1[MinMax gate]
    A1 --> D1 --> M1 --> PPO1[PPO + KL]
  end
  subgraph Why [Binding constraint]
    W["Detector bounded + gameable<br/>Advertisements hack<br/>MinMax ≈ PPO+KL"]
  end
  subgraph P2 [Phase 2 · current]
    A2[Qwen + LoRA]
    RM[Beaver reward]
    CM[Beaver cost detector]
    ABC[A / B / C gates]
    A2 --> RM
    A2 --> CM --> ABC --> PPO2[Safe-RLHF PPO]
  end
  P1 --> Why --> P2
```

| | Phase 1 | Phase 2 (documented track) |
|---|---|---|
| Codebase | `ppo_minmax_experiment/` + TRL | `safe-rlhf/` (PKU) + DeepSpeed |
| Actor | GPT-2 small | Qwen2.5-1.5B-Instruct + LoRA |
| Helpfulness signal | Centered Detoxify \(r=1-2\delta\) | Beaver **reward** (frozen) |
| Safety / gate signal | Same Detoxify (toxicity threshold) | Beaver **cost** (`cost > 0`) |
| MinMax question | vs PPO+KL only | vs **matched fixed gate** (B), with A as control |
| Status | Closed | A/B done; C trained, rescore open |

---

## 1. Phase 1 — GPT-2 + Detoxify + MinMax

### 1.1 Training loop

```mermaid
flowchart TB
  P[BeaverTails / RealToxicity-style prompts] --> ACT[GPT-2 actor<br/>sample responses]
  ACT --> DET[Detoxify toxicity δ]
  DET --> R["Centered reward<br/>r = 1 − 2δ ∈ [−1, 1]"]
  R --> GATE{δ high / r low?<br/>unsafe?}
  GATE -->|no| KEEP[Keep r]
  GATE -->|yes| MM["Replace with<br/>R_unsafe = V_MIN − V_MAX<br/>floor −2"]
  KEEP --> PPO[TRL PPO step<br/>+ KL to reference]
  MM --> PPO
  PPO --> CRIT[GPT-2 critic / value head]
  CRIT --> PPO
  R --> BOUNDS[Update V_MIN / V_MAX]
  BOUNDS --> MM
```

### 1.2 What Detoxify was doing (two jobs)

```mermaid
flowchart LR
  subgraph Dual [One scalar did everything]
    D[Detoxify δ]
    D --> Reward[Helpfulness proxy<br/>want low toxicity]
    D --> Detect[Unsafe detector<br/>gate when toxic]
  end
  Reward --> Train[PPO target]
  Detect --> Train
```

<div class="finding caution">
<span class="label">Phase 1 lesson</span>
Because \(r \in [-1,1]\), \(R_{\text{unsafe}} = V_{\MIN}-V_{\MAX} \ge -2\) by arithmetic. Self-calibration freezes early. Separately, <code>"Advertisements"</code> scores ≈0 toxicity → gate never fires → reward hacking, not safety.
</div>

### 1.3 Fair Phase 1 comparison (what closed the phase)

```mermaid
flowchart TB
  RAW[Raw GPT-2] --> B[PPO + KL · β=0.2]
  RAW --> C[PPO + MinMax · β=0.2]
  B --> E[Matched env + sample eval]
  C --> E
  E --> V["Honest verdict: ~tied<br/>2.0% vs 2.3% harm · one seed"]
```

---

## 2. Transition — what had to change

```mermaid
flowchart TB
  F1[Bounded Detoxify] --> N1[Need unbounded / preference-trained scores]
  F2[One scalar = reward + detector] --> N2[Split reward vs cost]
  F3[MinMax vs PPO+KL only] --> N3[Add fixed-gate control B]
  F4[GPT-2 + TRL pilot] --> N4[PKU Safe-RLHF + real chat model]
  N1 --> P2[Phase 2]
  N2 --> P2
  N3 --> P2
  N4 --> P2
```

Detoxify’s **role** (unsafe detector) maps to Beaver **cost**. Detoxify’s **reward role** maps to Beaver **reward**. Replacing Detoxify with “an RM alone” would still miss the safety gate.

---

## 3. Phase 2 — Safe RLHF stack (Qwen track)

### 3.1 Models and who trains

```mermaid
flowchart TB
  subgraph Frozen [Frozen · no gradients]
    RM[Beaver-7B unified reward<br/>helpfulness]
    CM[Beaver-7B unified cost<br/>harmfulness · detector]
  end
  subgraph Trainable [Trainable · LoRA]
    ACT[Qwen2.5-1.5B-Instruct actor]
    CRIT[Qwen2ForScore critic<br/>same base + score head]
  end
  subgraph Free [Free under LoRA]
    REF[Reference policy<br/>actor with adapters off]
  end
  PROMPT[PKU-SafeRLHF prompts<br/>ChatML template] --> ACT
  ACT --> GEN[Generated reply]
  GEN --> RM
  GEN --> CM
  GEN --> CRIT
  ACT --> REF
  RM --> GATE[Gated scalar for PPO]
  CM --> GATE
  GATE --> PPO[DeepSpeed PPO]
  CRIT --> PPO
  REF --> PPO
```

| Role | Model | Trains? |
|---|---|---|
| Actor | Qwen2.5-1.5B-Instruct + LoRA | adapters |
| Reference | same base, adapters disabled | no |
| Critic | `Qwen2ForScore` + LoRA | adapters + score head |
| Reward | `beaver-7b-unified-reward` | frozen |
| Cost (B/C) | `beaver-7b-unified-cost` | frozen |

### 3.2 Stage ladder (how the stack was built)

```mermaid
flowchart LR
  S0[0 Env] --> S1[1 LoRA]
  S1 --> S2[2 Template]
  S2 --> S3[3 PPO smoke]
  S3 --> S4[4 RM/CM probe]
  S4 --> S5[5 A/B/C]
```

---

## 4. Stage 5 — the gate (shared B and C)

### 4.1 Rollout → score → maybe replace → PPO

```mermaid
flowchart TB
  ACT[Actor generates y] --> RM[Reward model → r]
  ACT --> CM[Cost model → c]
  CM --> T{"c > 0 ?"}
  T -->|no| KEEP[PPO target = r]
  T -->|yes| REP[PPO target = R_unsafe]
  RM --> KEEP
  subgraph Mag [Only magnitude differs]
    B["Run B: R_unsafe = −2 fixed"]
    C["Run C: R_unsafe = V_MIN − V_MAX<br/>floor −50"]
  end
  Mag --> REP
  KEEP --> PPO[PPO + KL]
  REP --> PPO
```

### 4.2 A / B / C isolation

```mermaid
flowchart TB
  A[Run A · reward only<br/>no cost model] -->|shows disease| NEED[Need a detector]
  NEED --> B[Run B · cost > 0 → −2]
  B -->|shows medicine helps some| C[Run C · cost > 0 → MinMax]
  C --> CLAIM[Thesis: adaptive vs fixed]
```

| Factor | A | B | C |
|---|---|---|---|
| Actor / LoRA / data / seed / ChatML | ✓ | ✓ | ✓ |
| Reward model | ✓ | ✓ | ✓ |
| Cost model | — | ✓ | ✓ |
| Trigger | — | `cost > 0` | `cost > 0` |
| Replacement | — | −2 | \(V_{\MIN}-V_{\MAX}\) |

Everything below the line is the **treatment**. Fair MinMax claims are **within one actor family**.

### 4.3 MinMax bound bookkeeping (Run C only)

```mermaid
flowchart LR
  R[Batch reward scores] --> U[Expand V_MIN / V_MAX]
  U --> G["R_unsafe = V_MIN − V_MAX"]
  G --> F{Below floor −50?}
  F -->|yes| FLOOR[Use −50]
  F -->|no| USE[Use R_unsafe]
  USE --> GATE[Apply only if cost > 0]
  FLOOR --> GATE
```

Phase 2 Run C: \(R_{\text{unsafe}}\) settled near **−9.92**; floor never hit. Mean `train/cost` still drifted — adaptive magnitude ≠ “solved drift.”

---

## 5. Optional Llama track (same A/B/C, matched tokenizer)

Qwen + Beaver still **retokenizes** (Qwen vocab ≠ Llama). Optional track uses a Llama-family actor so Beaver RM/CM share vocab and skip `batch_retokenize`.

```mermaid
flowchart TB
  subgraph QwenTrack [Qwen track · documented results]
    Q[Qwen + ChatML] -->|"decode → encode"| B1[Beaver RM/CM]
  end
  subgraph LlamaTrack [Llama track · optional]
    L[Sheared-LLaMA-1.3B + Alpaca] -->|shared vocab| B2[Beaver RM/CM]
    V[verify_tokenizer_alignment.py<br/>--require-same] --> L
  end
```

| | Qwen track | Llama track |
|---|---|---|
| Actor | Qwen2.5-1.5B-Instruct | `Sheared-LLaMA-1.3B` |
| Template | ChatML | Alpaca |
| Scoring | `batch_retokenize` | same tokenizer when verify passes |
| Claims | Current A/B/C evidence | Within-track A/B/C only; vs Qwen = ablation |

TinyLlama was rejected: vocab matches Llama-2 except Beaver’s `<pad>`, and `pad_token=</s>` so Safe-RLHF never adds `<pad>`.

---

## 6. End-to-end map (one slide worth of structure)

```mermaid
flowchart TB
  RQ[Research question:<br/>Does cost-gated MinMax beat a fixed gate?]
  RQ --> P1[Phase 1: GPT-2 + Detoxify]
  P1 -->|detector fails| P2[Phase 2: Safe-RLHF + Beaver]
  P2 --> A[A: reward only]
  P2 --> B[B: fixed −2]
  P2 --> C[C: MinMax]
  A --> EV[Probe + distribution metrics]
  B --> EV
  C --> EV
  EV --> CL[Claims · open: C rescore,<br/>optional Llama / C2]
```

---

## Where to dig next

| Need | Link |
|---|---|
| Research question | [/worklog/01-research-question.md](/worklog/01-research-question.md) |
| Phase 1 chapters | [/worklog/phase1/README.md](/worklog/phase1/README.md) |
| Phase 2 chapters | [/worklog/phase2/README.md](/worklog/phase2/README.md) |
| Reward vs cost | [/worklog/04-reward-vs-cost.md](/worklog/04-reward-vs-cost.md) |
| A/B/C design | [/worklog/textbook/07-abc-design.md](/worklog/textbook/07-abc-design.md) |
| What we can claim | [/worklog/10-claims.md](/worklog/10-claims.md) |
| Foundations one-pager | [/worklog/fundamentals.md](/worklog/fundamentals.md) |
