<div class="cover-kicker">Architecture · stacks for slides</div>

# Architecture — stacks across phases

Wide horizontal pipelines for Google Slides. Each phase is one **stack**; later phases only mark what **replaced** the previous design. Details live in [Phase 1](/worklog/phase1/README.md) and [Phase 2](/worklog/phase2/README.md).

---

## Slide 1 — Phase 1 stack (baseline design)

```mermaid
flowchart LR
  P[Prompts] --> A[GPT-2 actor]
  A --> D[Detoxify]
  D --> G[MinMax gate]
  G --> PPO[PPO + KL]
  PPO --> C[GPT-2 critic]
  C --> PPO
```

| Slot | Choice |
|---|---|
| Actor / critic | GPT-2 |
| Signal | Detoxify (reward **and** detector) |
| Safety rule | MinMax vs plain PPO+KL |
| Train loop | TRL |

---

## Slide 2 — What broke → what we replace

```mermaid
flowchart LR
  subgraph Keep [Kept]
    K1[MinMax idea]
    K2[PPO + KL]
    K3[Unsafe → replace reward]
  end
  subgraph Drop [Replaced]
    X1[GPT-2] -->|→| Y1[Chat model + LoRA]
    X2[Detoxify as reward] -->|→| Y2[Beaver reward]
    X3[Detoxify as detector] -->|→| Y3[Beaver cost]
    X4[MinMax vs PPO only] -->|→| Y4[A / B / C isolation]
    X5[TRL pilot] -->|→| Y5[Safe-RLHF]
  end
```

---

## Slide 3 — Phase 2 stack (Qwen track)

```mermaid
flowchart LR
  P[Prompts · ChatML] --> A[Qwen + LoRA]
  A --> RM[Beaver reward]
  A --> CM[Beaver cost]
  RM --> G[Gate A / B / C]
  CM --> G
  G --> PPO[Safe-RLHF PPO]
  PPO --> CR[Qwen critic + LoRA]
  CR --> PPO
  A -.-> REF[Reference · adapters off]
  REF -.-> PPO
```

| Slot | Phase 1 | Phase 2 |
|---|---|---|
| Actor | GPT-2 | **Qwen2.5-1.5B-Instruct + LoRA** |
| Helpfulness | Detoxify \(r=1-2\delta\) | **Beaver reward** |
| Detector | Detoxify | **Beaver cost** (`cost > 0`) |
| Compare | MinMax vs PPO+KL | **A / B / C** |
| Framework | TRL | **Safe-RLHF + DeepSpeed** |

---

## Slide 4 — Gate only (B vs C) — same stack, one swap

```mermaid
flowchart LR
  CM[Beaver cost] --> T{cost > 0?}
  T -->|no| R[Keep reward]
  T -->|yes| U[R_unsafe]
  subgraph Swap [Only this changes]
    B[B · fixed −2]
    C[C · V_MIN − V_MAX]
  end
  Swap --> U
  R --> PPO[PPO]
  U --> PPO
```

| Run | Stack same? | What differs |
|---|---|---|
| **A** | No cost | Reward only |
| **B** | Full stack | \(R_{\text{unsafe}} = -2\) |
| **C** | Full stack | \(R_{\text{unsafe}} = V_{\MIN}-V_{\MAX}\) |

---

## Slide 5 — Optional Llama track (one more swap)

Same A/B/C gate. Only actor family + template + tokenizer path change.

```mermaid
flowchart LR
  subgraph Qwen [Documented track]
    Q1[Qwen + ChatML] --> Q2[Beaver] --> Q3[retokenize]
  end
  subgraph Llama [Optional track]
    L1[Sheared-LLaMA + Alpaca] --> L2[Beaver] --> L3[shared vocab]
  end
  Qwen -.->|ablation| Llama
```

| Slot | Qwen track | Llama track |
|---|---|---|
| Actor | Qwen + ChatML | **Sheared-LLaMA + Alpaca** |
| Scoring | retokenize | **same tokenizer** (verify first) |
| A/B/C | within track | within track |

---

## One-line story for the deck

> Phase 1 proved MinMax under a weak detector. Phase 2 **keeps** the replace-on-unsafe idea and **swaps** the stack: chat actor + Beaver reward + Beaver cost + A/B/C.
