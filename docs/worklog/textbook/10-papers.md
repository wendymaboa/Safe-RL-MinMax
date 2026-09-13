# 10. Papers & further study

Read to extract **mechanisms**, not vibes. For each paper, write three bullets: problem, method, what it implies for *your* A/B/C.

## Core stack

| Paper | Extract |
|---|---|
| **Christiano et al.** — Deep RL from Human Preferences | Preferences → reward model → RL |
| **Ziegler et al. / Stiennon et al.** — Learning to summarise / early LM RLHF | RM + PPO on LMs |
| **Ouyang et al.** — InstructGPT | Production RLHF recipe; SFT→RM→PPO; KL to reference |
| **Bai et al.** — Training a Helpful & Harmless Assistant | Helpfulness–harmlessness tension; why one scalar hurts |
| **Dai et al.** — Safe RLHF | Separate RM & CM; Beaver; PPO-Lag; PKU data |

## Adjacent / mechanism

| Paper / topic | Extract |
|---|---|
| **Schulman et al.** — PPO | Clipping, stability (you consume, not re-derive) |
| **Hu et al.** — LoRA | Why adapters make 1.5B+7B scorers feasible |
| **Goodhart / reward hacking surveys** (e.g. Skalse, Pan, et al. lines) | Proxy vs intent; your EduNipple / Advertisements stories |
| **ROSARL / MinMax-bound penalty** (your lineage) | Adaptive unsafe replacement; compare carefully to what you implemented |
| **Constitutional AI / RLAIF** | Alternative to human cost labels — context, not your method |

## How to read Dai et al. for your thesis

1. Note the **split** RM/CM — you keep this.  
2. Note **PPO-Lag** — you replace this with a hard gate.  
3. Note evaluation (harmlessness vs helpfulness) — mirror with probe + reward metrics.  
4. Never imply your gate **is** their algorithm.

## Optional deep dives (after the above)

- Tokenisers & multilingual Qwen quirks  
- GAE / advantage estimation details  
- Distributed RLHF (OpenRLHF, verl) if you rebuild the stack  
- Formal constrained RL (CMDP) vs heuristic gating  

## Expert checklist

- [ ] I can cite InstructGPT vs Safe RLHF roles in one breath.  
- [ ] I can say what I took from Dai et al. and what I changed.  
- [ ] I have a one-page reading notes file for each core paper.  
- [ ] I know which failure-mode literature covers Phase 1 vs Phase 2.

---

**Back to:** [Textbook home](/worklog/textbook/README.md) · [Claims](/worklog/10-claims.md) · [Foundations one-pager](/worklog/fundamentals.md)
