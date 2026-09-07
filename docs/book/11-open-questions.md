# 11. Open questions

## Blocking scientific questions

1. **Run C training** — Do \(V_{\MIN}/V_{\MAX}\) move meaningfully with an unbounded cost-triggered signal? Does \(R_{\text{unsafe}}\) exceed B’s fixed \(-2\)?
2. **Average vs probe** — Can any gate magnitude arrest distribution-wide cost drift, or only reshape hard probes?
3. **Reward model transplant** — Beaver RM was trained on Alpaca-7B-style responses. How sensible are its rankings on Qwen prose?

## Engineering debt that still bites

- Resume-from-adapter is not implemented (snapshots make restarts survivable but lossy).
- `batch_retokenize` Qwen→LLaMA round-trip drift not systematically audited beyond probes.
- Faulty nodes (`mscluster65`, `83`, `111`) still worth a Help Desk ticket.

## Optional extensions (after C)

- Category-scoped MinMax bounds (Phase 1 had this; Stage 5 v1 is global).
- Compare against PKU’s PPO-Lag / reward shaping under the same Qwen+LoRA setup.
- Multi-seed confirmation of A/B qualitative story.

When an item closes, move the result into the relevant chapter and shrink this list — do not leave resolved questions here.
