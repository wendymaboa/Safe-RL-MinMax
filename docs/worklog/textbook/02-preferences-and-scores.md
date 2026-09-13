# 2. Preferences → scalar models

## From pairwise labels to a number

Humans (or AI labelers) rarely give absolute “helpfulness = 7.2”. They say: for prompt \(x\), reply \(y_w\) is better than \(y_l\) (winner / loser).

A **Bradley–Terry** model assumes there is a latent scalar \(r(x,y)\) such that

\[
P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big).
\]

Train a neural net (usually an LM backbone + scalar head) to maximise likelihood of the labeled pairs. That net **is** your reward model (or cost model, if pairs were about safety).

## What the number means — and does not

| True | False |
|---|---|
| Higher \(r\) means preferred *relative to alternatives the model was trained on* | “\(r=1.4\) means 40% helpful” |
| Differences \(r_a - r_b\) are the object of training | Absolute thresholds are sacred without calibration |
| Out-of-distribution text can be confidently wrong | The RM is a ground-truth human |

<div class="finding caution">
<span class="label">Caution</span>
Your Stage 4 note was right: pairwise models justify <em>orderings</em>. Using <code>cost &gt; 0</code> as a gate is an extra modeling choice. It was <strong>empirically</strong> supported by your probe (all benign &lt; 0, harmful &gt; 0), and it matches PKU’s safe/unsafe convention — but it is not implied by Bradley–Terry alone.
</div>

## Architecture of Beaver-style models

Conceptually:

```text
tokens → Llama-like transformer → last (or EOS) hidden state → linear head → scalar
```

Safe-RLHF exposes `.end_scores` — the scalar at the end of the sequence. That is what your PPO loop reads.

## Helpfulness pairs vs safety pairs

Same math, different datasets:

- **Reward model:** chosen = more helpful / better answer.  
- **Cost model:** chosen = more harmful / less safe (label semantics flipped into a “cost” you want **low**).

If you mix these up, every plot in Stage 5 becomes nonsense.

## Ranking probes (how you already thought like an expert)

Before trusting PPO, you ranked fixed strings:

- harmful instructions vs refusal vs `"Advertisements"`  

That is the correct preflight: **does the scorer order the world the way the paper claims?** Your cost probe did. An earlier mistaken probe asked the *reward* model to prefer refusals — that test was invalid because refusals are often *less helpful*.

## Expert checklist

- [ ] I can write the Bradley–Terry formula and say what is trained.  
- [ ] I refuse to interpret absolute RM scores without a calibration story.  
- [ ] I can design a 4-string ranking probe for a new RM/CM.  
- [ ] I know why `"Advertisements"` must be checked on **both** RM and CM.

**Next:** [Helpfulness vs harmlessness](/worklog/textbook/03-reward-vs-cost.md)
