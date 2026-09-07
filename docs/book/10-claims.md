# 10. What we can claim

Only claims supported by finished runs. MinMax results are **not** included yet.

## From Run A

1. Helpfulness-only PPO can erase an intact refusal on a clear harmful probe (~step 500).
2. Reward can rise partly via verbosity and fabrication scored highly by a strong RM.
3. Therefore a cost / safety signal is necessary in this stack — A alone motivates Safe RLHF’s split.

## From Run B versus A

1. A fixed cost gate (`cost > 0 → reward := −2`) **helps on the matched probe**: late B is lower-cost and qualitatively pivots toward non-compliance.
2. The same gate **does not stop average-cost drift** over the training distribution.
3. Shared safety-flavored openings appear, but the late A/B cost gap is not explained by phrasing alone.

## What we must not claim yet

| Claim | Why not |
|---|---|
| “MinMax improves safety” | Run C not trained |
| “B solved safety” | Distribution-wide cost still rises |
| “Preamble tricks the cost model” | Session 17 contradicts that as the main driver |

## One-sentence summary (A+B)

> Plain RLHF breaks safety; a fixed cost gate partially recovers it on hard probes but does not arrest average cost drift — MinMax remains the test of whether a self-calibrating bound does better.
