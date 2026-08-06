# GRPO vs PPO: Policy Gradient Methods for RLHF
*Study notes — reinforcement learning fine-tuning of language models*

---

## 1. From Base Model to Aligned Model

A randomly initialised model just outputs gibberish. **Pre-training** on the entirety of internet text turns it into a **base model**, which usually just predicts the next token and continues a sentence rather than answering it. **Instruction fine-tuning** on curated instruction/response data (once cheap internet and human-labelled text is exhausted) turns this into an **instruction-tuned model** — one that behaves more conversationally.

From the instruction-tuned model, two further fine-tuning paths are common:

- **Preference fine-tuning** → a **preference-tuned model**. This includes OpenAI-style RLHF using **PPO** (updates weights based on human preference data via a reward model), and **DPO** (Direct Preference Optimization, which optimises directly on preference pairs without a separate RL loop).
- **Reasoning fine-tuning** → a **reasoning-tuned model**. This is where **RLVR** (Reinforcement Learning with Verifiable Rewards) methods such as **PPO** and **GRPO** sit.

> *Added context:* GRPO (Group Relative Policy Optimization) was introduced by DeepSeek (DeepSeekMath, later used in DeepSeek-R1) specifically to make RL fine-tuning for reasoning cheaper — it is the reinforcement learning algorithm behind DeepSeek's reasoning gains.

---

## 2. Policy Gradient Methods — the Shared Foundation

Both GRPO and PPO belong to the family of **policy gradient methods**. They achieve the desired effect — increasing the probability of the highest-reward token/action — by following the policy gradient and performing **gradient ascent** on the policy parameters θ:

$$\theta \leftarrow \theta + \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R$$

where R is the quantity used to weight each step. Two common choices for R:

- **G_t** — the raw return (classic REINFORCE)
- **(G_t − b(s_t))** — the return minus a **baseline**, used to judge whether an action was better or worse than expected at that state

### The return G_t

If a reward model gives intermediate, per-token (or per-sentence) rewards after every action, the return at time t is the sum of future rewards, discounted:

$$G_t = \sum_{k=0}^{T-1} \gamma^k r_t$$

e.g. $G_{T-2} = r_{T-2} + \gamma \, r_{T-1}$. The discount factor γ ∈ [0,1] exponentially reduces the impact of future rewards on the current step.

---

## 3. Getting the Baseline b(s_t)

The advantage of an action is $A_t = G_t - b(s_t)$. Different algorithms differ mainly in *how they estimate this baseline*.

### 3.1 Actor-Critic (PPO)

PPO estimates the baseline using a **state-value function V_φ** — the **critic**. This is a separate model from the policy, with its own trainable weights.

- The critic brings a lot of additional computational overhead.
- It usually needs to be initialised from a large model, roughly doubling the compute/memory requirement (an extra model has to be trained alongside the policy).

### 3.2 GRPO — group-relative baseline

GRPO stops calculating a separate advantage/baseline for every step. Instead of a per-state baseline, it uses one baseline B shared across all time steps in a rollout — unlike, say, a video game where feedback (score) arrives constantly at every step.

Concretely, GRPO samples a **group of trajectories** (multiple completions) for the same prompt, and takes the average of their rewards as the baseline. This is where the name "Group Relative" comes from, and it is also why GRPO needs no separate critic network — the group itself supplies the baseline.

$$A_t = \frac{R - \text{mean}(r_g)}{\text{std}(r_g)}$$

R is the final reward for a sampled completion; r_g is the set of rewards across the sampled group. Dividing by the standard deviation normalises the advantage values across the group.

---

## 4. The Variance Problem — and How PPO Fixes It

The vanilla policy gradient objective is:

$$L^{PG} = \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) \cdot A_t$$

**Problem: variance.** This shows up when policy updates are too large from one step to the next, destabilising training.

Two fixes for this exist in the literature:

- **TRPO** (Trust Region Policy Optimization) fixes this using a second-order derivative (a KL-constrained trust region).
- **PPO** fixes this more cheaply, via **clipping**.

### PPO's clipped objective

Define the probability ratio between the current and old (pre-update) policy:

$$r_\theta = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

PPO clips this ratio to the range $1-\varepsilon \le r_\theta \le 1+\varepsilon$, so the policy can't move too far from its previous version in a single update:

$$L^{PPO} = \sum_{t=0}^{T-1} \min\Big( r_\theta \cdot A_t,\ \text{clip}(r_\theta, 1-\varepsilon, 1+\varepsilon) \cdot A_t \Big)$$

Taking the min of the raw and clipped objective gives a pessimistic (lower) bound on the policy improvement, which is what keeps training stable.

---

## 5. GRPO's Loss — Adding a KL Penalty

GRPO reuses PPO's clipped surrogate objective, but adds an explicit KL-divergence penalty term against a fixed reference policy:

$$L^{GRPO} = L^{PPO} - \beta\, D_{KL}\big[\, \pi_\theta \,\|\, \pi_{ref} \,\big]$$

$D_{KL}$ is a distance measure between the current policy $\pi_\theta$ and a reference policy $\pi_{ref}$ — the state of the model *before* this round of fine-tuning (e.g. the preference-tuned model). β controls how strongly the policy is pulled back toward the reference, preventing it from drifting too far from known-good behaviour while it chases reward.

---

## 6. Summary: PPO vs GRPO

| Aspect | PPO | GRPO |
|---|---|---|
| Baseline b(s_t) | Learned critic V_φ(s_t) — a separate trained model | Mean reward across a sampled group of completions for the same prompt |
| Extra network required | Yes — the critic (roughly doubles compute/memory) | No separate critic needed |
| Advantage estimate | A_t = G_t − b(s_t) | A_t = (R − mean(r_g)) / std(r_g) |
| Stability mechanism | Clipped ratio objective | Clipped ratio objective + explicit KL penalty vs. reference policy |
| Origin | OpenAI (Schulman et al., 2017) | DeepSeek (DeepSeekMath / DeepSeek-R1) |

> *Added context:* Both algorithms are implemented in Hugging Face's **TRL** library (`PPOTrainer` and `GRPOTrainer`), which handles the rollout sampling, reward computation, and the clipped/KL-regularised loss described above.
