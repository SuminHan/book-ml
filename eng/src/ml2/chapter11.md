# Chapter 11. Advanced Policy Optimization: PPO (Proximal Policy Optimization)

Chapter 10's REINFORCE and Actor-Critic share a common limitation: once
you collect one trajectory and use it for a single gradient ascent step,
that data gets thrown away (on-policy) — the moment the policy changes,
probabilities/gradients computed from the old data are no longer accurate.
On top of that, even a single gradient ascent step that's too large can
suddenly push the policy in a bad direction it can't recover from — this
is disastrous in settings like robot control, where "one bad update
collapsing the policy" is hard to walk back from. **PPO** (Proximal Policy
Optimization, Schulman et al., proposed by OpenAI in 2017) is a practical
solution that reuses data several times while still preventing the policy
from moving too far in a single update.

## 11.1 The Intuition of a Trust Region

Imagine descending a mountain in thick fog, where you can only see one
step ahead. If you take a big step blindly in the steepest downhill
direction, you might not know how the terrain changes in that direction
and could fall off a cliff. The safe strategy is "limit the size of each
step to a range you can trust (a trust region)" — this is exactly PPO's
core idea. By constraining how much the policy can change in a single
update, it stays within a narrow range where each step is actually
guaranteed to be an improvement.

## 11.2 The Probability Ratio and Importance Sampling

The key tool is the **probability ratio** — the ratio between the
probability the policy we're currently updating, \\(\pi_\theta\\), assigns
to an action, and the probability the policy used to collect the data,
\\(\pi_{\theta_{\text{old}}}\\), assigned to that same action:

\\[r_t(\theta) := \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)},
\quad r_t(\theta_{\text{old}}) = 1\\]

\\(r_t > 1\\) means the action is now preferred more than before;
\\(r_t < 1\\) means less. This ratio lets us reuse "data collected by the
old policy" from "the new policy's perspective" (**importance sampling** —
the very same tool that first appeared for MC's off-policy learning in
Section 5.4):

\\[L^{IS}(\theta) = \mathbb{E}_t[r_t(\theta) A_t]\\]

\\(A_t\\) is the advantage defined in Section 10.7, \\(A_t := G_t -
V(s_t)\\). The problem is that simply maximizing this objective as-is can
let training run away, pushing \\(r_t\\) arbitrarily high (endlessly
reinforcing the same action) — the data gets reused, but there's no
guarantee anywhere that "the policy doesn't move too far."

## 11.3 GAE: Estimating the Advantage More Stably

Section 10.7's advantage, \\(A_t = G_t - V(s_t)\\), still uses the actual
return \\(G_t\\) directly, so its variance stays high. **GAE**
(Generalized Advantage Estimation) applies the exact same idea from
Chapter 7's n-step/eligibility traces to advantage estimation — it takes a
weighted average of several \\(n\\)-step advantage estimates using
\\(\lambda\\), finding a compromise that lowers variance while accepting
just enough bias:

\\[A_t^{\text{GAE}} = \sum_{k=0}^\infty (\gamma\lambda)^k \delta_{t+k},
\qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\\]

\\(\delta_t\\) is exactly Section 6.1's TD error — GAE boils down to
"exponentially weighting and summing TD errors from multiple time steps,
the same way eligibility traces do, to produce a single advantage
estimate." With \\(\lambda=0\\), it reduces to a TD(0)-based advantage
(low variance, high bias); with \\(\lambda=1\\), it approaches a Monte
Carlo-based advantage (unbiased, high variance) — Chapter 7.2's dial shows
up here again. Real-world PPO implementations almost always use \\(A_t\\)
computed via GAE.

## 11.4 Clipping: Preventing Runaway Updates

PPO's solution is simple — if \\(r_t\\) strays outside
\\([1-\epsilon, 1+\epsilon]\\) (usually \\(\epsilon = 0.2\\)), clip away
whatever gain came from going outside that range:

\\[L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]\\]

Why the min: taking the more "pessimistic" (smaller) of the clipped and
unclipped values as the objective self-limits the reward for pushing the
policy too far in one direction. Intuition: if the advantage is positive
(a good action) and \\(r_t\\) has already exceeded \\(1+\epsilon\\),
pushing further gives zero gradient (it's already been pushed enough) —
conversely, if the advantage is negative (a bad action) and \\(r_t\\) has
already dropped below \\(1-\epsilon\\), the gradient vanishes there too
(it's already been suppressed enough). The result is an implicit
**trust region** that neither over-reinforces good actions nor
over-suppresses bad ones — Section 11.1's "range you can trust," achieved
with a single clipping operation instead of having to solve an explicit
constrained optimization problem.

```python
def ppo_clip_loss(ratio, advantage, epsilon=0.2):
    unclipped = ratio * advantage
    clipped = max(min(ratio, 1 + epsilon), 1 - epsilon) * advantage
    return min(unclipped, clipped)  # one sample's worth of the objective (to be maximized)
```

## 11.5 Why PPO Became the Standard

Before PPO, the same problem (policy collapse from overly large updates)
was solved by **TRPO** (Trust Region Policy Optimization, 2015) — an
explicit KL-divergence constraint combined with heavyweight optimization
involving second derivatives (the Fisher information matrix), conjugate
gradient, and line search. PPO replaces that entire complex machinery
with a single line of clipping, while achieving similar stability — a
choice of "an approximation that's easy to implement and works well in
practice" over "a theoretically airtight guarantee" (TRPO's monotonic
improvement guarantee).

This practicality is why PPO became the most widely used policy-based
algorithm today: it's used in robot locomotion control, OpenAI Five
(Dota 2), AlphaStar (StarCraft II), and as the core tool of RLHF for
tuning language models to human feedback — general enough to apply
directly to both continuous control (a robot's joint forces) and discrete
choice (picking the next token).

**Where Q-learning is an indirect strategy — "evaluate how good things
are, then pick the best" — policy-based methods learn "what to do"
directly. GAE reduces the variance of the advantage estimate, and clipping
keeps updates based on that estimate from growing too large — this
stability is exactly why PPO is the standard for the continuous control
problems in the robot simulation we'll cover in Chapters 13-14.**

---

## Exercises

**1. (Coding)** Complete `ppo_clip_loss` above, and `td_error`, which
computes one term of GAE, \\(\delta_t = r_t + \gamma V(s_{t+1}) -
V(s_t)\\) (key lines left blank):

```python
def td_error(r, V_s, V_s_next, gamma):
    # ADD ADDITIONAL CODE HERE!!

def ppo_clip_loss(ratio, advantage, epsilon=0.2):
    # ADD ADDITIONAL CODE HERE!!
    # unclipped = ratio * advantage
    # clipped = ratio clipped to [1-epsilon, 1+epsilon], times advantage
    # return the smaller of the two

print(ppo_clip_loss(ratio=1.5, advantage=1.0, epsilon=0.2))  # 1.2 (the clipped value is smaller)
print(ppo_clip_loss(ratio=0.5, advantage=1.0, epsilon=0.2))  # 0.5 (the unclipped value is smaller)
print(ppo_clip_loss(ratio=1.5, advantage=-1.0, epsilon=0.2)) # -1.5 (note the sign flip when advantage is negative)
```

**2. (Conceptual)** Explain which methods from Chapter 7 correspond to
GAE's \\(\lambda=0\\) and \\(\lambda=1\\) respectively, and describe which
side of the bias/variance tradeoff each extreme suffers from.

**3. (Hand derivation, Tier B — hints provided)** Show that substituting
\\(\lambda = 0\\) into \\(A_t^{\text{GAE}} = \sum_{k=0}^\infty
(\gamma\lambda)^k \delta_{t+k}\\) leaves only \\(A_t^{\text{GAE}} =
\delta_t\\) (a single TD error). Using the definition \\(\delta_t = r_t +
\gamma V(s_{t+1}) - V(s_t)\\), confirm this is exactly the same as Section
10.7's 1-step advantage approximation, \\(A_t \approx r_t + \gamma
V(s_{t+1}) - V(s_t)\\).

**Confirm correctness**: as \\(\lambda\\) increases from 0 toward 1, GAE
starts including TD errors from further into the future — explain in one
sentence why this means it's "moving toward Monte Carlo."
