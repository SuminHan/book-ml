# Chapter 10. Policy-Based Reinforcement Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter10_reinforce_ppo.ipynb)

Consider the problem of deciding how much force to apply at a robot arm's
joint. This "action" can be any **continuous value** from -10Nm to +10Nm.
Chapter 6/9's Q-learning/DQN needs to compute \\(\max_{a'} Q(s',a')\\)
every step, but if actions are continuous, enumerating "every possible
action" to find the max is simply impossible — you can't evaluate
infinitely many candidates. The robot simulation we'll cover starting in
Chapter 13 uses exactly this kind of continuous action space, so we need
to confront this problem head-on.

## 10.1 Learning the Policy Directly, Without Going Through Q

So far (Chapters 6, 9), we've learned \\(Q(s,a)\\) first, and obtained the
policy \\(\pi(s) = \arg\max_a Q(s,a)\\) only indirectly from it.
**Policy-based methods** skip this intermediate step, and represent the
policy \\(\pi_\theta(a|s)\\) itself — a function with parameters
\\(\theta\\) that directly outputs a probability distribution over
actions — as a neural network and train it directly. Even in a continuous
action space, "what's the probability density of this action" is
well-defined, which naturally resolves the problem above.

## 10.2 Parameterizing the Policy

In a discrete action space, we turn the policy into a probability
distribution by applying softmax to the network's output:

\\[\pi_\theta(a|s) = \text{softmax}(f_\theta(s))_a\\]

\\(f_\theta(s)\\) is a neural network that takes state \\(s\\) as input
and outputs a raw score (logit) for each action. The goal is to find the
\\(\theta\\) that maximizes expected cumulative reward:

\\[J(\theta) = \mathbb{E}\_{\tau \sim \pi_\theta}[R(\tau)]\\]

\\(\tau\\) is a trajectory (an entire episode), and \\(R(\tau)\\) is that
trajectory's total reward.

## 10.3 The Strange Problem of "Differentiating a Reward"

The problem is that trying to directly differentiate the goal "find the
\\(\theta\\) that maximizes expected reward" runs into a strange wall:
expected reward is an average over "outcomes obtained by acting
randomly," and that very randomness depends on \\(\theta\\). Trying to
differentiate \\(J(\theta)\\) with respect to \\(\theta\\) directly
requires differentiating the probability distribution \\(\pi_\theta\\)
itself, which breaks the expectation (integral) form.

## 10.4 The Log-Derivative Trick

The **log-derivative trick** uses the following identity:

\\[\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta\\]

(This follows directly from the calculus identity \\(\nabla \log f =
\nabla f / f\\).) This substitution moves the derivative **outside** the
probability instead, replacing it with the gradient of the log
probability, which lets us rearrange things back into expectation form.
The final result is the **Policy Gradient Theorem**:

\\[\nabla_\theta J(\theta) = \mathbb{E}\_\tau\left[\sum_t \nabla_\theta \log
\pi_\theta(a_t|s_t) \, G_t\right]\\]

\\(G_t\\) is the discounted cumulative reward from time \\(t\\) onward
(Chapter 3's return). Intuition: "shift \\(\theta\\) so that the
probability of the action actually chosen (\\(\log
\pi_\theta(a_t|s_t)\\)) increases along trajectories with good outcomes
(large \\(G_t\\)), and decreases along trajectories with bad outcomes."

This derivation (considered tricky even at the graduate level) is the
centerpiece of this chapter's exercises, and the worksheet version's goal
is simply to firmly grasp the one key idea: "why does taking the log solve
the problem?"

## 10.5 The REINFORCE Algorithm

Implementing the Policy Gradient Theorem directly via gradient ascent
(maximizing, so `+=`) gives REINFORCE:

```python
import math

def softmax_policy(theta, state_feature):
    logits = [theta[0]*state_feature, theta[1]*state_feature]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

def reinforce_update(theta, episode, alpha, gamma):
    # episode: [(state_feature, action, reward), ...]
    T = len(episode)
    G = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = episode[t][2] + gamma * running
        G[t] = running
    for t, (s, a, r) in enumerate(episode):
        probs = softmax_policy(theta, s)
        if a == 0:
            grad_log_pi = [(1 - probs[0]) * s, -probs[1] * s]
        else:
            grad_log_pi = [-probs[0] * s, (1 - probs[1]) * s]
        theta[0] += alpha * G[t] * grad_log_pi[0]
        theta[1] += alpha * G[t] * grad_log_pi[1]
    return theta
```

## 10.6 Why This Is Model-Free

The key step in deriving the Policy Gradient Theorem is this: when you
take the log of a trajectory's probability, \\(P(\tau;\theta) = \prod_t
\pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t,a_t)\\), the **environment's
transition probability term \\(P(s_{t+1}|s_t,a_t)\\) is independent of
\\(\theta\\) and vanishes upon differentiation.** That is, the final
gradient expression contains only the policy \\(\pi_\theta\\) — the
environment model never appears at all. This is where the core
model-free RL property comes from: you can learn a policy without ever
knowing how the environment works.

## 10.7 Actor-Critic: Reducing Variance

REINFORCE uses \\(G_t\\) (the actually-observed cumulative reward) as-is,
but this is the result of sampling a single episode, so it's noisy (high
variance). **Actor-Critic** subtracts a baseline estimated by a value
function (the Critic, from Chapters 4/6) instead of using \\(G_t\\)
directly, reducing variance (\\(G_t - V(s_t)\\); this difference is called
the **advantage**) — a structure that trains the policy (Actor) and the
value function (Critic) simultaneously. The detailed derivation is beyond
this semester's scope, but the idea — "getting help from a value function
trains more stably than training the policy alone" — is worth
remembering; this advantage is reused directly in Chapter 11's PPO.

**Where Q-learning is an indirect strategy — "evaluate how good things
are, then pick the best" — policy-based methods learn "what to do" itself
directly, even in continuous action spaces. As Chapter 4 proved, an
optimal policy \\(\pi^*\\) is always guaranteed to exist for a finite MDP —
policy gradient methods head directly toward the target that existence
guarantees.**

---

## Exercises

**1. (Coding)** For a simple discrete action space (2 actions) with a
softmax policy, complete a REINFORCE update over one episode (key lines
left blank):

```python
import math

def softmax_policy(theta, state_feature):
    # ADD ADDITIONAL CODE HERE!!
    # logits = [theta[0]*state_feature, theta[1]*state_feature]
    # compute softmax probabilities (subtract max before exp for numerical stability)
    return probs

def reinforce_update(theta, episode, alpha, gamma):
    # episode: [(state_feature, action, reward), ...]
    # ADD ADDITIONAL CODE HERE!!
    # compute the return G_t backward from the end (with discounting)
    for t, (s, a, r) in enumerate(episode):
        probs = softmax_policy(theta, s)
        # ADD ADDITIONAL CODE HERE!!
        # grad_log_pi: if action=0, [1-probs[0], -probs[1]]*s; if action=1, [-probs[0], 1-probs[1]]*s
        # update theta: theta += alpha * G[t] * grad_log_pi
    return theta
```

**2. (Conceptual)** Explain, in two or three sentences, why Actor-Critic
has lower variance than REINFORCE, connecting it to the fact that
"subtracting a baseline doesn't change the expected value of the
gradient" (an intuitive-level explanation is enough).

**3. (Hand derivation, Tier C — top priority fallback)** Using the
log-derivative trick, derive that the gradient of expected return
\\(J(\theta) = \mathbb{E}\_{\tau \sim \pi_\theta}[R(\tau)]\\) under policy
\\(\pi_\theta\\) is

\\[\nabla_\theta J(\theta) = \mathbb{E}\_{\tau}\left[\sum_t \nabla_\theta \log
\pi_\theta(a_t|s_t) \, G_t\right]\\]

(An advanced exercise, mainly for stronger students who want it — most
students should default to the worksheet version below.)

**Fill-in-the-blank worksheet version** (default):

```
Goal: differentiate J(theta) = sum_tau P(tau; theta) * R(tau) with respect to theta.
Problem: differentiating P(tau; theta) directly breaks the expectation (integral) form, so it can't be estimated from samples.

Log-derivative trick: grad(f) = f * grad(log f)

Step 1: grad_theta P(tau;theta) = P(tau;theta) * ______________  [apply the log-derivative trick]

Step 2: grad_theta J(theta) = sum_tau ______________ * R(tau)
                             = E_tau[ grad_theta log P(tau;theta) * R(tau) ]

Step 3: the trajectory probability P(tau;theta) = prod_t pi_theta(a_t|s_t) * (environment transition probability, independent of theta)
        so grad_theta log P(tau;theta) = ______________

Conclusion: grad_theta J(theta) = E_tau[ (sum_t grad_theta log pi_theta(a_t|s_t)) * R(tau) ]
```

**Confirm correctness**: explain in one sentence why the fact in Step 3
that "the environment's transition probability is independent of theta"
matters (hint: without this, the core model-free RL property — being able
to learn from the policy alone without knowing the environment model —
would break down).
