# Chapter 8. Policy-Based Reinforcement Learning

Consider the problem of choosing how much force to apply at a robot arm's
joint. This "action" can be any **continuous value** from -10Nm to +10Nm.
Chapter 6's Q-learning has to compute \\(\max_{a'} Q(s',a')\\) at every
step — but with a continuous action space, "enumerate every possible
action and find the max" is simply impossible; there are infinitely many
candidates to check.

## 8.1 Learning the Policy Directly, Without Going Through Q

So far (Chapters 6-7), we learned \\(Q(s,a)\\) first and then obtained the
policy indirectly, \\(\pi(s) = \arg\max_a Q(s,a)\\). **Policy-Based
Methods** skip that intermediate step and represent the policy
\\(\pi_\theta(a|s)\\) itself — a function with parameters \\(\theta\\)
that directly outputs a probability distribution over actions — as a
neural network, trained directly. "How likely is this action" is
well-defined even in a continuous action space, so the earlier problem
disappears naturally.

## 8.2 Parameterizing the Policy

For a discrete action space, the policy is made a probability distribution
by applying softmax to the network's output:

\\[\pi_\theta(a|s) = \text{softmax}(f_\theta(s))_a\\]

\\(f_\theta(s)\\) is a neural network that takes state \\(s\\) as input
and outputs a raw score (logit) for each action. The goal is to find the
\\(\theta\\) that maximizes the expected cumulative reward:

\\[J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}[R(\tau)]\\]

\\(\tau\\) is a trajectory (the whole episode), and \\(R(\tau)\\) is that
trajectory's total reward.

## 8.3 The Strange Problem of "Differentiating a Reward"

The problem is that differentiating the goal "find the \\(\theta\\) that
maximizes expected reward" directly runs into a strange wall: expected
reward is an average over "the outcomes of acting randomly," but that
randomness itself depends on \\(\theta\\). Differentiating \\(J(\theta)\\)
with respect to \\(\theta\\) directly requires differentiating the
probability distribution \\(\pi_\theta\\) itself, which breaks the
expectation (integral) form.

## 8.4 The Log-Derivative Trick

The **log-derivative trick** uses the following identity:

\\[\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta\\]

(this follows directly from the calculus identity \\(\nabla \log f =
\nabla f / f\\)). This substitution turns the derivative into a gradient
of the log-probability instead of taking it outside the probability
directly, letting us reorganize the expression back into an expectation.
The final result is the **Policy Gradient Theorem**:

\\[\nabla\_\theta J(\theta) = \mathbb{E}\_\tau\left[\sum\_t \nabla\_\theta \log
\pi\_\theta(a\_t|s\_t) \\, G\_t\right]\\]

\\(G_t\\) is the discounted cumulative reward from timestep \\(t\\)
onward (Chapter 5's return). Intuition: "shift \\(\theta\\) to increase
the probability (\\(\log \pi_\theta(a_t|s_t)\\)) of actions actually taken
in trajectories with a good outcome (large \\(G_t\\)), and decrease the
probability of actions taken in trajectories with a bad outcome."

This derivation (considered challenging even at the graduate level) is the
centerpiece of this chapter's exercises, and the worksheet version aims
just to solidify one core idea: "why does taking a log solve the
problem?"

## 8.5 The REINFORCE Algorithm

REINFORCE is simply the Policy Gradient Theorem implemented as gradient
ascent (`+=`, since we're maximizing):

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

## 8.6 Why This Is Model-Free

The key step in deriving the Policy Gradient Theorem is that when you take
the log of a trajectory's probability, \\(P(\tau;\theta) = \prod_t
\pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t,a_t)\\), **the environment's
transition probability term \\(P(s_{t+1}|s_t,a_t)\\) doesn't depend on
\\(\theta\\), so it vanishes when differentiated.** That means only the
policy \\(\pi_\theta\\) remains in the final gradient expression — the
environment model never appears at all. This is where model-free RL's
core property comes from: you can learn a policy without knowing how the
environment works.

## 8.7 Actor-Critic: An Improvement That Reduces Variance

REINFORCE uses \\(G_t\\) (the actually observed cumulative reward)
directly, but since this comes from sampling a single episode, it's noisy
(high variance). **Actor-Critic** reduces this variance by subtracting a
baseline estimated with the value function (Critic) covered in Chapters
5-6, instead of using \\(G_t\\) alone (\\(G_t - V(s_t)\\), this
difference is called the **advantage**) — a structure that trains a
policy (Actor) and a value function (Critic) simultaneously. The detailed
derivation goes beyond this semester's scope, but the idea that "training
is more stable with help from a value function than training a policy
alone" is worth remembering.

**Q-learning is an indirect strategy — "evaluate how good things are, then
pick the best" — while policy-based methods learn "what to do" directly,
and work in continuous action spaces too.**

---

## Exercises

**1. (Coding)** For a simple discrete action space (2 actions) with a
softmax policy, complete the REINFORCE update for one episode (key lines
left blank):

```python
import math

def softmax_policy(theta, state_feature):
    # ADD ADDITIONAL CODE HERE!!
    # logits = [theta[0]*state_feature, theta[1]*state_feature]
    # compute softmax probabilities (subtract max for numerical stability)
    return probs

def reinforce_update(theta, episode, alpha, gamma):
    # episode: [(state_feature, action, reward), ...]
    # ADD ADDITIONAL CODE HERE!!
    # compute the return G_t cumulatively from the end (with discounting)
    for t, (s, a, r) in enumerate(episode):
        probs = softmax_policy(theta, s)
        # ADD ADDITIONAL CODE HERE!!
        # grad_log_pi: if action==0, [1-probs[0], -probs[1]]*s; if action==1, [-probs[0], 1-probs[1]]*s
        # update theta: theta += alpha * G[t] * grad_log_pi
    return theta
```

**2. (Hand derivation, Tier C — top-priority fallback)** Under policy
\\(\pi_\theta\\), derive that the gradient of the expected return
\\(J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}[R(\tau)]\\) equals

\\[\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau}\left[\sum\_t \nabla\_\theta \log
\pi\_\theta(a\_t|s\_t) \\, G\_t\right]\\]

using the log-derivative trick. (Advanced — for strong math students /
volunteers only; most should use the worksheet below as the default.)

**Fill-in-the-blank worksheet version** (default):

```
Goal: differentiate J(theta) = sum_tau P(tau; theta) * R(tau) with respect to theta.
Problem: differentiating P(tau; theta) directly breaks the expectation (integral) form,
so it can't be estimated from samples.

Log-derivative trick: grad(f) = f * grad(log f)

Step 1: grad_theta P(tau;theta) = P(tau;theta) * ______________  [apply the log-derivative trick]

Step 2: grad_theta J(theta) = sum_tau ______________ * R(tau)
                             = E_tau[ grad_theta log P(tau;theta) * R(tau) ]

Step 3: trajectory probability P(tau;theta) = prod_t pi_theta(a_t|s_t) * (environment transition probs, independent of theta)
        therefore grad_theta log P(tau;theta) = ______________

Conclusion: grad_theta J(theta) = E_tau[ (sum_t grad_theta log pi_theta(a_t|s_t)) * R(tau) ]
```

**Confirm correctness**: explain in one sentence why the fact "the
environment's transition probability doesn't depend on theta" in Step 3
matters (hint: without it, model-free RL's core property — learning a
policy without knowing the environment model — breaks down).
