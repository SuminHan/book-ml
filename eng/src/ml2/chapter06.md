# Chapter 6. Temporal-Difference Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter06_q_learning.ipynb)

In 1989, Chris Watkins proposed an algorithm called **Q-learning** in his
PhD thesis. Chapter 5's Monte Carlo could learn without a model, but you
had to wait for an episode to end before you could compute a return.
**Temporal-Difference (TD) learning** removes that wait entirely — it
takes just one step, then immediately updates by the difference between
"the current estimate" and "the reward just observed plus the next
state's estimate." No model needed, no waiting for episodes to end — the
most widely used compromise in reinforcement learning.

## 6.1 TD(0): Update After Just One Step

Recall Chapter 4's Bellman equation, \\(V^\pi(s) = R(s,\pi(s)) + \gamma
V^\pi(s')\\). Policy evaluation (model-based) computed the right-hand side
as an expectation over **every** possible next state. TD(0) replaces that
expectation with a single **observed sample**:

\\[V(s) \leftarrow V(s) + \alpha\big[r + \gamma V(s') - V(s)\big]\\]

The bracketed term is called the **TD error**. Comparing to MC makes the
key difference clear: MC uses the actual return \\(G_t\\) (the real sum
of rewards to the end of the episode) as its target, while TD uses
\\(r + \gamma V(s')\\) (one step's actual reward, plus an **estimate** of
the next state's value) — since it updates using a not-yet-certain
estimate of itself, this is called **bootstrapping**. Bootstrapping is
exactly what makes learning possible every single step, even without an
episode ending, even in tasks that never end at all.

| | Monte Carlo | Temporal-Difference (TD) |
|---|---|---|
| Target | actual return \\(G_t\\) (to the end of the episode) | \\(r + \gamma V(s')\\) (one step + estimate) |
| When it updates | after the episode ends | immediately, every step |
| Bias/Variance | unbiased, high variance | biased (uses an estimate), low variance |

## 6.2 Q-Learning: Model-Free, With State-Action Values

Chapter 4's \\(V(s)\\) was "the value of a state" — on its own, this
isn't enough to choose the next action, since you'd need the transition
probabilities \\(P\\) to know which action leads to a good next state (a
point already made in Section 5.3). **Q-learning** learns a more
fine-grained **Q-function** \\(Q(s,a)\\) — "the expected cumulative reward
of taking action \\(a\\) in state \\(s\\), then acting optimally
thereafter." Once you know \\(Q(s,a)\\), the optimal policy is immediately
just \\(\pi^*(s) = \arg\max_a Q(s,a)\\).

Every time the agent takes action \\(a\\) in state \\(s\\), receives
reward \\(r\\), and arrives at new state \\(s'\\):

\\[Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)\right]\\]

```python
def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    max_next_q = max(Q[s_next].values()) if isinstance(Q[s_next], dict) else max(Q[s_next])
    td_error = r + gamma * max_next_q - Q[s][a]
    Q[s][a] += alpha * td_error
    return Q
```

## 6.3 SARSA: Using the Action Actually Taken

Q-learning's target uses \\(\max_{a'} Q(s',a')\\) — **it always assumes
the best possible next action**, regardless of whether exploration
(Chapter 2's \\(\varepsilon\\)-greedy) meant that action wasn't actually
taken. **SARSA** (State-Action-Reward-State-Action — the name itself just
lists the five pieces needed for the update) takes a different approach:
it uses the Q-value of whichever action \\(a'\\) was actually **chosen**
next:

\\[Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma Q(s',a') -
Q(s,a)\right]\\]

This one difference fundamentally splits the two algorithms. Q-learning
directly learns the value of the **target policy** (the optimal policy)
regardless of what the behavior policy is (even if it mixes in random
exploration via \\(\varepsilon\\)-greedy) — in Section 5.4's language, it's
**off-policy**. SARSA learns the value of whatever policy it's actually
following right now (\\(\varepsilon\\)-greedy, exploration included) — it's
**on-policy**.

```python
import random

def epsilon_greedy(Q, s, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return max(range(n_actions), key=lambda a: Q[s][a])

def sarsa_train(env_step, n_states, n_actions, n_episodes, alpha, gamma, epsilon, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    for _ in range(n_episodes):
        s = start_state
        a = epsilon_greedy(Q, s, epsilon, n_actions)
        for _ in range(200):
            ns, r, done = env_step(s, a)
            na = epsilon_greedy(Q, ns, epsilon, n_actions)
            Q[s][a] += alpha * (r + gamma * Q[ns][na] - Q[s][a])  # uses the actually-chosen na
            s, a = ns, na
            if done:
                break
    return Q
```

## 6.4 Cliff Walking: Two Algorithms Learn Genuinely Different Policies

In the "Cliff Walking" environment — a grid where the direct path from
start to goal runs right alongside a cliff that gives a big penalty
(-100) if you fall in — training both algorithms for 500 episodes each
produces strikingly different routes:

- **Q-learning** learns the **shortest** path, skimming right along the
  edge of the cliff — this really is optimal from the target policy's
  (greedy) perspective.
- **SARSA** learns a **safer** path, well away from the cliff — even
  during training, it still occasionally mixes in random actions via
  \\(\varepsilon\\)-greedy, and if it learned a policy that hugs the
  cliff's edge, that randomness would occasionally cause it to actually
  fall in and take a large loss. Because SARSA values things according to
  **how it actually acts** (exploration included), it prefers to avoid
  that risk.

**Q-learning finds the optimal path under the assumption "I'll always do
my best from here on," while SARSA finds a path that accounts for the
reality "I might occasionally make a mistake (explore)."** Neither is
"better" in an absolute sense — what this example shows is that the two
are simply answering different questions from the start.

## 6.5 Why Q-Learning Converges (Intuition)

As established in Section 4.5, \\(Q^*(s,a)\\) is guaranteed to exist and
be unique in any finite MDP (the Bellman optimality equation, the Banach
fixed-point theorem). Q-learning never computes that target directly; it
only approximates it from sample-based updates, and yet it's proven to
reach exactly that value: **as long as every state-action pair is visited
infinitely often and the learning rate \\(\alpha\\) is decayed
appropriately** (the Robbins-Monro conditions), it converges to the true
\\(Q^*(s,a)\\). Intuitively: the point where the TD error becomes exactly
zero is precisely the point that satisfies the Bellman optimality
equation, so an update that keeps reducing the TD error will eventually
converge to that fixed point.

**Temporal-difference learning combines Monte Carlo's advantage ("no model
needed") with dynamic programming's advantage ("update after just one
step") into a single method — and the difference between Q-learning and
SARSA comes down to a small but consequential choice: apply the same
bootstrapping idea to "the ideal target policy," or to "the policy you're
actually following."**

---

## Exercises

**1. (Coding)** Complete `q_learning_train` below (key lines left blank):

```python
import random

def epsilon_greedy(Q, s, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return max(range(n_actions), key=lambda a: Q[s][a])

def q_learning_train(transition, n_states, n_actions, n_episodes, alpha, gamma, epsilon):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    for episode in range(n_episodes):
        s = 0  # assume every episode starts at state 0
        for _ in range(20):  # max episode length
            # ADD ADDITIONAL CODE HERE!!
            # 1. choose action a via epsilon_greedy
            # 2. get reward r and next state s_next via transition(s, a)
            # 3. apply the Q-learning update rule
            # 4. if s_next is terminal (marked as -1), end the episode; else s = s_next
            if s_next == -1:
                break
            s = s_next
    return Q
```

**2. (Conceptual)** In Section 6.4's Cliff Walking example, if
\\(\varepsilon\\) is lowered to 0 after training ends (no more
exploration) and SARSA's learned policy is executed, predict whether it
would still take the long way around, or switch to the shorter path, and
explain why. (Hint: think about the fact that the Q-values SARSA learned
were themselves computed under the assumption "exploration is happening.")

**3. (Hand derivation, Tier A — free derivation)** Argue, in three parts,
what conditions are needed for the Q-learning update
\\(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)]\\) to converge to the true \\(Q^*(s,a)\\): (1) show that the
\\(Q\\) for which the TD error is exactly zero is the same as the solution
to the Bellman optimality equation \\(Q^*(s,a) = R(s,a) + \gamma
\max_{a'} Q^*(s',a')\\) (a simple algebraic argument). (2) Explain why
convergence becomes unstable if \\(\alpha\\) is too large (e.g., always
\\(\alpha=1\\)), connecting it to Chapter 2's incremental-average update.
(3) Explain why convergence requires that every state-action pair be
**visited infinitely often**, connecting it to the fact that without
ε-greedy exploration, some state-action pairs might never be tried at
all.

**Confirm correctness**: match each of the three arguments above to the
part of the Q-learning algorithm it corresponds to (TD error computation,
the `alpha` parameter, the `epsilon_greedy` function).
