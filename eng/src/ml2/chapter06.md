# Chapter 6. Reinforcement Learning Algorithms

In 1989, Chris Watkins proposed an algorithm called **Q-learning** in his
PhD thesis. Policy evaluation, covered last chapter, could only be computed
if we knew the environment's transition probabilities \\(P(s'|s,a)\\)
exactly — in chess, that would mean knowing in advance exactly how likely
your opponent is to respond to any given move, which is unknowable in most
real situations. Q-learning's breakthrough was showing that optimal
behavior can be learned **without knowing the environment's model at
all** — just by acting and observing the results.

## 6.1 Model-Based vs. Model-Free

Last chapter's policy evaluation was a "model-based" method — it computed
things under the assumption that we know how the environment works (its
transition probabilities). But in actual games or robot control, there's
usually no perfect mathematical model of "exactly what happens if I press
this button." **Q-learning** is a **model-free** method: it learns purely
from experience (state, action, reward, next state) gathered by acting
directly, with no knowledge of the model.

## 6.2 The Q-Function: Value of Both State and Action

Chapter 5's \\(V(s)\\) was "the value of a state." Q-learning learns a more
fine-grained **Q-function** \\(Q(s,a)\\) — "the expected cumulative reward
of taking action \\(a\\) in state \\(s\\), then acting optimally
thereafter." Once you know \\(Q(s,a)\\), the optimal policy is immediately
just \\(\pi^*(s) = \arg\max_a Q(s,a)\\) — knowing only "the value of a
state" doesn't tell you which action to take (you'd need the transition
probabilities for that), but knowing "the value of a state-action pair"
lets you pick the best action right away. This is the key advantage.

## 6.3 The Q-Learning Update Rule

Every time the agent takes action \\(a\\) in state \\(s\\), receives
reward \\(r\\), and arrives at new state \\(s'\\):

\\[Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)\right]\\]

The bracketed term is called the **TD error** (Temporal Difference error):
the difference between "our current estimate of \\(Q(s,a)\\)" and "the
reward just observed, plus our estimate of doing our best from the next
state onward." We nudge \\(Q(s,a)\\) by this error each time — the same
pattern as Chapter 2's gradient descent: "move by the difference between
the current estimate and a better one."

```python
def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    max_next_q = max(Q[s_next].values())
    td_error = r + gamma * max_next_q - Q[s][a]
    Q[s][a] += alpha * td_error
    return Q
```

## 6.4 The Exploration-Exploitation Dilemma: ε-greedy

Think about choosing a restaurant: do you go back to a favorite you already
know is good (**exploitation**), or try somewhere new
(**exploration**)? A place you haven't tried might actually be better, but
there's also a risk of disappointment. Reinforcement learning agents face
the exact same dilemma — always taking the action currently believed best
(pure exploitation) might mean never discovering a better one, while acting
purely randomly (pure exploration) wastes everything already learned.

When choosing an action, with probability \\(1-\varepsilon\\) pick the
currently best-known action (exploit), and with probability
\\(\varepsilon\\) pick a random action (explore):

```python
import random

def epsilon_greedy(Q, s, epsilon, actions):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: Q[s][a])
```

A common strategy is to gradually decay \\(\varepsilon\\) as training
progresses — explore a lot early on, then shift increasingly toward
exploitation as the Q-values become more trustworthy.

## 6.5 Why Q-Learning Converges (Intuition)

Q-learning is proven to converge to the true \\(Q^\*(s,a)\\), regardless of
which policy actually generated the data (the exploration policy), **as
long as every state-action pair is visited infinitely often and the
learning rate \\(\alpha\\) is decayed appropriately** (the Robbins-Monro
conditions). Intuitively: the point where the TD error becomes exactly
zero is precisely the point that satisfies the Bellman optimality equation
\\(Q^\*(s,a) = R(s,a) + \gamma \max_{a'} Q^\*(s',a')\\), so an update that
keeps reducing the TD error will eventually converge to that fixed point.

## 6.6 Q-learning vs. SARSA (For Reference)

Q-learning's update always uses \\(\max_{a'} Q(s',a')\\) — **always
assuming the best possible next action**, even if exploration meant the
agent actually took a different one. This approach is called off-policy.
(SARSA is an on-policy method that instead uses the Q-value of whichever
action was actually taken next — not covered this semester, but useful
context for understanding Q-learning's design choice.)

**By proving that an agent can learn without knowing the environment,
Q-learning was the turning point that transformed reinforcement learning
from a theoretical curiosity into a tool actually applicable to robotics,
games, and recommendation systems.**

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

**2. (Hand derivation, Tier A — free derivation)** Argue, in three parts,
what conditions are needed for the Q-learning update
\\(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)]\\) to converge to the true \\(Q^\*(s,a)\\): (1) show that the
\\(Q\\) for which the TD error is exactly zero is the same as the solution
to the Bellman optimality equation \\(Q^\*(s,a) = R(s,a) + \gamma
\max_{a'} Q^\*(s',a')\\) (a simple algebraic argument). (2) Explain why
convergence becomes unstable if \\(\alpha\\) is too large (e.g., always
\\(\alpha=1\\)), connecting it to Chapter 2's learning-rate problem. (3)
Explain why convergence requires that every state-action pair be
**visited infinitely often**, connecting it to the fact that without
ε-greedy exploration, some state-action pairs might never be tried at all.

**Confirm correctness**: match each of the three arguments above to the
part of the Q-learning algorithm it corresponds to (TD error computation,
the `alpha` parameter, the `epsilon_greedy` function).
