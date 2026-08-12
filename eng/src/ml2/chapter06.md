# Chapter 6. Reinforcement Learning Algorithms

In 1989, Chris Watkins proposed an algorithm called **Q-learning** in his
PhD thesis. Policy evaluation, covered last chapter, could only be computed
if we knew the environment's transition probabilities \\(P(s'|s,a)\\)
exactly — in chess, that would mean knowing in advance exactly how likely
your opponent is to respond to any given move, which is unknowable in most
real situations. Q-learning's breakthrough was showing that optimal
behavior can be learned **without knowing the environment's model at
all** — just by acting and observing the results.

## 6.1 Does an Optimal Policy Always Exist?

So far (Chapter 5) we've learned how to evaluate a given policy. But the
problem reinforcement learning actually wants to solve is "find the best
policy." The phrase "the best policy" isn't as obvious as it sounds,
though — this section looks at why, and at why one always exists anyway.

Policies are compared state by state: policy \\(\pi\\) is "better than"
policy \\(\pi'\\) when \\(V^\pi(s) \ge V^{\pi'}(s)\\) holds **at every**
state \\(s\\) simultaneously. The problem is that \\(\pi\\) might win at
some states while \\(\pi'\\) wins at others — in that case the two
policies simply can't be compared (a partial order, not a total one).
For "the maximum over all policies" to mean anything, there has to be a
single policy that's better than or equal to every other policy at once,
even when some pairs of policies can't be compared at all — and that's
far from obvious.

In fact, the standard reinforcement learning textbook — Sutton and
Barto's Reinforcement Learning: An Introduction (2018) — only states that
"there is always at least one policy that is better than or equal to all
other policies," without proving it. This section fills that gap.

## 6.2 The Bellman Optimality Equation: A Unique Answer Exists

Chapter 5.4 accepted, as a stated result, that iterating the Bellman
expectation equation always converges to the exact \\(V^\pi\\) whenever
\\(\gamma < 1\\) (the contraction mapping property). The existence of an
optimal policy is proved with exactly the same tool.

First, define the "optimal value" \\(V^\*(s)\\) via the **Bellman
optimality equation** — the value you'd get at each state if you always
picked the action that maximizes the value of continuing optimally from
there on:

\\[V^\*(s) = \max\_a \left[R(s,a) + \gamma \sum\_{s'} P(s'|s,a)
V^\*(s')\right]\\]

This equation looks like it "defines" \\(V^\*\\), but it's actually
self-referential — \\(V^\*\\) appears on both sides, exactly the same
shape as Chapter 5.4's \\(V^\pi\\) = Bellman expectation equation
(\\(V^\pi\\)). So the same question remains: does a \\(V\\) satisfying
this equation actually exist? If so, is it unique?

The same logic from 5.4 shows up again here: treat the right-hand side of
the Bellman optimality equation as an **operator** \\(T\\)
(\\(T(V)(s) := \max\_a[\ldots]\\), the bracketed part above). \\(T\\) is
also proven to be a \\(\gamma\\)-contraction (the same property as 5.4's
\\(T^\pi\\), just a different shape — we accept this as a stated result
again this semester). A \\(\gamma\\)-contraction always has a unique fixed
point, and iterating it from anywhere converges to that fixed point (the
Banach fixed-point theorem) — so a \\(V^\*\\) satisfying the Bellman
optimality equation **exists, and is unique.**

## 6.3 Building an Optimal Policy From the Value That Exists

\\(V^\*\\) existing and being unique is one thing; "there's a policy that
actually achieves that value" is another — this last gap is what we close
here.

Knowing \\(V^\*\\), we can build a deterministic policy that, at each
state, simply picks the action that achieves that \\(\max\\):

\\[\pi^\*(s) \in \arg\max\_a \left[R(s,a) + \gamma \sum\_{s'}
P(s'|s,a) V^\*(s')\right]\\]

This definition is always possible for a simple reason — the set of
available actions \\(A(s)\\) at each state is **finite**, and the maximum
of a finite set is always achieved somewhere (an argmax can't be empty).
Trace the root of "an optimal policy exists" far enough and it comes down
to this simple fact: the state and action spaces are finite.

Following this \\(\pi^\*\\) can be shown to give a value
\\(V^{\pi^\*}\\) that's exactly equal to \\(V^\*\\) (and no other policy
can do better than \\(V^\*\\)) — the full proof is beyond this semester,
but the core idea is: "\\(\pi^\*\\) was chosen to exactly achieve the
\\(\max\\) of the Bellman optimality equation at every state, so the
Bellman expectation equation under \\(\pi^\*\\) becomes identical to the
Bellman optimality equation." That means \\(\pi^\*\\) is better than or
equal to every other policy at every single state — the "incomparability"
problem from section 6.1 dissolves once there's a shared target,
\\(V^\*\\), to aim at.

One more subtlety: if the argmax has ties, multiple optimal policies can
exist — but **whichever optimal policy you pick, its value is always
exactly \\(V^\*\\).** The number itself never wavers.

It's also worth noting what this proof guarantees: existence, not
computability or storability. In a game like Go, where the number of
states is astronomically large (roughly \\(10^{170}\\)), \\(\pi^\*\\) can
exist in theory while still being completely impossible to store as a
table over every state. That's exactly why the next chapter (Chapter 7)
turns to approximating the table with a neural network instead.

**This section's conclusion is also the reason Q-learning, covered
starting next section, is worth pursuing at all: the \\(Q^\*(s,a)\\) that
Q-learning tries to estimate isn't "a target that may or may not exist" —
in any finite MDP, it's guaranteed to exist and be unique. That's what
theoretically justifies repeatedly training toward that number.**

## 6.4 Model-Based vs. Model-Free

Last chapter's policy evaluation was a "model-based" method — it computed
things under the assumption that we know how the environment works (its
transition probabilities). But in actual games or robot control, there's
usually no perfect mathematical model of "exactly what happens if I press
this button." **Q-learning** is a **model-free** method: it learns purely
from experience (state, action, reward, next state) gathered by acting
directly, with no knowledge of the model.

## 6.5 The Q-Function: Value of Both State and Action

Chapter 5's \\(V(s)\\) was "the value of a state." Q-learning learns a more
fine-grained **Q-function** \\(Q(s,a)\\) — "the expected cumulative reward
of taking action \\(a\\) in state \\(s\\), then acting optimally
thereafter." Once you know \\(Q(s,a)\\), the optimal policy is immediately
just \\(\pi^*(s) = \arg\max_a Q(s,a)\\) — knowing only "the value of a
state" doesn't tell you which action to take (you'd need the transition
probabilities for that), but knowing "the value of a state-action pair"
lets you pick the best action right away. This is the key advantage.

## 6.6 The Q-Learning Update Rule

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

## 6.7 The Exploration-Exploitation Dilemma: ε-greedy

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

## 6.8 Why Q-Learning Converges (Intuition)

As sections 6.2-6.3 established, \\(Q^\*(s,a)\\) isn't "a goal that
might or might not exist" — in a finite MDP it's guaranteed to exist and
be unique. Q-learning never computes that target directly; it only
approximates it from sample-based updates, and yet it's proven to reach
exactly that value: it converges to the true \\(Q^\*(s,a)\\), regardless of
which policy actually generated the data (the exploration policy), **as
long as every state-action pair is visited infinitely often and the
learning rate \\(\alpha\\) is decayed appropriately** (the Robbins-Monro
conditions). Intuitively: the point where the TD error becomes exactly
zero is precisely the point that satisfies the Bellman optimality equation
\\(Q^\*(s,a) = R(s,a) + \gamma \max_{a'} Q^\*(s',a')\\), so an update that
keeps reducing the TD error will eventually converge to that fixed point.

## 6.9 Q-learning vs. SARSA (For Reference)

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
