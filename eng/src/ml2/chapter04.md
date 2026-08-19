# Chapter 4. Dynamic Programming

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter04_policy_evaluation.ipynb)

Chapter 3 formalized MDPs, but the goal "maximize cumulative reward" still
seems uncomputable — it looks like we'd need to peer infinitely far into
the future. Bellman's insight was that this infinite sum can be rewritten
as a **recursive relationship**. This chapter derives that recursion (the
Bellman equation) and covers how to compute an optimal policy under the
assumption that we know the MDP's transition probabilities \\(P\\)
exactly (model-based).

## 4.1 The Value Function

"If I start in this state right now and keep following this policy, how
much cumulative reward can I expect?" — the value function answers this
question with an exact number. Under policy \\(\pi\\), the value of state
\\(s\\) is the expected discounted cumulative reward starting from that
state:

\\[V^\pi(s) = \mathbb{E}\_\pi\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) \,\middle|\,
s_0 = s\right]\\]

## 4.2 The Bellman Expectation Equation: A Recursive Definition

That infinite sum can be rewritten as a recursive relationship:

\\[V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s,\pi(s)) V^\pi(s')\\]

Intuition: "the value of the current state = the reward received right
now, plus the discounted expected value of whatever states come next."
This holds because the infinite sum \\(\sum_{t=0}^\infty \gamma^t r_t =
r_0 + \gamma(r_1 + \gamma r_2 + \cdots)\\) can be regrouped as "the first
term plus a discounted remainder" — and that remainder is exactly the
definition of \\(V^\pi(s')\\). Thanks to this insight, instead of computing
everything "all the way to the end of the game" at once, we can reach the
exact answer just by repeating a computation that only looks one step
ahead.

## 4.3 Iterative Policy Evaluation

With many states, we approximate \\(V^\pi\\) by repeating the following
until the values stop changing:

```python
def policy_evaluation(P, R, policy, gamma, theta=1e-6):
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        delta = 0
        for s in range(n_states):
            a = policy[s]
            v_new = R[s][a] + gamma * sum(prob * V[next_s] for prob, next_s in P[s][a])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V
```

Every iteration updates every state's \\(V(s)\\) using the Bellman
equation's right-hand side — but the \\(V(s')\\) used on that right-hand
side is itself only an estimate so far, not yet the true value. Even so,
this iteration is provably guaranteed to converge to the exact
\\(V^\pi\\), as long as \\(\gamma < 1\\) (the **contraction mapping**
property — covered in detail in Section 4.5).

## 4.4 The Bellman Optimality Equation and Policy Iteration

So far we've learned how to evaluate a given policy. What reinforcement
learning actually wants to solve is "find the best policy." Define the
"optimal value" \\(V^*(s)\\) via the **Bellman optimality equation** — the
value you'd get at each state if you always picked the action that
maximizes the value of continuing optimally from there on:

\\[V^*(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]\\]

**Policy Iteration** alternates between two steps toward this optimum:

1. **Policy evaluation**: compute \\(V^\pi\\) for the current policy
   \\(\pi\\) (Section 4.3).
2. **Policy improvement**: at each state, if there's a better action
   according to \\(V^\pi\\), switch the policy to that action:
   \\(\pi'(s) = \arg\max_a[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')]\\).

Repeat until the policy stops changing, and you've reached the optimal
policy. **Value Iteration** merges these two steps — instead of letting
policy evaluation fully converge before improving, it does exactly one
evaluation step and immediately improves, repeating this every step — this
is the same as directly solving the Bellman optimality equation itself by
repeated substitution:

```python
def value_iteration(P, R, n_actions, gamma, theta=1e-6):
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        delta = 0
        for s in range(n_states):
            v_new = max(R[s][a] + gamma * sum(prob * V[next_s] for prob, next_s in P[s][a])
                        for a in range(n_actions))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    policy = [max(range(n_actions),
                  key=lambda a: R[s][a] + gamma * sum(prob * V[next_s] for prob, next_s in P[s][a]))
              for s in range(n_states)]
    return V, policy
```

## 4.5 Why This Iteration Converges: The Banach Fixed-Point Theorem

The fact that "at least one optimal policy always exists" isn't obvious —
policy \\(\pi\\) might win at some states while \\(\pi'\\) wins at others,
which could make the very idea of "the maximum over all policies"
collapse. This gets resolved by treating the right-hand side of 4.2's
Bellman expectation equation (or 4.4's Bellman optimality equation) as a
single **operator** \\(T\\) — \\(T\\) is proven to be a
**\\(\gamma\\)-contraction**: for any two value functions \\(V_1, V_2\\),

\\[\|TV_1 - TV_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty\\]

That is, every application of \\(T\\) shrinks the (maximum) difference
between the two functions by at least a factor of \\(\gamma\\). The
**Banach fixed-point theorem** guarantees that such a contraction always
has a **unique fixed point**, and that iterating it from anywhere
converges to that fixed point — so a \\(V^*\\) satisfying the Bellman
optimality equation exists and is unique, and Section 4.4's value
iteration (starting from any initial values and repeatedly applying
\\(T\\)) actually reaches it.

Knowing \\(V^*\\), we can build a deterministic policy \\(\pi^*\\) that,
at each state, simply picks the action that achieves that \\(\max\\).
This definition is always possible for a simple reason — the set of
available actions at each state is **finite**, and the maximum of a finite
set is always achieved somewhere.

It's also worth noting what this proof guarantees: existence, not
computability or storability. In a game like Go, where the number of
states is astronomically large (roughly \\(10^{170}\\)), \\(\pi^*\\) can
exist in theory while still being completely impossible to store as a
table over every state. That's exactly why Chapter 9 (DQN) turns to
approximating the table with a neural network instead. (Why this power
iteration converges is exactly the same mathematics as PageRank finding
the stationary distribution of a random walk over a graph.)

## 4.6 The Model-Based Assumption

Every algorithm in this chapter rests on the assumption that we know the
transition probabilities \\(P(s'|s,a)\\) **exactly** — in chess, that
would mean knowing in advance exactly how likely your opponent is to
respond to any given move, which is unknowable in most real situations.
Chapters 5-6 cover methods (Monte Carlo, TD learning) that learn purely
from experience, without knowing this model.

**Dynamic programming's core idea — "a big problem = smaller subproblems +
recursion" — is a pattern that recurs throughout computer science, and
reinforcement learning applies that pattern to the question of "how do we
compute an uncertain future reward right now?" And the fact that this
iteration always converges to the right answer is itself guaranteed by a
single mathematical property: contraction mapping.**

---

## Exercises

**1. (Coding)** For a 3-state MDP (states 0, 1, 2), complete
`policy_evaluation` and `value_iteration` below (key lines left blank):

```python
def policy_evaluation(P, R, policy, gamma, theta=1e-6):
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        # ADD ADDITIONAL CODE HERE!!
        # initialize delta -> compute v_new for each state, update delta, update V[s]
        # -> stop once delta is below theta

    return V

def value_iteration(P, R, n_actions, gamma, theta=1e-6):
    # ADD ADDITIONAL CODE HERE!!
    # similar to policy_evaluation, but take max_a instead of following a fixed policy
    # once converged, build a policy by picking the best action at each state

    return V, policy
```

**2. (Hand derivation, Tier B — hints provided)** Consider the following
3-state MDP: State 0 always moves to State 1 (reward -1), State 1 always
moves to State 2 (reward -1), State 2 is terminal (\\(V(2)=0\\)). Discount
factor \\(\gamma = 0.9\\).

Using the Bellman equation \\(V(s) = R(s) + \gamma V(s')\\), substitute
backward from State 2 (terminal) to compute \\(V(1)\\) and \\(V(0)\\)
**directly**.

**Hint**: solving backward from the terminal state (backward induction)
gives the answer through substitution alone, with no need to solve a
system of equations at once. \\(V(2) = 0\\) →
\\(V(1) = -1 + 0.9 \times V(2) = ?\\) →
\\(V(0) = -1 + 0.9 \times V(1) = ?\\)

**Confirm correctness**: compare against the value you get by running
Exercise 1's `policy_evaluation` on the same MDP.

**3. (Hand derivation, Tier C — fallback prepared)** Accept that \\(T\\)
is a \\(\gamma\\)-contraction, meaning \\(\|TV_1 - TV_2\|_\infty \le
\gamma \|V_1 - V_2\|_\infty\\), and argue why it follows that "starting
from any initial value \\(V_0\\) and repeatedly applying \\(T\\)
converges to \\(V^*\\)."

**Fill-in-the-blank fallback version** (if free-form argument is too
difficult):

```
Assume: T is a gamma-contraction, and V* is T's fixed point (TV* = V*).

Step 1: ||V_1 - V*|| = ||TV_0 - TV*|| <= gamma * ||______________||
Step 2: similarly, ||V_2 - V*|| = ||TV_1 - TV*|| <= gamma * ||______________||
                                 <= gamma^2 * ||______________||
Step 3: after n repetitions, ||V_n - V*|| <= gamma^n * ||______________||

Conclusion: since gamma < 1, as n grows gamma^n ______________ (approaches 0 / grows larger)
            so V_n ______________ (converges to V* / diverges)
```

**Confirm correctness**: explain in one sentence why this result
guarantees that policy evaluation / value iteration always reaches the
same answer no matter what initial value \\(V_0\\) you start from (e.g.,
all zeros).
