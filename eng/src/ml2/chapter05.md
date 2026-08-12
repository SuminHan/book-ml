# Chapter 5. Reinforcement Learning Basics & Policy Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter05_policy_evaluation.ipynb)

In ML1, we learned the term "curse of dimensionality." That term was coined
by mathematician Richard Bellman, who named it while developing an
optimization technique called dynamic programming in the 1950s. Another
tool he built is the centerpiece of this chapter: the **Bellman equation**.
His idea — "don't solve a complex problem all at once; break it into
smaller subproblems and combine their answers recursively" — forms the
theoretical backbone of reinforcement learning today.

## 5.1 Formalizing What ML1 Only Previewed

ML1 only gave you an intuitive look at states, actions, rewards, and
policies; this chapter formalizes them into the mathematical framework of
an **MDP** (Markov Decision Process). An MDP is defined by five components:
\\((\mathcal{S}, \mathcal{A}, P, R, \gamma)\\)

- \\(\mathcal{S}\\): the set of possible states
- \\(\mathcal{A}\\): the set of possible actions
- \\(P(s'|s,a)\\): the probability of transitioning to state \\(s'\\)
  after taking action \\(a\\) in state \\(s\\)
- \\(R(s,a)\\): the immediate reward received for taking action \\(a\\) in
  state \\(s\\)
- \\(\gamma \in [0,1)\\): the discount factor

**The Markov property**: the next state depends only on the **current**
state and action — how you got there (the entire past history) doesn't
matter. For example, knowing only a chess board's current arrangement is
enough to decide the next move; the sequence of moves that led there is
irrelevant.

## 5.2 The Value Function

"If I start in this state right now and keep following this policy, how
much cumulative reward can I expect?" — the value function answers this
question with an exact number. Under policy \\(\pi\\), the value of state
\\(s\\) is the **expected discounted cumulative reward** starting from that
state:

\\[V^\pi(s) = \mathbb{E}\_\pi\left[\sum\_{t=0}^\infty \gamma^t R(s\_t, a\_t) \\,\middle|\\,
s\_0 = s\right]\\]

## 5.3 The Bellman Equation: A Recursive Definition

Computing the value function directly seems to require accounting for
every possible thing that could happen from now until the game ends, but
the Bellman equation offers a much cleverer shortcut. That infinite sum can
be rewritten as a recursive relationship:

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

## 5.4 Iterative Policy Evaluation

For a deterministic policy (always choosing the same action in a given
state), the Bellman equation can be solved directly as a **system of
equations** (when there are few states), but with many states, it's instead
approximated by repeating the following:

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
\\(V^\pi\\), as long as \\(\gamma < 1\\) (based on the contraction mapping
property — we'll just accept the result this semester without proving it).

## 5.5 Working Backward From a Terminal State: A Special Case

If the MDP has a path-like structure with a terminal state
(\\(V=0\\)), you can get the exact answer immediately just by substituting
backward from the terminal state, with no iteration needed — the case
covered in this chapter's exercises. This can actually be viewed as a
special case of iterative policy evaluation, one that starts from "a state
whose answer is already known" (the terminal state) and propagates in a
single direction.

**Dynamic programming's core idea — "a big problem = smaller subproblems +
recursion" — is a pattern that recurs throughout computer science, and
reinforcement learning applies that pattern to the question of "how do we
compute an uncertain future reward right now?"**

---

## Exercises

**1. (Coding)** For a 3-state MDP (states 0, 1, 2) under a fixed policy,
complete `policy_evaluation`, which computes each state's value function
\\(V(s)\\) via iterative policy evaluation (key lines left blank):

```python
def policy_evaluation(P, R, policy, gamma, theta=1e-6):
    # P[s][a] = [(prob, next_state), ...]  transition probabilities
    # R[s][a] = immediate reward (scalar), policy[s] = the deterministic policy's action
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        # ADD ADDITIONAL CODE HERE!!
        # initialize delta -> compute v_new for each state, update delta, update V[s]
        # -> stop once delta is below theta

    return V
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
Exercise 1's code on the same MDP.
