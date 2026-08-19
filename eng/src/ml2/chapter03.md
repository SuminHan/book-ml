# Chapter 3. Markov Decision Processes

In the 1950s, mathematician Richard Bellman coined the term "curse of
dimensionality" while developing an optimization technique called dynamic
programming. His core insight — "don't solve a complex problem all at
once; break it into smaller subproblems and combine their answers
recursively" — forms the theoretical backbone of reinforcement learning
today. This chapter adds "state" to Chapter 2's bandits, building the
framework that lets us define a reinforcement learning problem with
mathematical rigor: the **MDP** (Markov Decision Process).

## 3.1 The Five Components of an MDP

An MDP is defined by five components: \\((\mathcal{S}, \mathcal{A}, P, R,
\gamma)\\)

- \\(\mathcal{S}\\): the set of possible states
- \\(\mathcal{A}\\): the set of possible actions
- \\(P(s'|s,a)\\): the probability of transitioning to state \\(s'\\)
  after taking action \\(a\\) in state \\(s\\)
- \\(R(s,a)\\): the immediate reward received for taking action \\(a\\) in
  state \\(s\\)
- \\(\gamma \in [0,1)\\): the discount factor

Comparing this to Chapter 2's bandits shows exactly what's been added —
bandits had neither \\(\mathcal{S}\\) nor \\(P\\) (pulling an arm didn't
lead to a "next state"). An MDP explicitly captures the fact that the
current action affects the entire future through the next-state transition
\\(P(s'|s,a)\\).

## 3.2 The Markov Property

**The Markov property**: the next state depends only on the **current**
state and action — how you got there (the entire past history) doesn't
matter:

\\[P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0) = P(s_{t+1} | s_t, a_t)\\]

For example, knowing only a chess board's current arrangement is enough to
decide the next move; the sequence of moves that led there is irrelevant.
Thanks to this assumption, remembering only "the current state" (rather
than "the entire history so far") is enough — this is exactly why the
algorithms we'll cover can make decisions based on a single state.

## 3.3 Cumulative Reward and the Discount Factor

Reinforcement learning's goal isn't a single step's reward, but the sum of
all future rewards — the **return** \\(G_t\\):

\\[G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k}\\]

**Why do we need the discount factor \\(\gamma\\)?**: (1) Mathematically,
as long as \\(\gamma < 1\\), this infinite series always converges when
rewards are bounded (a geometric series). If \\(\gamma=1\\), the return
can diverge to infinity in a never-ending task. (2) Intuitively, "a reward
of 1 right now" is often considered more valuable than "a reward of 1, 10
steps from now" — this points the same direction as human time preference
or the concept of interest rates in finance. An agent with \\(\gamma\\)
close to 0 is "short-sighted" (only cares about immediate reward), while
one close to 1 is "long-sighted" (properly weighs even distant future
rewards).

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)

# Let's actually confirm the MDP's five components in this environment
print("State space S:", env.observation_space)   # a 4-dimensional continuous vector
print("Action space A:", env.action_space)         # 2 discrete actions (push left/right)

total_return, gamma, discount = 0.0, 0.95, 1.0
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_return += discount * reward   # directly accumulate G_t
    discount *= gamma
    if terminated or truncated:
        break
print("Discounted return G_0 over 10 steps:", round(total_return, 3))
env.close()
```

In this code, `reward` is the immediate reward \\(R(s,a)\\) at each step,
and the rule the environment uses internally to transition to the next
state is exactly \\(P(s'|s,a)\\) — Gymnasium environments simply hide this
transition probability behind the `step()` function, but it corresponds
exactly to the MDP's definition.

## 3.4 Toward the Value Function

The goal "maximize cumulative reward" is now well-defined, but we still
have no way to compute it — we seem to need to look infinitely far into
the future. The next chapter (dynamic programming) rewrites this infinite
sum as a recursive form — the **Bellman equation** — turning it into an
actually computable procedure.

**An MDP is just Chapter 2's bandit problem with one fact added: the
current action changes the state itself in the future. But that one
difference makes reinforcement learning a far harder, and far more
interesting, problem.**

---

## Exercises

**1. (Coding)** Using the code above as a reference, create a
`FrozenLake-v1` (`is_slippery=False`) environment and print
`env.observation_space` and `env.action_space` to check whether the state
and action spaces are each discrete or continuous, and how many values
each has. Describe in one sentence how this compares to CartPole.

```python
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
# ADD ADDITIONAL CODE HERE!!
# print observation_space, action_space, and compare with CartPole
```

**2. (Conceptual)** Think of a situation where the Markov property seems
like it would actually fail to hold (e.g., in self-driving, deciding based
on "just the current camera frame" alone — information from a sign passed
a moment ago in fog might still be needed). Propose, in one paragraph, how
you'd redefine the state to restore the Markov property in that case (e.g.,
including the last several frames in the state).

**3. (Hand derivation, Tier B — hints provided)** For a constant reward
\\(R\\) at every step (i.e., \\(R_t = R\\) for all \\(t\\)), show that the
return \\(G_t = \sum_{k=0}^\infty \gamma^k R\\) converges to the
closed-form \\(G_t = \frac{R}{1-\gamma}\\) when \\(\gamma < 1\\).

**Hint**: this follows directly from the geometric series formula
\\(\sum_{k=0}^\infty x^k = \frac{1}{1-x}\\) (for \\(|x|<1\\)). Compute
\\(G_t\\) directly for \\(R=1, \gamma=0.9\\), then compute it again for
\\(\gamma=0.99\\) and see how the value changes.

**Confirm correctness**: based on your calculations, explain in one
sentence why \\(G_t\\) grows larger (approaching divergence) as
\\(\gamma\\) gets closer to 1.
