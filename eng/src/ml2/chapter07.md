# Chapter 7. n-Step Bootstrapping, Eligibility Traces & Planning

Chapter 5's Monte Carlo and Chapter 6's TD(0) look at first like two
opposite extremes — MC uses the actual reward all the way **to the end**
of the episode as its target, while TD(0) looks just **one step** ahead
and replaces the rest with an estimate. This chapter shows these are
really just the two ends of a single dial — "how many steps do we
actually look at" — and covers everything in between. We close with
planning, which leverages a learned model alongside experience.

## 7.1 n-Step TD: The Dial Between MC and TD(0)

TD(0)'s target was \\(r + \gamma V(s')\\) (1 step of actual reward, plus
an estimate for the rest). Generalize this by using \\(n\\) steps of
actually-observed rewards and replacing the rest with an estimate:

\\[G_t^{(n)} = R_t + \gamma R_{t+1} + \cdots + \gamma^{n-1} R_{t+n-1} +
\gamma^n V(s_{t+n})\\]

\\(n=1\\) is exactly TD(0), and \\(n \to \infty\\) (all the way to the end
of the episode) is exactly Monte Carlo. As \\(n\\) grows, we rely more on
actual observations — **bias decreases, variance increases** (continuing
the exact tradeoff from Section 6.1's table) — \\(n\\) is a hyperparameter
to tune somewhere in between depending on the situation.

```python
def n_step_td(env_step, n, n_episodes, alpha, gamma, n_states, start_state):
    V = [0.0] * n_states
    for _ in range(n_episodes):
        states, rewards = [start_state], []
        s, T, t = start_state, float('inf'), 0
        while True:
            if t < T:
                a = 0 if False else __import__('random').randrange(2)  # a random demo policy
                ns, r, done = env_step(s, a)
                states.append(ns)
                rewards.append(r)
                if done:
                    T = t + 1
                s = ns
            tau = t - n + 1  # we now update the tau-th state
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * rewards[i]
                        for i in range(tau, min(tau + n, len(rewards))))
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1:
                break
            t += 1
    return V
```

Notice that `tau` (the time step being updated) always lags the current
time by \\(n-1\\) — we need \\(n\\) steps' worth of actual rewards to
accumulate before we can compute that target. Training this on a 7-state
1D grid with `n=3`, the values increase smoothly the closer a state is to
the goal (right end, +1) — e.g., from around -0.84 near the left end up
to around 0.69 monotonically toward the right.

## 7.2 Eligibility Traces: All Values of n at Once

n-step TD requires fixing a single \\(n\\) each time. **Eligibility
traces** take a different approach — attach a trace \\(e(s)\\) to every
state, tracking "how often, and how recently, has this state been
visited," and whenever a TD error occurs, update **every state at once,
in proportion to its trace**:

\\[e(s) \leftarrow \gamma\lambda \, e(s) + \mathbb{1}[s = s_t], \qquad
V(s) \leftarrow V(s) + \alpha \, \delta_t \, e(s) \; \text{(for every } s \text{)}\\]

\\(\lambda \in [0,1]\\) is the new dial — with \\(\lambda=0\\), the trace
nearly vanishes every step, matching TD(0); with \\(\lambda=1\\), the
trace never shrinks, moving toward Monte Carlo. The name **TD(\\(\lambda\\))**
comes exactly from this dial — where n-step TD is a "forward" view that
fixes a single \\(n\\) and looks ahead, eligibility traces are a
"backward" view that leaves traces from past visits and propagates
updates backward; it's known that these two views are equivalent (the
proof of forward-backward equivalence is beyond this semester).

## 7.3 Planning and Learning: Dyna-Q

The model-free methods we've covered so far (MC, TD) throw away
experience after using it once. But that experience actually contains
information about the environment — roughly, "if I take this action in
this state, I tend to get about this reward and land in about this next
state." **Dyna-Q** doesn't discard this — it stores it in a simple
**model**, and interleaves that model into extra **imaginary** learning
(planning) between real interactions with the environment:

```python
import random

def dyna_q(env_step, n_states, n_actions, n_episodes, planning_steps, alpha, gamma, epsilon, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    model = {}  # (s, a) -> (r, s') -- store exactly what was observed (assumes a deterministic environment)
    for _ in range(n_episodes):
        s = start_state
        for _ in range(100):
            a = random.randrange(n_actions) if random.random() < epsilon \
                else max(range(n_actions), key=lambda x: Q[s][x])
            ns, r, done = env_step(s, a)
            Q[s][a] += alpha * (r + gamma * max(Q[ns]) - Q[s][a])  # 1) learn from real experience
            model[(s, a)] = (r, ns)                                 # 2) update the model
            for _ in range(planning_steps):                         # 3) learn imaginary steps from the model (planning)
                (ps, pa), (pr, pns) = random.choice(list(model.items()))
                Q[ps][pa] += alpha * (pr + gamma * max(Q[pns]) - Q[ps][pa])
            s = ns
            if done:
                break
    return Q
```

The larger `planning_steps` is, the more times each real interaction gets
"reviewed" through the model. On the same grid environment, comparing how
many episodes it takes for the goal state's (state 3, moving right)
Q-value to exceed 0.5, the difference is dramatic:
`planning_steps=0` (pure Q-learning) takes 253 episodes, while
`planning_steps=10` reaches it in just 9 episodes — with 10 extra rounds
of learning from the model at every real step, far more gets squeezed out
of the same real experience.

## 7.4 Three Axes, In One Place

Putting together everything covered in this chapter shows that Block A of
this semester has really been a series of different answers to a single
question: "how much bootstrapping should we do?"

| Method | Target | Characteristics |
|---|---|---|
| MC (Chapter 5) | actual return \\(G_t\\) | unbiased, high variance, needs the episode to end |
| TD(0) (Chapter 6) | \\(r + \gamma V(s')\\) | biased, low variance, learns every step |
| n-step TD | \\(n\\) steps actual + rest estimated | a compromise between the two, \\(n\\) is the dial |
| TD(\\(\lambda\\)) | all values of \\(n\\), weighted by traces | no need to fix \\(n\\), \\(\lambda\\) is the dial |
| Dyna-Q | TD(0) + planning with a learned model | combines model-free and model-based |

**Not bootstrapping at all (MC) and bootstrapping every step (TD) are just
two extremes — in practice, something in between, plus reusing experience
through planning, is usually better. That's this chapter's conclusion.**

---

## Exercises

**1. (Coding)** Complete `dyna_q` above (key lines left blank), and
compare how many episodes it takes for the same state-action's Q-value to
exceed 0.5, with `planning_steps` set to 0 versus 10:

```python
import random

def dyna_q(env_step, n_states, n_actions, n_episodes, planning_steps, alpha, gamma, epsilon, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    model = {}
    # ADD ADDITIONAL CODE HERE!!
    # 1) choose an action via epsilon-greedy, interact with the environment, apply the Q-learning update
    # 2) store model[(s,a)] = (r, ns)
    # 3) for planning_steps iterations, sample a random (s,a)->(r,ns) from the model and apply the same update

    return Q
```

**2. (Conceptual)** Dyna-Q's model assumes the environment is
deterministic (the same (s,a) always gives the same (r,s')). If the
environment were stochastic (the same (s,a) could give different s'
each time), explain what problem arises from storing only the last
experience, as `model[(s,a)] = (r, ns)` does, and propose one way to
improve this.

**3. (Hand derivation, Tier B — hints provided)** For the \\(n\\)-step
return \\(G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k} + \gamma^n
V(s_{t+n})\\), confirm that setting \\(n=1\\) gives exactly TD(0)'s target
\\(R_t + \gamma V(s_{t+1})\\), and show that as \\(n \to \infty\\) with an
episode ending at a finite length \\(T\\) (so \\(V(s_T) = 0\\)),
\\(G_t^{(n)}\\) becomes equal to Monte Carlo's actual return \\(G_t =
\sum_{k=0}^{T-t-1} \gamma^k R_{t+k}\\).

**Confirm correctness**: explain in one sentence why this result supports
the claim that "TD(0) and MC are the two special cases of n-step TD."
