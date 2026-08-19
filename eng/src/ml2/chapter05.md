# Chapter 5. Monte Carlo Methods

Chapter 4's dynamic programming was powerful, but it rested on one fatal
assumption — knowing the transition probabilities \\(P(s'|s,a)\\)
exactly. Think about a card game: nobody walks around with a "probability
table" of what card the dealer will play next. And yet, people get better
at strategy just by playing the game over and over. **Monte Carlo** (MC)
methods translate exactly this approach into an algorithm — with no
knowledge of the model at all, **actually play episodes all the way to
the end, and use the results (the returns actually received) directly as
value estimates.**

## 5.1 Learning From Experience Without a Model

Recall the definition of the value function: \\(V^\pi(s) =
\mathbb{E}\_\pi[G_t \mid s_t = s]\\) — "the **expected value** of the
return, starting from this state and following the policy." Computing an
expectation exactly requires knowing the transition probabilities, but an
expectation can also be approximated by **averaging many samples** — this
is the basic principle of statistics (the law of large numbers). MC uses
exactly this principle: play many episodes under policy \\(\pi\\), and
average the returns actually received every time state \\(s\\) is
visited — that average becomes the estimate of \\(V^\pi(s)\\).

## 5.2 First-Visit MC Prediction

A state can be visited more than once within a single episode (e.g.,
returning to the same position in a game). **First-visit** MC counts only
the return from the **first** time that state is visited in each episode
(there's also **every-visit** MC, which uses every visit — this semester
covers only the first-visit version):

```python
def mc_prediction(policy, env_sample_episode, n_episodes, gamma):
    # env_sample_episode(policy) -> [(state, action, reward), ...] one episode
    returns_sum = {}
    returns_count = {}
    for _ in range(n_episodes):
        episode = env_sample_episode(policy)
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if s not in visited:  # count only the first visit
                visited.add(s)
                returns_sum[s] = returns_sum.get(s, 0.0) + G
                returns_count[s] = returns_count.get(s, 0) + 1
    return {s: returns_sum[s] / returns_count[s] for s in returns_sum}
```

Why we sweep the episode **from the end backward**: since the return
\\(G_t = R_t + \gamma G_{t+1}\\) is defined recursively, accumulating
backward from the very last step lets us compute every step's \\(G_t\\) in
a single pass — the same "substitute backward from the terminal state"
pattern we saw in Chapter 4.

## 5.3 MC Control: All the Way to Policy Improvement

To go from prediction (evaluating a given policy) to control (finding a
better one), we use the same framework as Chapter 4's policy iteration —
except now we estimate **Q(s,a)** instead of \\(V(s)\\) (because without
knowing the transition probabilities, \\(V\\) alone can't tell you "which
action is best" — a reason we already emphasized in Section 5.3 above).
We repeat greedy policy improvement (picking the action with the highest
Q-value at each state) together with \\(\varepsilon\\)-greedy (the same
strategy from Chapter 2) — pure greedy behavior would mean any
(state, action) pair never tried even once stays forever unknown.

```python
import random

def mc_control(env_step, n_states, n_actions, n_episodes, epsilon, gamma, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    counts = [[0] * n_actions for _ in range(n_states)]

    def generate_episode():
        s, episode = start_state, []
        for _ in range(100):
            a = random.randrange(n_actions) if random.random() < epsilon \
                else max(range(n_actions), key=lambda x: Q[s][x])
            ns, r, done = env_step(s, a)
            episode.append((s, a, r))
            s = ns
            if done:
                break
        return episode

    for _ in range(n_episodes):
        episode = generate_episode()
        G, visited = 0.0, set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                counts[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / counts[s][a]  # the same incremental average as Chapter 2
    return Q
```

Training this for 5000 episodes on a small 1D grid environment (with
terminal rewards -1 and +1 at either end), the closer a state is to the
goal (+1), the larger the Q-value grows for the action that moves toward
it — for example, at the state right before the goal, the Q-value for the
goal-directed action converges to nearly 1.0.

## 5.4 Importance Sampling: A Bridge to Off-Policy Learning

So far, MC used the same policy both "to act" and "to evaluate"
(**on-policy**) — we acted with \\(\varepsilon\\)-greedy and learned the
value of that very same \\(\varepsilon\\)-greedy policy. But what if we
want to learn "the value of the greedy policy" from "data collected by
acting randomly" (**off-policy**)? You can't just average the observed
returns directly — if the policy used to collect data (the **behavior
policy** \\(b\\)) differs from the policy we want to evaluate (the
**target policy** \\(\pi\\)), naively averaging the observed returns gives
a biased estimate.

**Importance sampling** corrects this bias by multiplying each episode's
return by a weight — "the probability this trajectory came from \\(\pi\\),
relative to the probability it came from \\(b\\)":

\\[\rho = \prod_{t} \frac{\pi(a_t|s_t)}{b(a_t|s_t)}\\]

Intuition: a trajectory obtained because behavior policy \\(b\\) happened
to pick an action the target policy \\(\pi\\) would almost never pick gets
down-weighted in proportion to how rare that choice is (a small
\\(\rho\\)). This idea reappears, in different shapes, in Chapter 9's
experience replay buffer (reusing data collected by past policies) and
Chapter 11's PPO probability ratio.

## 5.5 MC's Limitations and the Next Chapter

MC has the powerful advantage of needing no model, but it has a
fundamental constraint: it can only learn **once an episode ends** —
because the return \\(G_t\\) itself is the sum of rewards all the way to
the end of the episode. If a game is very long, or a task never ends
(a continuing task), MC simply can't be used at all. Chapter 6's
temporal-difference learning solves this with a completely different
compromise: "update immediately after just one step, without waiting for
the episode to end."

**Monte Carlo carries over the most basic principle of statistics —
"with no model, if you try enough times, the average converges to the
truth" — directly into reinforcement learning. The cost is that you
always have to wait for the game to end before you find out the result.**

---

## Exercises

**1. (Coding)** For the following 1D grid environment (`step` function,
states 0-4, action 0=left/1=right, states 0 and 4 are terminal), complete
`mc_control` above (key lines left blank):

```python
import random

def step(s, a):
    ns = s - 1 if a == 0 else s + 1
    ns = max(0, min(4, ns))
    if ns == 0:
        return ns, -1.0, True
    if ns == 4:
        return ns, 1.0, True
    return ns, 0.0, False

def mc_control(env_step, n_states, n_actions, n_episodes, epsilon, gamma, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    counts = [[0] * n_actions for _ in range(n_states)]
    # ADD ADDITIONAL CODE HERE!!
    # define a generate_episode inner function (epsilon-greedy action choice, env_step for transitions)
    # after generating an episode, compute first-visit returns backward, incrementally update Q

    return Q

random.seed(1)
Q = mc_control(step, 5, 2, 5000, 0.1, 0.9, start_state=2)
for s in range(5):
    print(s, [round(v, 2) for v in Q[s]])
# state 3's action 1 (right, toward the goal) should have the largest Q-value
```

**2. (Conceptual)** First-visit MC and every-visit MC can produce
different estimates from the same data. Give a concrete example of a
situation where the same state is visited multiple times within one
episode, and explain why the two methods' estimates can differ.

**3. (Hand derivation, Tier B — hints provided)** Suppose behavior policy
\\(b\\) picks each of two actions with 50% probability, and target policy
\\(\pi\\) is a deterministic policy that always picks action 0. For a
length-2 trajectory \\((s_0,a_0{=}0),(s_1,a_1{=}0)\\), compute the
importance sampling ratio \\(\rho = \prod_t
\frac{\pi(a_t|s_t)}{b(a_t|s_t)}\\). Then compute what \\(\rho\\) becomes
if the trajectory's second action had instead been \\(a_1{=}1\\) (an
action \\(\pi\\) would never pick), and explain in one sentence why that
result means "this trajectory contributes nothing to the target policy's
value estimate."
