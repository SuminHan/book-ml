# Chapter 2. Multi-Armed Bandits

Imagine a casino with several slot machines, each with its own lever
(arm). Each arm pays out a different, unknown-to-you average reward. If
you can only pull the levers a limited number of times, in what order
should you pull them to maximize your total reward? This simple question —
the **multi-armed bandit** problem — is the smallest possible skeleton of
every problem reinforcement learning deals with. It has no state, no
transitions spanning multiple steps like the chapters ahead will introduce
— it isolates, in its purest form, the single core problem of
reinforcement learning: the exploration-exploitation dilemma.

## 2.1 Exploration and Exploitation

Let \\(q^*(a)\\) be the true average reward of each of \\(k\\) arms (which
we don't know). Let \\(Q_t(a)\\) be our estimate based on what we've
observed so far. The simplest strategy is **greedy** selection — always
pull the arm with the largest \\(Q_t(a)\\). The problem is that an arm
that happened to produce a bad result early on might get permanently
ignored afterward — getting stuck on a wrong conclusion without ever
sufficiently trying the arm that's actually best.

**\\(\varepsilon\\)-greedy** pulls the currently best-known arm with
probability \\(1-\varepsilon\\) (exploit), and a random arm with
probability \\(\varepsilon\\) (explore) — the exact same strategy we'll
meet again in Chapter 6 (TD learning). Each arm's estimate is updated as
the average of the rewards actually observed:

\\[Q_{t+1}(a) = Q_t(a) + \frac{1}{N_t(a)}\big(R_t - Q_t(a)\big)\\]

\\(N_t(a)\\) is the number of times arm \\(a\\) has been pulled so far.
This update rule has the shape "move the estimate by an amount
proportional to the difference between the newly observed value and the
current estimate" — a pattern we'll keep reusing in Chapters 5 (Monte
Carlo) and 6 (TD learning).

```python
import random

def epsilon_greedy_bandit(true_means, epsilon, steps):
    k = len(true_means)
    Q = [0.0] * k
    N = [0] * k
    for t in range(steps):
        if random.random() < epsilon:
            a = random.randrange(k)          # explore
        else:
            a = max(range(k), key=lambda i: Q[i])  # exploit
        r = random.gauss(true_means[a], 1.0)  # true reward has mean true_means[a], plus noise
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
    return Q
```

## 2.2 Optimistic Initialization

\\(\varepsilon\\)-greedy explores completely randomly — it re-tries even
arms it's already fairly confident are bad, with the same probability as
any other. **Optimistic initialization** uses a different trick: initialize
every \\(Q_0(a)\\) much higher (more optimistically) than any actually
achievable reward. Then greedy selection alone naturally tries unexplored
arms first, since they're still "overestimated" — try one, get
disappointed (the actual reward is lower than the initial value), and the
estimate drops, moving on to the next arm. A handful of forced early
exploration steps get generated purely from one optimistic starting value.

## 2.3 UCB: Using Uncertainty Itself

**UCB** (Upper Confidence Bound) goes a step further — it directly
exploits the fact that "the less an arm has been pulled, the less certain
we can be about how accurate its estimate is." When choosing an arm, it
adds an "uncertainty bonus" to the mean estimate before comparing:

\\[a_t = \arg\max_a \left[Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]\\]

The smaller \\(N_t(a)\\) is (the less that arm has been pulled), the
larger the square-root term grows, boosting the bonus — giving fair
credit to the possibility that this arm might actually be the best one.
The more it gets pulled (the larger \\(N_t(a)\\) grows), the smaller the
bonus shrinks, until the true magnitude of \\(Q_t(a)\\) eventually
dominates the choice.

```python
import math

def ucb_bandit(true_means, c, steps):
    k = len(true_means)
    Q = [0.0] * k
    N = [0] * k
    for t in range(1, steps + 1):
        unplayed = [i for i in range(k) if N[i] == 0]
        if unplayed:
            a = unplayed[0]  # try any unpulled arm first
        else:
            a = max(range(k), key=lambda i: Q[i] + c * math.sqrt(math.log(t) / N[i]))
        r = random.gauss(true_means[a], 1.0)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
    return Q
```

Pulling three arms (true mean rewards 1.0, 1.5, 2.0) 2000 times each,
both \\(\varepsilon\\)-greedy and UCB converge to estimates close to the
true means (e.g., `[0.98, 1.32, 2.01]` and `[1.15, 1.48, 2.0]`), but UCB
tends to earn slightly more total reward — because instead of exploring
randomly, it preferentially probes wherever uncertainty is largest.

## 2.4 Bandits vs. a Full MDP: A Bridge to the Next Chapter

The bandit problem has no **state** — pulling an arm doesn't move you to a
"next state." It's a world where nothing changes over time; you always
choose among the same \\(k\\) arms. The **MDP** (Markov Decision Process)
we'll cover starting in Chapter 3 adds one decisive thing on top of this:
**the action you take now affects what state you'll see next.** Think
about the difference between pulling a slot machine's lever and making a
move in chess: in chess, the move you make now changes the board
(the state), and that board determines what you can do next. The
exploration-exploitation dilemma we learned from bandits carries over
unchanged, but now it's layered with a completely new problem — "the
current choice changes the choices themselves that will be available in
the future."

**Bandits are the minimal skeleton of reinforcement learning with state
and transitions stripped away — \\(\varepsilon\\)-greedy, optimistic
initialization, and UCB, the three exploration strategies covered here,
reappear (in different shapes, especially ε-greedy in Chapter 6) again
and again throughout the semester.**

---

## Exercises

**1. (Coding)** Complete `epsilon_greedy_bandit` and `ucb_bandit` below
(key lines left blank):

```python
import random, math

def epsilon_greedy_bandit(true_means, epsilon, steps):
    # ADD ADDITIONAL CODE HERE!!
    # initialize Q, N for k arms to 0, then repeat steps times:
    # with probability epsilon pick a random arm, otherwise the arm with max Q
    # after observing the reward, update Q[a] += (r - Q[a]) / N[a]

    return Q

def ucb_bandit(true_means, c, steps):
    # ADD ADDITIONAL CODE HERE!!
    # first try every unpulled arm once, then
    # pick the arm that maximizes Q[a] + c*sqrt(ln(t)/N[a])

    return Q

random.seed(0)
true_means = [1.0, 1.5, 2.0]
print(epsilon_greedy_bandit(true_means, 0.1, 2000))  # roughly [1.0, 1.3~1.5, 2.0]
```

**2. (Conceptual)** Explain what problem arises if you fix
\\(\varepsilon\\) at 0 (pure greedy selection), and what problem arises if
you fix \\(\varepsilon=1\\) (pure random selection). Then discuss why
gradually decaying \\(\varepsilon\\) as training progresses is a
reasonable compromise between the two.

**3. (Hand derivation, Tier B — hints provided)** For an arm pulled
\\(n\\) times with rewards \\(R_1, \ldots, R_n\\), show that expanding the
incremental update \\(Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)\\)
recursively starting from \\(n=1\\) gives exactly \\(Q_{n+1} =
\frac{1}{n}\sum_{i=1}^n R_i\\) (the simple average).

**Hint**: start from \\(Q_1 = 0\\), expand directly for \\(n=1, 2, 3\\)
and look for the pattern (e.g., \\(Q_2 = R_1\\), \\(Q_3 = R_1 +
\frac{1}{2}(R_2-R_1) = \frac{R_1+R_2}{2}\\)), then prove the general case
for \\(n\\) by induction.

**Confirm correctness**: explain in one sentence why this result means the
incremental update formula is mathematically identical to computing a
plain average online (one data point at a time).
