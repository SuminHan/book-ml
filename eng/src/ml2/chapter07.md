# Chapter 7. Deep Reinforcement Learning

In 2013, DeepMind trained an algorithm on several Atari 2600 games by
replacing Q-learning's Q-table entirely with a neural network — without
ever telling it the rules of the game, using only screen pixels and score.
This result, called **DQN** (Deep Q-Network), was published in Nature in
2015 and achieved human-level or better performance on several games.

## 7.1 A Scale the Q-Table Can't Handle

Chapter 6's Q-learning stored every state-action pair's value in a
**table**, `Q[s][a]`. This works fine when there are a few hundred states,
but the number of possible states an Atari screen (hundreds of pixels wide
and tall, with color combinations) can produce is effectively infinite —
it can't possibly all fit in a table, and most states never appear even
once during training.

## 7.2 Approximating the Q-Function With a Neural Network

DQN's idea is simple: approximate \\(Q(s,a)\\) with a **neural network**
instead of a table (\\(Q(s,a;\theta)\\), where \\(\theta\\) is the
network's weights). Similar screens (states) will produce similar Q-values
through the network, so it can generalize to states it has never seen
exactly before. The loss function is Chapter 6's TD error, squared:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta) -
Q(s,a;\theta)\right)^2\right]\\]

We update \\(\theta\\) via gradient descent (specifically, backpropagation)
to minimize this loss — the same shape as supervised learning, but with the
difference that the target value that plays the role of "the correct
answer" (\\(r + \gamma \max_{a'} Q(s',a';\theta)\\)) is itself computed
using the very \\(\theta\\) currently being trained.

## 7.3 Problem 1: The Target Keeps Moving

If \\(\theta\\) is updated every step, the loss function's target value
changes every step too — in supervised learning, the correct answer
\\(y\\) never changes, but here the "correct answer" is effectively
chasing itself. This makes training unstable (oscillation, divergence).

**Solution: the Target Network.** Keep a second network,
\\(Q(s,a;\theta^-)\\), identical in structure to \\(Q\\), and use **only
this target network** to compute the target value:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) -
Q(s,a;\theta)\right)^2\right]\\]

\\(\theta^-\\) isn't updated every step — instead, its value is copied
wholesale from \\(\theta\\)'s current value every fixed interval (e.g.,
every 1000 steps). During that interval, the target stays fixed, so the
loss function becomes (over that short window) exactly the same kind of
optimization problem as supervised learning, with a fixed correct
answer — effectively pinning down the moving target.

```python
def dqn_loss(Q_net, target_net, s, a, r, s_next, gamma, actions):
    target = r + gamma * max(target_net(s_next, a2) for a2 in actions)
    prediction = Q_net(s, a)
    return (target - prediction) ** 2
```

## 7.4 Problem 2: The Data Is Correlated

Consecutive frames (states) are nearly identical, so training on them in
order produces a model that overfits to the last handful of similar
experiences — and neural network training generally assumes mini-batches
that are independent and diverse to be stable.

**Solution: Experience Replay.** Instead of immediately using each
experienced \\((s, a, r, s')\\) for training, store it in a large storage
called a **replay buffer**. During training, sample a mini-batch
**randomly** from this buffer — this avoids batches consisting only of
temporally adjacent experiences, and lets old experiences be reused,
improving data efficiency too.

```python
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

## 7.5 What These Two Devices Have in Common

Experience replay and the target network solve different problems (data
correlation vs. a moving target), but they share the same underlying
philosophy: **"artificially recreate the conditions under which supervised
learning works well (independent data, a fixed correct answer) inside
reinforcement learning, an environment where those conditions are broken
by default."**

**The naive expectation that "attaching a neural network to reinforcement
learning will just work" is wrong — DQN's real contribution isn't the
neural network itself, but the handful of engineering devices that make
the combination train stably.**

---

## Exercises

**1. (Coding)** Complete `ReplayBuffer` and `should_update_target` below
(key lines left blank):

```python
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        # ADD ADDITIONAL CODE HERE!!
        # if the buffer exceeds capacity, remove the oldest item (FIFO) before adding

    def sample(self, batch_size):
        # ADD ADDITIONAL CODE HERE!!
        # return batch_size items sampled randomly without replacement

buf = ReplayBuffer(capacity=3)
buf.push(("s1","a1",1,"s2"))
buf.push(("s2","a2",0,"s3"))
buf.push(("s3","a3",1,"s4"))
buf.push(("s4","a4",0,"s5"))  # buffer is full, the first item gets pushed out
print(len(buf.buffer))  # 3
print(buf.buffer[0])    # ("s2","a2",0,"s3")

def should_update_target(step, update_freq):
    # ADD ADDITIONAL CODE HERE!!

print([should_update_target(s, 1000) for s in [999, 1000, 1500, 2000]])
# [False, True, False, True]
```

**2. (Hand derivation, Tier C — fallback prepared)** Suppose we train DQN
without a target network (i.e., using \\(\theta\\) directly for the target
value too). Point out that in the loss \\(J(\theta) = (r + \gamma
\max_{a'} Q(s',a';\theta) - Q(s,a;\theta))^2\\), \\(\theta\\) appears in
**both** the target term \\(Q(s',a';\theta)\\) and the prediction term
\\(Q(s,a;\theta)\\), and argue that this means a single gradient update
doesn't just move the prediction toward the target — it moves the target
itself, too. Then explain how a target network (using \\(\theta^-\\) only
for the target, updated on a fixed schedule) avoids this problem.

**Fill-in-the-blank fallback version** (if free-form argument is too
difficult):

```
Without a target network:
  theta appears in both max_a' Q(s',a';theta) and Q(s,a;theta).
  After one update, the prediction moves ______________ (closer to/farther from) the target.
  But the target itself also ______________ (changes along with it/stays fixed).

With a target network (theta^-):
  theta^- ______________ (changes/stays fixed) every step.
  So while theta is being updated, the target ______________ (moves/stays fixed).
```

**Confirm correctness**: explain, in one sentence each, what problem
arises if the target network's update frequency (`update_freq`) is set
extremely large (e.g., 1 million steps), and what problem arises if it's
set extremely small (e.g., 1 step).
