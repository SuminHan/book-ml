# Chapter 9. Function Approximation & Deep Q-Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter09_dqn_tricks.ipynb)

In 2013, DeepMind trained an algorithm that swapped Q-learning's Q-table
for a neural network wholesale, and used it to learn several Atari 2600
games — with no knowledge of the game's rules at all, just from raw pixels
and score. This result, called **DQN** (Deep Q-Network), was published in
Nature in 2015, and reached human-level or better performance on several
games.

## 9.1 A Scale the Q-Table Can't Handle

Chapter 6's Q-learning stored every state-action pair's value in a
**table**, `Q[s][a]`. This works fine when there are a few hundred
states, but the number of possible states an Atari game's screen (hundreds
of pixels wide and tall, in color) can produce is effectively infinite —
it can't all fit in a table, and most states never appear even once during
training. This is exactly the problem flagged in Section 4.5: "in a game
like Go, where the number of states is astronomically large, storing a
table over every state is simply impossible."

## 9.2 Approximating the Q-Function With a Neural Network

DQN's idea is simple: approximate \\(Q(s,a)\\) with a **neural network**
instead of a table (\\(Q(s,a;\theta)\\), where \\(\theta\\) are the
network's weights). Similar screens (states) produce similar Q-values
through the network, so it can generalize even to states it's never seen
exactly before. The loss function is Chapter 6's TD error, squared:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta) -
Q(s,a;\theta)\right)^2\right]\\]

We update \\(\theta\\) via gradient descent (specifically, backpropagation)
to minimize this loss — the same shape as supervised learning, except that
the target value \\(r + \gamma \max_{a'} Q(s',a';\theta)\\), which plays
the role of the "correct label," is itself computed using the very
\\(\theta\\) currently being trained.

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)
```

## 9.3 Problem 1: The Target Keeps Moving

Since \\(\theta\\) is updated every step, the loss function's target value
changes every step too — in supervised learning, the correct label \\(y\\)
never changes, but here the "correct label" is chasing itself. This makes
training unstable (oscillation, divergence).

**Solution: the Target Network.** Keep a second neural network,
\\(Q(s,a;\theta^-)\\), with the exact same architecture as \\(Q\\), and use
**only this target network** for computing the target value:

\\[J(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s',a';\theta^-) -
Q(s,a;\theta)\right)^2\right]\\]

\\(\theta^-\\) isn't updated every step — instead, it's copied wholesale
from \\(\theta\\)'s current value periodically (e.g., every 1000 steps).
During the interval between copies, the target stays fixed, so the loss
function becomes (over that short interval) an ordinary optimization
problem with a "fixed correct label," just like supervised learning —
effectively freezing a moving target.

```python
def dqn_loss(Q_net, target_net, states, actions, rewards, next_states, dones, gamma):
    q_pred = Q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target_net(next_states).max(dim=1).values
        target = rewards + gamma * q_next * (1 - dones)
    return ((target - q_pred) ** 2).mean()
```

## 9.4 Problem 2: The Data Is Correlated

Consecutive frames (states) are nearly identical to each other, so
learning from them in sequence causes overfitting to a small handful of
recent, similar experiences — and neural network training generally
assumes mini-batches are independent and diverse to be stable.

**Solution: Experience Replay.** Instead of using each experienced
\\((s, a, r, s')\\) for training immediately, first accumulate it in a
large storage called the **replay buffer**. When training, sample
mini-batches **randomly** from this buffer — this avoids training batches
being made up entirely of temporally adjacent experiences, and lets old
experience get reused, improving data efficiency too (another form of the
idea from Section 5.4: reusing past experience for a different purpose).

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

## 9.5 Hands-On DQN on CartPole

Combining these pieces on the CartPole environment introduced in Section
1.6 gives a training loop that samples a mini-batch from the replay
buffer, computes the loss, and periodically syncs the target network:

```python
import torch

env_state_dim, n_actions = 4, 2  # for CartPole
Q_net = QNetwork(env_state_dim, n_actions)
target_net = QNetwork(env_state_dim, n_actions)
target_net.load_state_dict(Q_net.state_dict())  # start identical
optimizer = torch.optim.Adam(Q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(capacity=10000)

# One training step (assume this is called repeatedly once the buffer has enough data)
def train_step(batch_size, gamma):
    batch = buffer.sample(batch_size)
    states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
    actions = torch.tensor([b[1] for b in batch])
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    loss = dqn_loss(Q_net, target_net, states, actions, rewards, next_states, dones, gamma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 9.6 What the Two Fixes Have in Common

Experience replay and the target network solve different problems (data
correlation vs. a moving target), but they share the same underlying
philosophy: **"artificially recreate the conditions under which supervised
learning works well (independent data, a fixed correct label), inside
reinforcement learning — a setting where those conditions are broken by
default."**

**The naive hope that "bolting a neural network onto reinforcement
learning will just work" turns out to be wrong — DQN's real contribution
isn't the neural network itself, but the handful of engineering devices
that made the combination train stably.**

---

## Exercises

**1. (Coding)** Complete the `ReplayBuffer` class and
`should_update_target` below (key lines left blank):

```python
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        # ADD ADDITIONAL CODE HERE!!
        # if the buffer exceeds capacity, remove the oldest item (FIFO) before appending

    def sample(self, batch_size):
        # ADD ADDITIONAL CODE HERE!!
        # return batch_size items sampled without replacement from the buffer

buf = ReplayBuffer(capacity=3)
buf.push(("s1","a1",1,"s2"))
buf.push(("s2","a2",0,"s3"))
buf.push(("s3","a3",1,"s4"))
buf.push(("s4","a4",0,"s5"))  # buffer is full, so the first item gets pushed out
print(len(buf.buffer))  # 3
print(buf.buffer[0])    # ("s2","a2",0,"s3")

def should_update_target(step, update_freq):
    # ADD ADDITIONAL CODE HERE!!

print([should_update_target(s, 1000) for s in [999, 1000, 1500, 2000]])
# [False, True, False, True]
```

**2. (Hands-on)** In the CartPole environment (`gym.make("CartPole-v1")`),
run a random policy for 100 steps, pushing each
`(state, action, reward, next_state, done)` into a `ReplayBuffer`. Once
the buffer holds 32 or more entries, call the `train_step` function above
5 times and confirm the loss actually gets computed and backpropagated
(runs without error).

**3. (Hand derivation, Tier C — fallback prepared)** Suppose we train DQN
without a target network (i.e., the target value also uses \\(\theta\\)
directly). Point out that in the loss \\(J(\theta) = (r + \gamma
\max_{a'} Q(s',a';\theta) - Q(s,a;\theta))^2\\), \\(\theta\\) appears in
**both** the target term \\(Q(s',a';\theta)\\) and the prediction term
\\(Q(s,a;\theta)\\), and argue that this means a single gradient update
doesn't just move the prediction closer to the target — it moves the
target itself, too. Then explain how the target network (using
\\(\theta^-\\) only for the target value, updated periodically) avoids
this problem.

**Fill-in-the-blank fallback version** (if free-form argument is too
difficult):

```
Without a target network:
  theta appears both inside max_a' Q(s',a';theta) and inside Q(s,a;theta).
  Updating theta once moves the prediction ______________ (closer to / further from) the target
  But the target itself also ______________ (moves along with it / stays the same)

With a target network (theta^-):
  theta^- ______________ (changes / stays fixed) every step
  So while theta is being updated, the target ______________ (moves / stays fixed)
```

**Confirm correctness**: explain in one sentence each what problem would
arise if the target network's update frequency (`update_freq`) were set
extremely large (e.g., a million steps), and what problem would arise if
it were set extremely small (e.g., 1 step).
