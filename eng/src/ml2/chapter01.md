# Chapter 1. Course Introduction & Neural Network Mini-Review

When you play chess or Go, nobody tells you "this move is correct" — only
once the game ends do you get a single signal back: won or lost. This
problem — **acting, and having to figure out for yourself what a good
action was purely from the reward that comes back later** — is completely
different from the supervised learning (data with correct answers \\(y\\))
and unsupervised learning (data with only structure) you may have covered
before. It's the subject of this semester: **Reinforcement Learning
(RL)**. And this semester covers reinforcement learning not through games
on a screen, but on a stage governed by the laws of physics: **robot
simulation**.

## 1.1 What Is Reinforcement Learning: How It Differs From Supervised Learning

| | Supervised Learning | Reinforcement Learning |
|---|---|---|
| Data | Given in advance, \\((x,y)\\) pairs | The agent generates it by acting |
| Signal | Correct label \\(y\\) | Reward — a score for how good something was |
| Time | Each sample is independent | The current action affects which states you'll see in the future |
| New dilemma | None | Exploration vs. exploitation |

The most important difference is that **the data isn't given in advance**.
Supervised learning trains on "photos and labels already collected," but a
reinforcement learning agent's choice of action right now determines what
state it sees next. Should it try an untested path (**exploration**), or
repeat what's currently known to work best (**exploitation**)? This
entirely new dilemma reappears, in different shapes, throughout the
semester.

## 1.2 Why Robot Simulation

Robot control is one of the settings where reinforcement learning shows
its real power — instead of a person manually writing rules for how much
force to apply at each joint or which direction to move, the goal is to
have the system discover "how to walk without falling" on its own through
trial and error. This semester uses not just abstract examples like game
screens, but simulation environments where you actually move a robot on
top of a physics engine (covered in depth in Chapters 13-14) for hands-on
practice.

## 1.3 Roadmap for This Semester

- **Chapters 2-7 (Block A)**: build reinforcement learning's theoretical
  backbone step by step, using tables — starting from multi-armed
  bandits, through MDPs, dynamic programming, Monte Carlo, TD learning,
  and n-step/eligibility traces. This order follows the table of contents
  of the standard RL textbook, Sutton and Barto's *Reinforcement Learning:
  An Introduction*.
- **Chapter 8**: Block A team project and midterm review.
- **Chapters 9-11 (early Block B)**: extend from tables to neural
  networks with deep reinforcement learning (DQN), and policy-based
  methods that learn a policy directly without going through Q-values
  (REINFORCE, PPO).
- **Chapter 12**: imitation learning and learning from human feedback —
  learning from demonstrations or preferences instead of trial and error.
- **Chapters 13-14**: the fundamentals of robot simulation and control,
  then more sophisticated physics engines (MuJoCo) and GPU-accelerated
  simulation (NVIDIA Isaac Sim).
- **Chapter 15**: model-based RL, which directly leverages a model of the
  environment, and Monte Carlo Tree Search (MCTS) — the core idea behind
  AlphaGo's conquest of Go.
- **Chapter 16**: Block B team project and a semester review.

## 1.4 This Course Is Fully Independent of ML1

You can start this course without having taken ML1 — the neural-network
knowledge this semester needs is roughly forward propagation,
backpropagation, and the intuition of reducing loss via gradient descent,
and Section 1.5 below compresses just that core into a quick refresher.
(If you did take ML1, this will be familiar — feel free to skim.)

## 1.5 Neural Network Mini-Review

A neural network is a function that transforms input \\(x\\) through
layers to produce a prediction. Forward propagation for a two-layer
network:

\\[z_1 = W_1 x + b_1, \quad a_1 = \sigma(z_1), \quad z_2 = W_2^T a_1 + b_2,
\quad a_2 = \sigma(z_2)\\]

**Training** is the process of quantifying how wrong the prediction is
with a loss function \\(J\\), and nudging the parameters
\\(W_1, W_2, b_1, b_2\\) a little in the direction that reduces \\(J\\)
(the opposite direction of the gradient). The way we compute that
gradient is **backpropagation** — applying the chain rule backward from
output to input, computing exactly how responsible each parameter is for
the final loss. This semester, we let PyTorch's `nn.Module` and
`.backward()` handle this computation automatically — we assume the
hand-derivation exercise (if you took ML1) is already behind you.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# A quick regression exercise: learning y = 2x + 1
torch.manual_seed(0)
model = MLP(1, 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

X = torch.rand(64, 1) * 4 - 2
y = 2 * X + 1

for epoch in range(200):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss:", round(loss.item(), 4))
print("Prediction at x=1:", round(model(torch.tensor([[1.0]])).item(), 3), "(expected: 3.0)")
```

## 1.6 Setting Up the Practice Environment: Gymnasium

This semester's exercises run on environments built with **Gymnasium**
(the successor to the original OpenAI Gym), a standard library — every
environment, whatever it is, follows the same interface: observe the
current state (`reset`), take one action (`step`), and get back the next
state, reward, and whether the episode ended. This unified interface means
that whatever algorithm you learn in Chapters 2-11, you can reuse it as-is
just by swapping in a different environment.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")  # balance a pole without letting it fall
obs, info = env.reset(seed=0)
print("Initial state (cart position, velocity, pole angle, angular velocity):", obs)

for _ in range(3):
    action = env.action_space.sample()  # a random policy for now
    obs, reward, terminated, truncated, info = env.step(action)
    print("action:", action, "-> next state:", obs, "reward:", reward)

env.close()
```

This CartPole environment gives you a visual, intuitive handle on the
early chapters' table-based algorithms, and it reappears in Chapter 9's
DQN.

**Reinforcement learning is a completely different game from supervised
and unsupervised learning — it's neither predicting a correct answer nor
finding structure, but figuring out what a good action is purely from
experience gathered by acting. Precisely defining the rules of this game
is where the next chapter begins.**

---

## Exercises

**1. (Coding)** Using the `MLP` class and training loop above as a
reference, train a neural network to approximate the function
\\(y = x^2\\) (use `hidden_dim=32`, train for at least 200 epochs, and
check that the prediction at `x=1.5` is close enough to the true value of
`2.25`).

**2. (Conceptual)** Referring to the table in 1.1, explain why the
"exploration vs. exploitation dilemma" doesn't exist in supervised
learning, and why it's fundamentally unavoidable in reinforcement
learning, in two or three sentences.

**3. (Hands-on)** Run the Gymnasium code above as-is, then print
`env.action_space` and `env.observation_space` to check how many actions
CartPole has (and whether they're discrete or continuous), and what
dimension the state vector is. Switch to the `Pendulum-v1` environment
and check the same things, then compare in one sentence how its action
space differs in form (discrete vs. continuous) from CartPole's.
