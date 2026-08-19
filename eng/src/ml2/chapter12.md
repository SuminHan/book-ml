# Chapter 12. Imitation Learning & Learning from Human Feedback

Every algorithm covered in Chapters 2-11 had the agent learn by trial and
error, guided by a reward signal it experienced directly. But for some
problems, "trial and error" itself is dangerous or expensive — you can't
just let a robot arm try random motions while working next to a person.
This chapter covers two methods that learn from **human demonstrations**
or **human preferences** instead of directly optimizing a reward.

## 12.1 Imitation Learning: No Reward, Just Demonstrations

The simplest form of **imitation learning** is **Behavior Cloning** (BC) —
collect (state, action) demonstration data left behind by an expert
(human or an existing policy), and turn it directly into a supervised
learning problem: "in this state, take this action." There's no need to
design a reward function, and no need to explore by interacting with the
environment — you can simply train a neural network, exactly the way we'd
handle logistic regression or multi-class classification, to take a state
as input and output an action (classification if discrete, regression if
continuous).

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-20, min(20, z))))

def train_behavior_cloning(demos, epochs, lr):
    # demos: [(state, action), ...] expert demonstrations, action is 0 or 1 (binary example)
    w, b = 0.0, 0.0
    for _ in range(epochs):
        for s, a in demos:
            pred = sigmoid(w * s + b)
            grad = pred - a          # exactly the same gradient shape as logistic regression
            w -= lr * grad * s
            b -= lr * grad
    return w, b
```

## 12.2 Why Behavior Cloning Alone Isn't Enough: Compounding Error

Behavior cloning is simple but has a fundamental weakness — training
happens only on states the **expert actually visited**, but the deployed
policy can drift, from just one small mistake, into a state the expert
never visited. In that unfamiliar state, the policy has never learned
what to do, so it makes an even bigger mistake, which leads to yet another
unfamiliar state — errors snowball over time, a problem called
**compounding error**.

**DAgger** (Dataset Aggregation) mitigates this with iteration: (1) roll
out the policy learned so far by actually navigating the environment
directly, (2) go back to the expert and ask "what should I do here?" for
the states encountered along that path, getting fresh correct-action
labels, and (3) merge this new data into the original demonstration set
and retrain. Repeating this makes the training data increasingly include
"states the policy actually visits," reducing compounding error — similar
in spirit to Chapter 6's SARSA learning the value of the policy it
actually follows: **train based on situations you'll actually encounter,
not an idealized situation.**

## 12.3 Preference-Based Reward Models: Applying RLHF to Robot Control

When even demonstrations are hard to obtain (e.g., "this robot gait looks
more natural" is hard to pin down as a single demonstration, but easy to
judge by comparing two gaits), a different strategy works — you can apply
this section's idea directly to the robot control problems introduced in
Chapter 13. Instead of a person providing the "correct action" every
time, they only need to **pick which of two attempts (trajectories) the
robot produced is better** — the exact same structure, just reshaped, as
RLHF (Reinforcement Learning from Human Feedback), used to tune LLMs
toward what people want.

For two trajectories \\(y_w\\) (the one a person preferred, the "winner")
and \\(y_l\\) (the "loser") produced in the same situation \\(x\\), train a
reward model \\(r_\phi(x,y)\\) to assign a higher score to \\(y_w\\). The
Bradley-Terry model represents this preference probability with a
sigmoid:

\\[P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\\]

The loss that maximizes this probability is once again the **same
cross-entropy form as logistic regression**:

```python
def reward_model_loss(r_win, r_lose):
    # Bradley-Terry: P(y_w > y_l) = sigmoid(r_win - r_lose)
    return -math.log(sigmoid(r_win - r_lose))
```

Once this reward model \\(r_\phi\\) is trained, its score can be used
directly as the reward for the reinforcement learning algorithms covered
in Chapters 9-11 (especially PPO) to train a policy — a particularly
powerful strategy for "problems where a person can't easily write down the
reward function by hand" (defining exactly what a natural gait means
mathematically is hard, but comparing two examples is easy).

## 12.4 Imitation Learning vs. Reinforcement Learning: When to Use Which

| | Imitation Learning (BC/DAgger) | Reinforcement Learning (Ch.2-11) |
|---|---|---|
| What's needed | Expert demonstrations (or comparisons) | A reward function, interaction with the environment |
| Exploration needed? | Basically no (DAgger needs a little) | Required (Chapter 2's exploration-exploitation dilemma) |
| Safety | Relatively safe (only imitates expert behavior) | Can try dangerous actions during training |
| Performance ceiling | Capped at expert-level (can't exceed it) | Can in principle surpass the expert (Chapter 4's optimal policy) |

In practice, the two are often combined — for example, quickly building a
"reasonably good starting policy" with behavior cloning, then fine-tuning
on top of it with reinforcement learning (PPO, etc.) to push past the
expert's limits.

**Imitation learning is a shortcut — "learn by watching something that
already does it well, without having to go through trial and error" —
while preference-based reward models are a workaround — "learn the reward
function itself just from comparisons, when it's hard to design directly."
Neither replaces the reinforcement learning algorithms we worked hard to
learn in Chapters 2-11; both are answers to the question of how to obtain
the data and reward those algorithms need, more safely and more cheaply.**

---

## Exercises

**1. (Coding)** Complete `train_behavior_cloning` and `reward_model_loss`
above (key lines left blank):

```python
import math, random

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-20, min(20, z))))

def train_behavior_cloning(demos, epochs, lr):
    # ADD ADDITIONAL CODE HERE!!
    # initialize w, b to 0, then for epochs iterations, for each (s,a) in demos,
    # compute grad = pred - a (the same as logistic regression) and update w, b

    return w, b

def expert_policy(state):
    return 1 if state < 0 else 0  # expert: always moves toward the origin (0)

random.seed(0)
demos = [(s, expert_policy(s)) for s in [random.uniform(-5, 5) for _ in range(200)]]
w, b = train_behavior_cloning(demos, epochs=30, lr=0.05)
print(sigmoid(w * (-2) + b) > 0.5)  # should be True (predicts action=1 at state=-2)

def reward_model_loss(r_win, r_lose):
    # ADD ADDITIONAL CODE HERE!!
    # -log(sigmoid(r_win - r_lose))

print(round(reward_model_loss(2.0, -1.0), 3))  # 0.049
```

**2. (Conceptual)** Explain why the compounding error problem from Section
12.2 gets worse "the longer the episode," and summarize in 1-2 sentences
the mechanism by which DAgger mitigates it.

**3. (Conceptual)** Using the example of training steering for a
self-driving car, describe (a) one concrete failure scenario that could
happen with pure behavior cloning alone, and (b) one safety problem that
could happen with pure reinforcement learning (including random
exploration) alone. Discuss why practitioners often combine the two
approaches.
