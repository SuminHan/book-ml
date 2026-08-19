# Chapter 13. Robot Simulation & Control Basics

The reinforcement learning algorithms covered so far were treated in
simplified environments like CartPole. Working with an actual robot — a
physical system with arms, legs, and wheels — means accounting not just
for the fact that "actions are continuous forces/torques," but for the
fact that those forces move the robot through actual physical laws
(inertia, gravity, friction). This chapter covers the fundamentals of
robot simulation, where you'll actually apply the algorithms learned in
Chapters 9-12.

## 13.1 Robotics Basics: State Space and Kinematics

A robot's state is usually represented by each joint's **angle** and
**angular velocity** — for a 2-joint robot arm, the state would be a
4-dimensional vector like \\((\theta_1, \theta_2, \dot\theta_1,
\dot\theta_2)\\). **Forward kinematics** computes the actual position of
the end-effector (the arm's tip) from these joint angles, and **inverse
kinematics** does the reverse — computing "what joint angles are needed to
send the end-effector to this position." This geometric computation is a
deep subject of robotics in its own right, but from a reinforcement
learning perspective, one thing matters most: **states and actions are now
continuous values with real physical meaning.** Section 3.1's MDP
\\(\mathcal{S}, \mathcal{A}\\) are no longer abstract symbols — they now
carry units of actual angles and forces.

## 13.2 Why Simulation Is Necessary: Sim-to-Real

Training reinforcement learning directly on a physical robot runs into
several real-world problems — the robot can break down or pose a danger
to people over tens of thousands of trial-and-error attempts, and above
all, it's **slow** (the physical world moves far slower than a computer
simulation). So in practice, a policy is first trained in a **simulation**
running on a physics engine, and then that policy is transferred to the
physical robot — this transfer process is called **sim-to-real**.

The problem is that simulation never perfectly reproduces real physics
(friction coefficients, sensor noise, subtle material differences, etc.) —
a policy that works well only in simulation and fails on the real robot
suffers from what's called the **reality gap**. A common way to shrink
this gap is **domain randomization** — randomizing the simulation's
physical parameters (friction, mass, lighting, etc.) every training
episode. Forcing the policy to work well not on one fixed simulation
setup, but across "a wide variety of physical conditions," tends to make
it generalize better to any one of them — including the real robot's
actual physical conditions. This is similar in spirit to how regularization
"reduces variance by making the model less flexible": here, "diversifying
the environment itself" keeps the policy from overfitting to any one
specific simulation setup.

## 13.3 Gymnasium/PyBullet Robot Environments

This semester's exercises run on Gymnasium's robot-like environments —
the CartPole and Pendulum we already used in Chapters 1 and 9 are actually
very simplified "robot control" problems too (balancing a pole and
stabilizing a pendulum are both the smallest possible forms of joint
control). Let's move to a continuous control problem a bit closer to a
real robot:

```python
import gymnasium as gym
import numpy as np

env = gym.make("Pendulum-v1")  # the simplest "joint control" problem with a continuous action space
obs, info = env.reset(seed=0)
print("State (cos theta, sin theta, angular velocity):", obs)
print("Action space:", env.action_space)  # Box(-2.0, 2.0, (1,)) -- a continuous torque value

# A simple hand-designed PD (proportional-derivative) controller: push back harder the larger the angle and angular velocity are
total_reward = 0.0
for _ in range(50):
    cos_th, sin_th, thdot = obs
    theta = np.arctan2(sin_th, cos_th)
    action = np.clip(-2.0 * theta - 0.5 * thdot, -2.0, 2.0)
    obs, reward, terminated, truncated, info = env.step([action])
    total_reward += reward
print("PD controller's cumulative reward over 50 steps:", round(total_reward, 2))
env.close()
```

This PD (Proportional-Derivative) controller isn't reinforcement learning
at all — it's just a rule a human designed by hand. Even so, it performs
far better than random actions (a random policy's 50-step cumulative
reward is usually around -1200 to -1600, while this simple rule improves
it to around -240). What this comparison shows: **the goal of
reinforcement learning is to find, directly from data, a policy that does
better than a hand-designed controller like this, or that works even in
situations too complex for a human to design rules for.**

## 13.4 Applying PPO to Continuous Control

Chapter 11's PPO extends naturally not just to discrete actions (a softmax
policy) but to continuous ones too — just have the network output the
mean and standard deviation of a normal distribution for each action
dimension:

\\[\pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta(s)^2)\\]

The log-probability \\(\log \pi_\theta(a|s)\\) is computed directly from
the normal distribution's density function, and Chapter 10.4's
log-derivative trick and Chapter 11's clipped objective keep exactly the
same shape regardless of whether actions are discrete or continuous —
only "how the action is parameterized" changes, while the optimization
machinery behind it stays the same. This is a major strength of
policy-based methods.

```python
import math

def gaussian_log_prob(action, mu, sigma):
    # log-probability for a continuous action -- the only difference from softmax is using a normal distribution
    return -0.5 * math.log(2 * math.pi * sigma**2) - (action - mu)**2 / (2 * sigma**2)
```

**Robot simulation is the proving ground that moves the algorithms learned
so far (DQN, PPO) into a more realistic setting — a continuous action
space governed by actual physical laws. The next chapter extends this
simulation to more sophisticated physics engines, and to large-scale
GPU-accelerated environments.**

---

## Exercises

**1. (Hands-on)** Run the code above as-is, then double the PD
controller's coefficients (`-2.0`, `-0.5`), and also try halving them, and
compare how the 50-step cumulative reward changes. Describe what problem
occurs when the coefficients are too large or too small (does it
oscillate? does it respond too slowly?).

**2. (Coding)** Complete `gaussian_log_prob` above (key lines left blank):

```python
import math

def gaussian_log_prob(action, mu, sigma):
    # ADD ADDITIONAL CODE HERE!!
    # log of the normal distribution's density function: -0.5*log(2*pi*sigma^2) - (action-mu)^2/(2*sigma^2)

print(round(gaussian_log_prob(0.0, 0.0, 1.0), 4))   # log-density of the standard normal at x=0, about -0.9189
print(round(gaussian_log_prob(2.0, 0.0, 1.0), 4))   # further from the mean means a smaller (more negative) log-probability
```

**3. (Conceptual)** Explain, in two or three sentences, why domain
randomization plays a role similar to regularization, from the
perspective of "preventing the model from overfitting to one specific
condition."
