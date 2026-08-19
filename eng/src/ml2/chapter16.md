# Chapter 16. Block B Capstone: Team Project & Semester Review

Chapters 9-15 extended tabular reinforcement learning to neural networks
(DQN), learned policies directly (REINFORCE, PPO), learned from human
demonstrations and preferences (imitation learning), and covered actual
physical simulation (MuJoCo, Isaac Sim) and tree search (MCTS). This
chapter is Chapter 8's counterpart — a chance to apply this second half's
tools directly to a robot simulation environment, and wrap up the
semester.

## 16.1 Project Structure

- **Environment**: pick one of PyBullet's or Gymnasium's continuous
  control environments (Pendulum, a MuJoCo-based robot environment, etc.).
- **Application**: train a policy using at least one method from Chapters
  9-15 (DQN, PPO, imitation learning, or MCTS).
- **Required — at least one mathematical justification**: for example (1)
  if you used PPO, present the effect of Section 11.4's clipping on
  training stability via a with/without-clipping comparison, (2) if you
  used imitation learning, check whether Section 12.2's compounding error
  actually showed up (does performance drop outside the expert's data
  distribution?), or (3) if you applied domain randomization, connect the
  effect to Section 13.2's discussion of the reality gap.
- As in Chapter 8, always present a learning curve (cumulative reward
  over episodes) — the reinforcement learning equivalent of a supervised
  learning loss curve, and the basic evidence for "is learning actually
  happening."

## 16.2 Presentation and Peer Review (Blocks 1-2)

The format is the same as Chapter 8. Add this item to the peer-review
checklist: "was the exploration-exploitation balance appropriate" (how did
Chapter 2's dilemma show up in the actual learning curve), and "did the
algorithm choice fit the environment's characteristics (discrete/continuous
actions, sparse rewards, etc.)."

## 16.3 ML2 Concept Map (Review)

| Chapter | What We Learned | The Key Question |
|---|---|---|
| Ch02 | Multi-armed bandits | How do we handle exploration vs. exploitation even without state? |
| Ch03 | MDPs | How do we mathematically formalize a reinforcement learning problem? |
| Ch04 | Dynamic programming | How do we compute an optimal policy when we know the model? |
| Ch05 | Monte Carlo | How do we learn with no model, just from episode outcomes? |
| Ch06 | Temporal-difference learning | How do we learn without waiting for an episode to end? |
| Ch07 | n-step/eligibility traces/Dyna | What's between MC and TD, and how do we reuse experience? |
| Ch09 | DQN | How do we approximate the Q-function with a neural network instead of a table? |
| Ch10 | Policy-based RL | How do we learn a policy directly, in continuous spaces, without going through Q? |
| Ch11 | PPO | How do we keep a policy from moving too far in a single update? |
| Ch12 | Imitation learning/RLHF | How do we learn from demonstrations or preferences instead of trial and error? |
| Ch13 | Robot simulation | How do we handle continuous control problems governed by physical laws? |
| Ch14 | MuJoCo/Isaac Sim | How do we balance simulation precision against speed? |
| Ch15 | MCTS | How do we read ahead when we have a perfect model? |

## 16.4 The One Pattern That Runs Through ML1 → ML2

Let's give one final summary of the structure that runs through both
courses — familiar if you took ML1, and something you've already
confirmed plenty of times across this semester's 13 chapters even if ML2
is all you took:

1. **Define a model**: decide the form of the function mapping state to
   action (or a distribution over actions) — a Q-table, a neural Q-function,
   a policy network, a game tree, ...
2. **Quantify how wrong (or how far from the goal) things are**: the TD
   error, a policy gradient objective, PPO's clipped objective, MCTS's
   win-rate statistics, ... — the form is different every time, but the
   role — "summarize how good or bad things are right now in a single
   number" — is always the same.
3. **Adjust parameters (or the search direction) to improve that number**:
   gradient descent, or iterating a tree search — this is, in the end,
   the final step of every algorithm covered this semester.

Whenever you encounter a new paper or system, developing the habit of
breaking it down into "what does it say for each of these three
questions" is the most durable tool these two courses hope to leave you
with.

## 16.5 A Final Look at RL and Robotics Trends

These are too early to cover this semester, but here's a brief look at a
few directions of active research right now — the goal isn't depth, but
sketching a map of where what you've learned can lead.

- **Offline Reinforcement Learning**: research on training a policy purely
  from a large pre-collected log of data, with no direct interaction with
  the environment at all. Especially important in settings where
  exploring directly with a physical robot is itself dangerous or
  expensive (self-driving, medicine) — similar in spirit to Chapter 12's
  imitation learning, but different in that it learns from arbitrary log
  data (a mix of good and bad), not clean demonstrations.
- **Generalist Robot Policies**: instead of a policy specialized to one
  task, efforts to train a single large policy that generalizes across
  many robots and tasks (RT-2, OpenVLA, etc.) — close to applying ML1's
  LLM pretraining idea to robot control: first pretrain broadly on a huge
  amount of robot demonstration/interaction data, then fine-tune for a
  specific task.
- **Advances in Sim-to-Real**: beyond Section 13.2's domain
  randomization, research continues into closing the gap between
  simulation and the real world through learning itself — Section 14.3's
  GPU-accelerated simulation (Isaac Sim, etc.) is contributing to this
  problem in the direction of "exposing the policy to a much wider variety
  of simulated conditions, much faster."

## 16.6 After the Presentation: What's Next

If you want to dig deeper after finishing this course, the parts each
chapter flagged as "beyond this semester's scope" (TRPO's detailed math,
DAgger's convergence proof, multi-agent RL's Nash equilibria, the details
of AlphaZero's self-play training, and more) are the natural next targets
— if this semester's goal was to make these concepts feel unintimidating
the first time you met them, what comes next is deepening each one
according to your own interests.

**If ML1 was a world of "learning from data with correct answers," ML2 was
a completely different world of "learning purely from experience gained by
acting." Starting from the exploration-exploitation dilemma of bandits, we
built up the theory through MDPs, dynamic programming, Monte Carlo, and
temporal-difference learning, then combined it with neural networks to
carry that theory all the way into the realistic stage of robot
simulation. Now, in both worlds, you should be able to ask the same
question: what data (or experience) do I actually have, and what am I
trying to optimize?**
