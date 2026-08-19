# Chapter 8. Block A Capstone: Team Project & Review

Chapters 2-7 covered the whole of table-based (tabular) reinforcement
learning theory — bandits, MDPs, dynamic programming, Monte Carlo,
temporal-difference learning, and n-step methods. This chapter introduces
no new concepts — instead it's a chance to implement and compare these
methods directly on Gymnasium's tabular environments, and prepare for the
midterm.

## 8.1 Project Structure

- **Environment**: pick one of Gymnasium's tabular (discrete state and
  action) environments — `FrozenLake-v1`, `CliffWalking-v0`, and
  `Taxi-v3` are all good candidates.
- **Application**: apply **at least two** algorithms from Chapters 4-7
  (e.g., Q-learning and SARSA, or MC control and n-step TD) to the same
  environment and compare their learning curves and final policies.
- **Required — at least one mathematical justification**: for example (1)
  if you compared Q-learning and SARSA, explain why they learned different
  policies using Section 6.3-6.4's on-policy/off-policy distinction, (2) if
  you experimented with different \\(\varepsilon\\) or \\(\alpha\\)
  values, connect the results to Section 6.5's convergence conditions, or
  (3) if you used Dyna-Q, directly measure and present the relationship
  between Section 7.3's number of planning steps and learning speed.

## 8.2 Presentation (Block 1)

5-7 minutes per team: problem definition (explain the environment, 1 min)
→ algorithms applied and reasoning (2 min) → learning curve/final policy
comparison and mathematical justification (2-3 min) → what didn't work and
why (1 min).

## 8.3 Peer Review (Block 2)

Evaluate two other teams' presentations against this checklist:

| Item | What to check |
|---|---|
| Environment description | Are states, actions, and rewards clearly defined? |
| Algorithm choice | Did they pick methods that fit a tabular environment? |
| Mathematical justification | Was the requirement met, and does it match what was actually observed? |
| Honesty of results | Were unstable stretches of training analyzed rather than hidden? |

## 8.4 Midterm Review (Block 3)

| Chapter | Key question | Concepts to revisit |
|---|---|---|
| Ch02 | Does the exploration-exploitation dilemma exist even without state? | ε-greedy, optimistic initialization, UCB |
| Ch03 | How do we mathematically formalize a reinforcement learning problem? | The 5 MDP components, the Markov property, the discount factor |
| Ch04 | How do we compute an optimal policy when we know the model? | The Bellman equation, policy/value iteration, contraction mapping |
| Ch05 | Can we learn with no model, just from episode outcomes? | First-visit MC, MC control, importance sampling |
| Ch06 | Can we learn without waiting for an episode to end? | The TD error, Q-learning vs. SARSA |
| Ch07 | What's between MC and TD, and how do we reuse experience? | n-step TD, eligibility traces, Dyna-Q |

The midterm applies these six chapters' concepts to new MDPs (including
hand derivations) — a good way to review is checking whether the pattern
of "solving the Bellman equation by backward substitution from the
terminal state" (Chapter 4) and "arguing the convergence conditions of the
TD error" (Chapter 6) still applies to a different MDP.

**Every algorithm covered in Chapters 2-7 is ultimately answering one
question: "how do we quantify an uncertain future reward, right now?" Now
it's time to combine that question with neural networks and move toward
problems too large to fit in a table.**
