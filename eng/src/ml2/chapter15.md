# Chapter 15. Model-Based RL & Monte Carlo Tree Search

In 2016, DeepMind's **AlphaGo** made headlines by defeating world-champion
Go player Lee Sedol. The number of possible Go board configurations is
said to exceed the number of atoms in the universe, so Chapter 9's DQN
approach of "approximating value with a neural network" alone wasn't
enough — one of AlphaGo's core weapons was this chapter's topic, **Monte
Carlo Tree Search** (MCTS). This chapter takes the "planning with a model"
idea we glimpsed in Section 7.3 and extends it much further, for problems
like games where the rules are known precisely.

## 15.1 Model-Based RL, Revisited

Chapter 7.3's Dyna-Q built a simple model from real experience and used it
for **extra imaginary training**. **Model-Based Reinforcement Learning** is
the general name for this idea — if you know or can learn a model of the
environment (transition probabilities and a reward function, or an
approximation of them), you can simulate many moves ahead **without
actually acting** in the real world. Where Dyna-Q used this for updating
values, MCTS uses it directly to **decide what action to take right now**.

## 15.2 In Problems Like Go, the Model Is Free

Games like chess and Go have a special property — the rules are known
perfectly. There's no need to learn the transition probability
\\(P(s'|s,a)\\) — "make this move and the board changes exactly like
this" — it's already known (the ideal condition for using Chapter 4's
model-based methods). The problem is that there are too many states (Go's
board has roughly \\(10^{170}\\) possible configurations) to compute
everything the way Chapter 4's value iteration does — so we need a search
that digs deep only "near the current situation."

## 15.3 MCTS: Four Steps, Repeated

Starting from the current state, MCTS repeats the following four steps a
fixed number of times, building up statistics on "which move looks most
promising":

1. **Selection**: starting from the tree's root (the current state), pick
   a child node at each fork using a rule that balances "performance so
   far" against "how little this has been tried" (exactly the same idea
   as Chapter 2's UCB), descending until reaching a node that hasn't been
   fully expanded yet.
2. **Expansion**: add one new child node (an untried move) to that node.
3. **Simulation (Rollout)**: play the game all the way to the end from
   that new node, using a simple policy (or, in the extreme, completely
   random play) — just like Chapter 5's Monte Carlo, actually play it out
   and get a result (win/loss).
4. **Backpropagation**: propagate that result back as statistics (visit
   count +1, win rate updated) to every node visited along the way during
   selection — this "backpropagation" shares only a name with ML1's neural
   network backpropagation, and is a completely different concept: it's
   not propagating a neural network's gradient, but a tree's win/loss
   statistics.

After repeating these four steps hundreds to tens of thousands of times,
we pick the root's **most-visited** child (the move repeatedly confirmed
to be the most promising) as the move to actually play.

```python
import math, random

def uct_select(node_children, parent_visits, c=1.4):
    # node_children: {move: (wins, visits)}
    # exactly the same shape as UCB1 -- Section 2.3 applied directly to a game tree
    return max(node_children, key=lambda m: node_children[m][0] / node_children[m][1] +
               c * math.sqrt(math.log(parent_visits) / node_children[m][1]))
```

## 15.4 Confirming It With a Small Example

In a simple game (a variant of Nim) where players alternately take 1-3
stones and whoever takes the last stone wins, running MCTS with 3000
simulations from a position with 5 stones left correctly finds that
"taking 1 stone, leaving 4 for the opponent" is the most promising
move — exactly matching this game's known mathematical winning strategy
(making the number of stones left a multiple of 4). MCTS never knew the
game's rule (the multiple-of-4 strategy) at all — it discovered this
optimal strategy entirely on its own, simply from "simulating a lot and
picking the move with the highest win rate."

## 15.5 AlphaGo/AlphaZero: Combining MCTS With Neural Networks

Pure MCTS plays the game randomly (or with a very simple rule) all the way
to the end during the simulation step — in a game like Go, where the
impact of one move might not show up until much later, random simulation
can be inaccurate. AlphaGo and its successor AlphaZero's core improvement
was replacing this random rollout with a **neural network**: one network
estimates "how promising is each move from this state" (a policy, playing
the same role as Chapter 10's \\(\pi_\theta\\)), and another (or a
different output head of the same network) estimates "how favorable is
this state ultimately" (a value, playing the same role as ML1's
\\(V(s)\\)). MCTS uses these network estimates as a guide to narrow its
search much more efficiently, and the more accurate moves that MCTS finds
through that search are, in turn, used as the supervised-learning targets
for training the neural network further — **a loop where search (MCTS)
and learning (the neural network) improve each other.**

## 15.6 A Look at Multi-Agent RL (Optional)

So far we've dealt with environments containing only a single agent.
**Multi-Agent Reinforcement Learning** covers cases where multiple agents
act simultaneously in the same environment — they might cooperate (a team
sport, collaborating logistics robots) or compete (Go, game AI). MCTS
itself, in a two-player game where "the opponent also plays optimally,"
can actually be viewed as a probabilistic approximation of **minimax**
search — there's a structural similarity here to ML1's GAN and its
min-max game (Chapter 15 of ML1), where two participants interact toward
different goals. The detailed theory (Nash equilibria, cooperative games,
etc.) is beyond this semester, but it's worth remembering that "the problem
changes fundamentally once there's more than one agent."

**MCTS pushes the idea "if you have a model (know the rules), you can read
ahead before actually acting" to its limit — where Chapter 7's Dyna-Q used
a learned model to improve values, MCTS uses a perfect model to improve
the decision happening right now. And as AlphaZero showed, combining this
search with neural network learning reaches a level of performance neither
pure search nor a pure neural network could achieve alone.**

---

## Exercises

**1. (Coding)** Complete `uct_select` above (key lines left blank):

```python
import math

def uct_select(node_children, parent_visits, c=1.4):
    # node_children: {move: (wins, visits)}
    # ADD ADDITIONAL CODE HERE!!
    # for each move, compute wins/visits + c*sqrt(log(parent_visits)/visits), return the move with the max

children = {"A": (7, 10), "B": (3, 5), "C": (0, 1)}
print(uct_select(children, parent_visits=16))
# a barely-tried move like C can get picked because its uncertainty bonus is large
```

**2. (Conceptual)** Among Section 15.3's four steps (selection-expansion-
simulation-backpropagation), match which step most directly corresponds
to Chapter 2's UCB, and which most directly corresponds to Chapter 5's
Monte Carlo, and explain why for each.

**3. (Conceptual)** Explain why AlphaZero using a neural network instead
of random rollouts matters especially for "games like Go, where the
impact of a move might not show up until much later," connecting it to
the limitations of random rollouts.
