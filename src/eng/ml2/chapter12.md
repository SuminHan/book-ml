# Chapter 12. Team Project: Presentation

If last chapter was about implementing the project, this chapter is about
presenting the results and reflecting on the semester as a whole. Finally,
we close out the semester with a brief look at a few recent trends that
are still actively developing right now.

## 12.1 Presentation Checklist

- **Problem definition** (1 minute): what were you trying to solve? Why
  is that problem interesting?
- **Methodology** (2-3 minutes): what model/algorithm did you use, and
  why did you choose it?
- **Mathematical justification** (2 minutes): make sure to hit at least
  one of the justifications from last chapter's report during the talk
  too.
- **Results** (2 minutes): show both quantitative results (numbers,
  graphs) and qualitative results (examples).
- **Limitations and reflection** (1-2 minutes): what didn't work, and why
  do you think that happened? What would you have tried with more time?

A good presentation weighs "why you built it that way" and "what you
learned" more heavily than "what you built." In particular, not hiding the
parts that didn't work, and diagnosing the cause using concepts from this
semester — for example, "the loss wasn't decreasing, and on closer
inspection the gradient was vanishing" — is worth far more than simply
saying "we achieved 90% accuracy."

## 12.2 ML2 Concept Map (Review)

| Chapter | What We Learned | The Key Question |
|---|---|---|
| Ch02 | RNN | How do we remember and process ordered data? |
| Ch03 | Attention/Transformer | How do we grasp context all at once, without sequential processing? |
| Ch04 | LLM | How does next-token prediction lead to such broad capability? |
| Ch05 | MDP/Policy Evaluation | How do we compute an uncertain future reward right now? |
| Ch06 | Q-learning | How do we learn optimal behavior without a model of the environment? |
| Ch07 | DQN | How do we combine neural networks and RL stably? |
| Ch08 | Policy Gradient | How do we learn a policy directly in a continuous action space? |
| Ch09 | VAE | How do we work around an intractable likelihood to optimize it anyway? |
| Ch10 | GAN/Diffusion | How do adversarial training and gradual noise removal achieve the same goal through different principles? |

## 12.3 The One Pattern That Runs Through ML1 → ML2

Let's give one final summary of the structure that runs through all 24
chapters across both semesters:

1. **Define a model**: decide the form of the function mapping input to
   output (a line, a tree, a neural network, a Transformer, ...).
2. **Quantify how wrong it is**: a loss function (MSE, cross-entropy, TD
   error, ELBO, a min-max objective, ...) — the form is different every
   time, but the role is always the same.
3. **Adjust parameters to reduce that loss**: gradient descent (directly,
   or via backpropagation) — this is, in the end, the final step of every
   algorithm covered this semester.

Whenever you encounter a new paper or model, developing the habit of
breaking it down into "what does it say for each of these three
questions" is the most durable tool these two courses hope to leave you
with.

## 12.4 A Final Look at Recent Trends

These are too early to cover this semester, but here's a brief look at a
few directions of active research right now — the goal isn't depth, but
sketching a map of where what you've learned can lead.

- **Theorem Proving**: a research area where LLMs are used to actually
  prove mathematical theorems. Building on Chain-of-Thought (Chapter 4),
  this is developing toward models that generate and verify step-by-step
  arguments themselves.
- **Autoformalization**: research on automatically converting
  mathematical claims written in natural language by humans into a
  formal language that a computer can verify — aiming at the goal of "the
  computer itself checks whether a proof is correct."
- **Extensions of Chain-of-Thought (CoT)**: the CoT prompting we saw in
  Chapter 4 is now developing further, into training the model itself to
  "reason across multiple steps" (reasoning models).

## 12.5 Closing Out the Semester

Let's recall the one sentence that ran through both of these courses:
**define a model, quantify how wrong it is with a loss function, and
adjust parameters to reduce that loss.** From linear regression to
Transformer to GAN, this principle repeated all semester, changing only
its shape. Whenever you encounter a new model, returning to these three
questions — what is it predicting, how does it measure how wrong it is,
and how does it reduce that — lets you read even the most complicated-
looking recent paper in the same language.
