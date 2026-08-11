# Introduction

**Machine Learning 1** and **Machine Learning 2** are KSA's 3-1-3
advanced elective courses, building on CS1/CS2 and Math for ML (linear
algebra and probability/statistics fundamentals) as prerequisites. This
is the English edition; a [Korean
edition](https://smhanlab.com/book-ml/kor/) with the same structure is
also available — either one teaches the same material.

## How the two courses relate

**ML1** covers classical ML (linear/logistic regression, kNN,
tree-based models, GBDT) through the basics of neural networks and
backpropagation. Reinforcement learning and generative models get only
a brief conceptual preview.

**ML2** picks up ML1's neural-network foundations and goes deeper into
sequence/attention architectures (RNN → Transformer → LLM), starts
reinforcement learning properly from the MDP (Q-learning → DQN →
Policy Gradient), and covers generative models through all three major
principles (Autoencoder → VAE → GAN → Diffusion). The last two chapters
wrap up with a team project.

## How this textbook is organized

Each chapter is a single page structured as numbered sections (e.g.,
2.1, 2.2, ...) in genuine-textbook form — the section numbers follow
the actual week number the chapter covers, independent of the chapter
number shown in the sidebar.

Every chapter follows this flow:

- **Introduction**: a short real-world case or historical anecdote
  showing why the chapter's ideas matter.
- **Body sections**: concepts, formulas, and example code building up
  in order.
- **Exercises**: one coding problem (fill-in-the-blank stub) and one
  hand-derivation problem at the end, sized to about one lab-hour
  credit's worth.

## Hand-derivation difficulty tiers

Hand-derivation exercises are graded into three tiers:

| Tier | Meaning |
|---|---|
| **Tier A** (free derivation) | Well within prerequisite knowledge — proceed as-is |
| **Tier B** (appropriate but watch) | Proceed, but provide hints |
| **Tier C** (fallback needed) | Prepare both a free-derivation version and a fill-in-the-blank worksheet version — choose based on student response |

## Assessment

In-class 20% + Assignment 20% + Final 50% + Attendance 10%. In
addition to CS1/2-style "fill-in-the-blank code completion," the final
exam includes at least 2-3 proof-style questions — gradient
derivations, loss-function property proofs, algorithm-correctness
arguments.
