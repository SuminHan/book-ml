# Chapter 8. Block A Capstone: Team Project & Review

Regression, generative classifiers (Naive Bayes/GDA), distance- and
margin-based models (kNN, SVM), regularization, and tree ensembles (GBDT)
from Chapters 2-7 were each treated as an independent chapter's problem.
In front of real data, though, you have to judge for yourself which model
fits. This chapter introduces no new concepts — instead it's a chance to
apply what you've learned to real data, review other teams' results, and
prepare for the midterm.

## 8.1 Why a Project Now

By the time you've covered tree-based models (Chapter 7), you already hold
nearly a full toolkit for tabular data — much of it can be handled with
these tools alone, without the neural networks of Block B. Now is the best
time to check whether each chapter's code actually works in front of real
data.

## 8.2 Project Structure

- **Data**: pick a tabular dataset from Kaggle (kaggle.com/datasets) or the
  UCI Machine Learning Repository (archive.ics.uci.edu) — since this
  project is meant to validate the classification/regression models from
  Chapters 2-7, table (row/column) data is a better fit than images or text.
- **Application**: apply **at least two** models from Chapters 2-7 (e.g.,
  logistic regression and GBDT) to the same data and compare performance.
- **Required — at least one mathematical justification**: for example (1)
  if you applied regularization, explain why performance changed from
  Chapter 6's bias-variance perspective, (2) if you used GBDT, decompose one
  prediction with Chapter 7's SHAP, or (3) if you compared multiple models,
  present the comparison using Chapter 2.7's PR-AUC to account for class
  imbalance.
- **Follow the validation discipline**: stick to Chapter 6.3's
  train/validation/test three-way split — check test performance only once.

## 8.3 Presentation (Block 1)

5-7 minutes per team: problem definition (1 min) → data and preprocessing
(1 min) → model choices and reasoning (2 min) → results and mathematical
justification (2 min) → what didn't work and why (1 min). Explaining "why
this model, and why this result" earns more credit than simply reporting
"we achieved 90% accuracy."

## 8.4 Peer Review (Block 2)

Evaluate two other teams' presentations against this checklist:

| Item | What to check |
|---|---|
| Problem definition | Is it clear what's being predicted/classified? |
| Methodology fit | Does the chosen model fit the data's characteristics (imbalance, dimensionality)? |
| Mathematical justification | Was the requirement met, and is the explanation actually sound? |
| Honesty of results | Were the parts that didn't work analyzed rather than hidden? |

Checklist-based review counts toward part of a team's score. A good review
asks **specific, falsifiable questions** rather than saying "good job" —
e.g., "This didn't use Chapter 6's cross-validation — could this validation
performance be a fluke?"

## 8.5 Midterm Review (Block 3)

| Chapter | Key question | Concepts to revisit |
|---|---|---|
| Ch02 | How does the same linear model solve both regression and classification? | Gradient descent, normal equation, cross-entropy |
| Ch03 | What's the difference between the generative and discriminative approaches? | Bayes' rule, the GDA/logistic-regression relationship |
| Ch04 | What are the limits of predicting from "closeness" alone? | Bias-variance, curse of dimensionality |
| Ch05 | What changes if you decide boundaries by margin instead of probability? | Support vectors, the kernel trick |
| Ch06 | How do you control overfitting through the loss function? | L1/L2 regularization, cross-validation |
| Ch07 | How do you ensemble trees, and how do you explain their predictions? | Information gain, random forest vs. GBDT, SHAP |

The midterm applies these six chapters' concepts to new situations
(including hand derivations) — a good way to review is checking whether the
normal-equation/Ridge-regression derivation pattern from Chapters 2 and 6
carries over to other loss functions.

**No model you've learned so far is "always best" — the biggest lesson this
project leaves you with is that real practice requires judging which tool
fits the data's actual characteristics.**
