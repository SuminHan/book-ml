# Chapter 11. Team Project: Implementation

Everything covered over the past 10 chapters — defining loss functions,
deriving gradients, implementing models, and arguing their correctness —
was treated as an independent chapter's problem each time. The last two
chapters are a time to assemble these pieces directly into one finished
project.

## 11.1 Why a Team Project

Real machine learning problems aren't neatly divided as "this chapter is
logistic regression, that chapter is CNN." Cleaning data, judging which
model fits, diagnosing why training isn't working (overfitting? vanishing
gradients? no signal in the data at all?), and explaining the results to
someone else — going through this entire process yourself is the surest
way to integrate the past 10 chapters' separate topics into one whole.

## 11.2 Project Structure

- **Code**: must be reproducible — running the same code (with randomness
  fixed) should give the same result. The entire pipeline, from data
  preprocessing to final evaluation, should be organized into
  scripts/notebooks.
- **Report**: follows the structure problem definition → methodology →
  experimental results → (required) at least one mathematical
  justification → limitations and future improvements.
- **Poster**: a summary for the presentation (Chapter 12) — condensed to
  1-2 key figures (learning curves, example results, etc.) and 3-4 of the
  most important conclusions.

## 11.3 Candidate Project Topics

The topic is free, but here are some directions to consider:

- **Graph anomaly detection**: using publicly available synthetic data,
  detect abnormal nodes/edges with a graph neural network (GNN, beyond
  this semester's scope but connected to ML1 Chapter 10's embedding
  concept) or traditional methods.
- **Image classification applications**: apply the CNN structure from ML1
  Chapter 9 to a real image dataset, trying transfer learning.
- **A simple chatbot/LLM application project**: apply the prompting
  techniques from Chapters 2-4 (the LLM section) to a real task
  (summarization, classification, search).

## 11.4 Real-World Problems You'll Often Hit in a Team Project

Here's a preview of how the concepts covered this semester show up in an
actual project:

- **Data is messier than you'd think**: check for missing values,
  outliers, and class imbalance first (a place you'll really feel why ML1
  Chapter 3's Precision/Recall matters).
- **When training isn't working**: if the loss isn't decreasing, suspect,
  in order, the learning rate (ML1 Chapter 2), vanishing gradients (ML1
  Chapter 8), and whether there's simply no signal in the data.
- **Don't pick a model without validation**: choosing a model based only
  on training-data performance is an easy way to end up selecting an
  overfit model — always do a final evaluation on separate
  validation/test data.

## 11.5 Requirement: At Least One Mathematical Justification

The project report must include **at least one choice justified
mathematically**. Here are examples that satisfy this requirement (adapt
to your project's topic):

1. If you chose a particular loss function, derive why its gradient takes
   a form suitable for the problem (reusing the pattern from ML1 Chapters
   2-3).
2. If you used a regularization technique (dropout, weight decay, etc.),
   explain how it reduces overfitting with a formula or intuitive argument
   (ML1 Chapter 8).
3. Quantitatively compare why you chose a particular model architecture,
   using parameter count/computational complexity formulas (reusing ML1
   Chapter 9's CNN parameter-counting pattern).
4. For an RL/generative project, connect one of the concepts from Chapters
   5-10 (Bellman equation, ELBO, Nash equilibrium) directly to your
   implementation.

This is meant to preserve, at the larger scale of a project, the
distinction emphasized all semester between "code that happens to be
correct" and "knowing why it works."

## 11.6 Aiming for External-Submission Quality

Prepare this project as a standardized set of three deliverables:
**code + report + poster** — aim for a level ready to hand over
immediately if an outside institution (a university, etc.) requests these
materials later.

## 11.7 Grading Criteria

- Clarity of the problem definition
- Appropriateness of the methodology (were the concepts learned applied
  correctly?)
- Accuracy and depth of the mathematical justification
- Honest reporting of results (were the parts that didn't work hidden, or
  was the cause analyzed?)
- Reproducibility of the code
