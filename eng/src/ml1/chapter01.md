# Chapter 1. Orientation

Before AlexNet appeared at the 2012 ImageNet competition, the only way to tell a
computer "there's a cat in this photo" was for a person to hand-design the
features it should look for — edge detectors, color histograms, hand-tuned
filters for particular shapes, all trying to define "cat-ness" numerically.
AlexNet did none of that hand-design; it just looked at millions of photos with
correct labels and learned for itself what to look for, beating every existing
method by a wide margin. That's the question these two courses are about:
**given data with known answers, how do we get a machine to find the rule
itself, instead of a person writing the rule by hand?**

## 1.1 Formalizing a Machine Learning Problem

Before solving a machine learning problem, three things need to be decided:

1. **Input \\(x\\)**: what will we use as features? (e.g., a house's square
   footage, number of rooms, year built)
2. **Output \\(y\\)**: what are we trying to predict? A continuous value
   (regression) or a category (classification)?
3. **Hypothesis \\(h\\)**: what form will we use to approximate the function
   from input to output? (A line? A tree? A neural network?)

**Training** is the process of adjusting the parameters of \\(h\\), given data
\\((x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})\\), so that \\(h(x^{(i)})\\)
gets as close as possible to \\(y^{(i)}\\).

## 1.2 Supervised, Unsupervised, and Reinforcement Learning

Machine learning problems fall into three broad categories.

| | Form of data | Goal | Examples this semester |
|---|---|---|---|
| Supervised | \\((x, y)\\) pairs | Predict \\(y\\) for new \\(x\\) | Regression/classification, kNN, SVM, trees, neural nets |
| Unsupervised | \\(x\\) only | Discover hidden structure in the data | k-means, PCA, EM/GMM |
| Reinforcement | State, action, reward | A policy that maximizes cumulative reward | Not covered this semester — ML2 covers reinforcement learning and robot simulation as its own dedicated subject |

- **Supervised Learning**: inputs and correct answers are given in pairs —
  "predict the price from a house's square footage and room count"
  (regression), "predict whether this email is spam" (classification). Almost
  every chapter of ML1 falls here. Supervised learning itself splits into two
  further branches: the **discriminative** approach learns \\(P(y|x)\\) (the
  probability of \\(y\\) given \\(x\\)) directly (regression, logistic
  regression, SVM, neural nets), while the **generative** approach first
  learns \\(P(x|y)\\) (how each class generates its data) and flips it via
  Bayes' rule to get \\(P(y|x)\\) (Chapter 3's Naive Bayes and GDA). We cover
  the difference between these two opposite-direction philosophies for the
  same problem in detail in Chapter 3.
- **Unsupervised Learning**: only the structure of the data is given, with no
  correct answers — "group these customers by similar tendencies"
  (clustering), "reduce 1000 features down to 10 with minimal information
  loss" (dimensionality reduction). Covered in ML1 Chapters 12-13.
- **Reinforcement Learning**: instead of correct answers, only a reward is
  given, and an agent must discover good behavior through trial and error —
  in chess or Go, nobody tells you "this move is correct," but there is a
  win/lose signal. **This semester (ML1) does not cover it at all** — ML1
  and ML2 are designed as independent courses, and ML2 covers reinforcement
  learning and robot simulation from the ground up as its dedicated subject
  (you don't need to take ML1 first to take ML2).

## 1.3 Regression vs. Classification: Same Model, Different Output

Supervised learning further splits by whether the output is continuous or
categorical: **regression** predicts a number (house price, temperature),
**classification** predicts a category (spam/not-spam, dog/cat/bird). The same
linear model \\(w^Tx + b\\) becomes regression if you use its output directly,
or classification if you pass it through a sigmoid and read it as a 0-1
probability (both covered in Chapter 2) — this pattern repeats for the next
several chapters: **the model structure stays similar; only how the output is
interpreted, and the loss function, change to fit the problem.** The same
algorithm is often used for both kinds of problems, so telling them apart is a
skill you'll need all semester.

## 1.4 The Loss Function: A Common Language

Nearly every algorithm you'll learn from here on follows this template:

1. Define a **loss function** \\(J(w)\\) that summarizes, as a single number,
   how wrong the model is.
2. Find the \\(w\\) that minimizes \\(J(w)\\) — usually via gradient descent
   (Chapter 2).

Regression's loss function (mean squared error) and classification's loss
function (cross-entropy) look different, but the underlying structure —
"quantify how wrong the model is, then reduce it" — is identical.

## 1.5 Math and Python Prerequisites Review

This semester keeps reusing three areas of math — come back here whenever
unfamiliar notation shows up:

- **Linear algebra**: dot products (\\(w^Tx = \sum_j w_jx_j\\)),
  matrix-vector products, and the "gradient" (the vector formed by
  partial-differentiating a multivariable function with respect to each
  variable, \\(\nabla_w J\\)). Eigenvectors/eigenvalues get a full treatment
  again in Chapter 12 (PCA).
- **Calculus**: the chain rule — that the derivative of a composite function
  \\(f(g(x))\\) is \\(f'(g(x))g'(x)\\). This is exactly the core tool of
  Chapter 8 (backpropagation).
- **Probability**: conditional probability and Bayes' rule \\(P(y|x) =
  \frac{P(x|y)P(y)}{P(x)}\\). Used heavily starting in Chapter 3.

```python
import numpy as np

X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 matrix
w = np.array([0.5, -0.5])               # length-2 vector
print(X @ w)   # matrix-vector product: [-0.5, -0.5]
```

`X @ w` (matrix multiplication) expresses the \\(w^Tx\\) computation that
recurs every week, in a single line — faster and more readable than writing
the loop yourself. That said, many of this book's exercise stubs are
deliberately written in plain Python (lists and for-loops) rather than
numpy/PyTorch — real practice uses numpy/PyTorch for speed, but seeing the
loop hidden behind a vector operation, at least once, helps build the right
intuition early on.

## 1.6 A Very Short History: From the Perceptron to LLMs

Knowing roughly when and in what order this semester's ideas actually
appeared makes it clearer why we cover them in this particular order.

| Year | Event |
|---|---|
| 1950s | The perceptron (single-layer neural net) appears — a similar-era idea to this semester's early classical ML models |
| 1969 | Minsky & Papert prove a single-layer perceptron can't solve XOR -> the first "AI winter" (Chapter 8) |
| 1986 | Backpropagation is rediscovered, making multi-layer network training possible (Chapter 8) |
| 1998 | LeCun's CNN (LeNet) succeeds at handwritten digit recognition (Chapter 10) |
| 2012 | AlexNet dominates ImageNet -> the start of the deep learning era (this chapter's opener) |
| 2013 | word2vec, VAE, and other modern forms of representation learning/generative models appear (Chapter 14, 15) |
| 2017 | The Transformer ("Attention Is All You Need") appears (Chapter 12) |
| 2022 | ChatGPT -- the popularization of large language models tuned via pretraining + RLHF (Chapter 13) |

**A recurring pattern**: most "breakthroughs" aren't entirely new math —
they're old ideas (gradient descent, the chain rule, probability theory)
recombined on top of bigger data and more compute. This semester's goal is
to trace, by hand, the reused fundamental ideas themselves.

## 1.7 Roadmap for This Semester

This semester splits into two eight-week blocks. **Block A (Chapters 2-7)**
covers classical ML models that are powerful even without a neural network
(regression, Naive Bayes/GDA, distance-based models, SVM, regularization,
tree-based models) — these still routinely beat neural networks on tabular
data. Chapter 8 wraps up Block A with a team project and midterm review.

After the midterm, **Block B (Chapters 9-16)** moves to neural networks.
We derive backpropagation by hand (Chapter 9), build up through CNNs
(Chapter 10), then trace the core lineage of modern deep learning in one
line — sequence models, the Transformer, and LLMs (Chapters 11-13).
Finally we close with unsupervised learning (PCA, embeddings) and
latent-variable generative models, from EM/GMM through VAE, GAN, and
Diffusion (Chapters 14-15). Chapter 16 is the second team project and a
semester review.

**Reinforcement learning and robot simulation are not covered at all this
semester (ML1)** — ML1 and ML2 are designed as independent courses, and ML2
covers this subject from the ground up on its own. You don't need to take
ML1 first to take ML2 (ML2's Week 1 gives a compressed refresher of the
neural-network basics it needs).

**The question to ask before choosing any model is always the same: what data
does this problem actually have, and what am I trying to predict?**
