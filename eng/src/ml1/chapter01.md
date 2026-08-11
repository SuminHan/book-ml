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
| Supervised | \\((x, y)\\) pairs | Predict \\(y\\) for new \\(x\\) | Linear regression, logistic regression, kNN, trees, neural nets |
| Unsupervised | \\(x\\) only | Discover hidden structure in the data | k-means, PCA |
| Reinforcement | State, action, reward | A policy that maximizes cumulative reward | ML2 Ch05-Ch08 |

- **Supervised Learning**: inputs and correct answers are given in pairs —
  "predict the price from a house's square footage and room count"
  (regression), "predict whether this email is spam" (classification). Almost
  every chapter of ML1 falls here.
- **Unsupervised Learning**: only the structure of the data is given, with no
  correct answers — "group these customers by similar tendencies"
  (clustering), "reduce 1000 features down to 10 with minimal information
  loss" (dimensionality reduction). Covered in ML1 Chapter 10.
- **Reinforcement Learning**: instead of correct answers, only a reward is
  given, and an agent must discover good behavior through trial and error —
  in chess or Go, nobody tells you "this move is correct," but there is a
  win/lose signal. ML1 only previews the concept; ML2 covers it in depth.

## 1.3 Regression vs. Classification: Same Model, Different Output

Supervised learning further splits by whether the output is continuous or
categorical: **regression** predicts a number (house price, temperature),
**classification** predicts a category (spam/not-spam, dog/cat/bird). The same
linear model \\(w^Tx + b\\) becomes regression if you use its output directly
(Chapter 2), or classification if you pass it through a sigmoid and read it as
a 0-1 probability (Chapter 3) — this pattern repeats for the next several
chapters: **the model structure stays similar; only how the output is
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

## 1.5 Getting Ready: Python/NumPy

This semester's coding problems use both plain Python (lists) and `numpy`
(vector/matrix operations). Here's a taste of why `numpy` matters:

```python
import numpy as np

X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 matrix
w = np.array([0.5, -0.5])               # length-2 vector
print(X @ w)   # matrix-vector product: [-0.5, -0.5]
```

`X @ w` (matrix multiplication) expresses the \\(w^Tx\\) computation that
recurs every week, in a single line — faster and more readable than writing
the loop yourself.

## 1.6 Roadmap for This Semester

Chapters 2-6 cover classical ML models that are powerful even without a neural
network (linear/logistic regression, kNN, tree-based models, GBDT) — these
still routinely beat neural networks on tabular data. From Chapter 7 onward we
move to neural networks: deriving backpropagation by hand and building
structures like CNNs on top of that principle. The last two chapters give a
taste of unsupervised learning and RL/generative models — both return in
depth in ML2.

**The question to ask before choosing any model is always the same: what data
does this problem actually have, and what am I trying to predict?**
