# Chapter 3. Logistic Regression & Classification Metrics

In 1838, Belgian mathematician Pierre François Verhulst was working on a
problem: population can't grow exponentially forever. Resources are finite, so
at some point growth must slow down and converge to some ceiling. He named
the S-shaped curve he devised to capture this the **logistic** curve. A
hundred and eighty years later, that same curve is used to solve a problem
that seems to have nothing to do with population growth: "what percent chance
is there that this email is spam?"

## 3.1 From Regression to Classification

The name "logistic regression" is misleading — this is not a regression
algorithm, it's a **classification** algorithm. Linear regression's output
\\(w^Tx\\) can be anything from \\(-\infty\\) to \\(+\infty\\), but "the
probability of being spam" must lie strictly between 0 and 1. The logistic
function (sigmoid) does exactly this conversion:

\\[\sigma(z) = \frac{1}{1+e^{-z}}\\]

As \\(z \to +\infty\\), \\(\sigma(z) \to 1\\); as \\(z \to -\infty\\),
\\(\sigma(z) \to 0\\); at \\(z=0\\), \\(\sigma(0)=0.5\\). Any real number
passed through this function lands between 0 and 1, so it can be read as a
probability.

## 3.2 The Model

\\[h_w(x) = \sigma(w^Tx) = \frac{1}{1+e^{-w^Tx}}\\]

\\(h_w(x)\\) is interpreted as "the probability that \\(x\\) belongs to the
positive class (class 1)." We predict class 1 if \\(h_w(x) \ge 0.5\\), class
0 otherwise — since \\(h_w(x)=0.5\\) is exactly the point where \\(w^Tx=0\\),
the decision boundary is still a **straight line** (or hyperplane).

## 3.3 The Loss Function: Why Not MSE?

If we apply mean squared error directly to the sigmoid, \\(J(w)\\) becomes
**non-convex** in \\(w\\), risking that gradient descent gets stuck in a local
minimum. Instead we use the **cross-entropy** loss:

\\[J(w) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log h_w(x^{(i)}) +
(1-y^{(i)}) \log(1-h_w(x^{(i)}))\right]\\]

Intuition: if the true label is \\(y=1\\) but the model confidently predicts
\\(h_w(x) \to 0\\), then \\(-\log h_w(x) \to \infty\\) — the loss blows up.
**Being confidently wrong is punished proportionally hard.** Remarkably, when
you differentiate this loss, you get **exactly the same form** of gradient as
linear regression:

\\[\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^m \left(h_w(x^{(i)})
- y^{(i)}\right) x_j^{(i)}\\]

So the gradient descent code itself is nearly identical to Chapter 2 — just
add a sigmoid where \\(h_w\\) is computed.

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def logistic_gradient_descent(X, y, alpha, epochs):
    m, n = len(X), len(X[0])
    w = [0.0] * (n + 1)
    for _ in range(epochs):
        grad = [0.0] * (n + 1)
        for i in range(m):
            pred = sigmoid(w[0] + sum(w[j+1] * X[i][j] for j in range(n)))
            error = pred - y[i]
            grad[0] += error
            for j in range(n):
                grad[j+1] += error * X[i][j]
        for j in range(n + 1):
            w[j] -= alpha * grad[j] / m
    return w
```

## 3.4 Why "Accuracy" Alone Isn't Enough

Imagine a cancer-screening model. If 99% of all patients are healthy, a model
that always predicts "healthy" boasts 99% accuracy — yet it's a useless model
that catches not a single cancer patient. Precision/Recall/F1 exist precisely
to reveal the truth that accuracy hides in situations like this (class
imbalance).

| Actual\\Predicted | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

- **Precision** \\(= \frac{TP}{TP+FP}\\): "of everything predicted positive,
  what fraction was actually positive" — how well false alarms were avoided.
- **Recall** \\(= \frac{TP}{TP+FN}\\): "of everything actually positive, what
  fraction was caught" — how few positives were missed.
- **F1** \\(= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision}
  + \text{Recall}}\\): the harmonic mean of precision and recall — it drops
  if either one is low, so a model can't score well by excelling at only one.

**The precision-recall tradeoff**: raising the threshold from 0.5 to 0.9
increases precision (only predict positive when confident) but lowers recall
(ambiguous positives get missed). Lowering the threshold does the opposite.
**PR-AUC** (the area under the precision-recall curve) summarizes performance
across the entire tradeoff without depending on any single threshold —
especially trustworthy under heavy class imbalance, where accuracy is not.

---

## Exercises

**1. (Coding)** Complete `logistic_gradient_descent` above (key lines left
blank) and the following `precision_recall_f1`:

```python
def precision_recall_f1(y_true, y_pred):
    # ADD ADDITIONAL CODE HERE!!
    # count TP, FP, FN, then compute precision, recall, f1
    # (if a denominator is 0, treat that value as 0.0)

y_true = [1, 1, 1, 0, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
print(precision_recall_f1(y_true, y_pred))  # (0.75, 0.75, 0.75)
```

**2. (Hand derivation, Tier B — hints provided)** For a single sample's
cross-entropy loss:

\\[J^{(i)}(w) = -y^{(i)}\log h_w(x^{(i)}) - (1-y^{(i)})\log(1-h_w(x^{(i)}))\\]

derive that differentiating with respect to \\(w_j\\) gives
\\(\frac{\partial J^{(i)}}{\partial w_j} = (h_w(x^{(i)}) - y^{(i)})x_j^{(i)}\\).

**Hint** (apply the chain rule in three steps): (1) First find
\\(\frac{\partial J^{(i)}}{\partial h}\\) (where \\(h\\) abbreviates
\\(h_w(x^{(i)})\\)), using \\(\frac{d}{dh}\log h = \frac{1}{h}\\). (2) Then use
\\(\sigma'(z) = \sigma(z)(1-\sigma(z))\\) to find \\(\frac{\partial
h}{\partial z}\\) (where \\(z=w^Tx^{(i)}\\)). (3) Using \\(\frac{\partial
z}{\partial w_j} = x_j^{(i)}\\), multiply the three pieces together via the
chain rule — remarkably, the \\(h(1-h)\\) term cancels out entirely. Check
why that happens, and confirm that the result you derived has exactly the
same form as Chapter 2's linear regression gradient.
