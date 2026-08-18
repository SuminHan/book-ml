# Chapter 6. Regularization & Model Selection

In 1996, statistician Robert Tibshirani proposed a method called the
"Lasso" (Least Absolute Shrinkage and Selection Operator). The idea was
simple — add the sum of the absolute values of the weights as a penalty to
a regression's loss function, and remarkably, the weights on unimportant
features shrink to **exactly zero**. Instead of a person manually picking
which of thousands of features actually matter, changing a single loss
function lets the model filter them out on its own. This chapter covers
the actual **dial** for controlling the bias-variance tradeoff we saw in
Chapter 4.4.

## 6.1 Regularization: Adding a Penalty to the Loss Function

As we saw in Chapter 4.4, when a model is too flexible (large, unconstrained
parameters), variance increases and it overfits. **Regularization** adds a
penalty term to the loss function that discourages weights from growing too
large, artificially reducing the model's effective flexibility:

\\[J(w) = \underbrace{\frac{1}{2m}\sum_{i=1}^m (h_w(x^{(i)})-y^{(i)})^2}\_{\text{original loss (fit)}} +
\underbrace{\lambda R(w)}\_{\text{regularization term (demands simplicity)}}\\]

When \\(\lambda\\) (regularization strength) is 0, this is just the
original loss function (no regularization); as \\(\lambda\\) grows, the
model cares more about "keeping the weights small" than "fitting the data
exactly" — in Chapter 4.4's language, raising \\(\lambda\\) trades variance
down for bias up.

## 6.2 L2 (Ridge) and L1 (Lasso): the Shape of the Penalty Matters

The two most common penalties are:

- **L2 regularization (Ridge)**: \\(R(w) = \|w\|_2^2 = \sum_j w_j^2\\)
- **L1 regularization (Lasso)**: \\(R(w) = \|w\|_1 = \sum_j |w_j|\\)

Both aim to "keep the weights small," but **the shape produces different
results**. L2 smoothly shrinks all weights toward 0, but they rarely land
exactly on 0. L1 tends to push unimportant weights to **exactly** 0 —
meaning Lasso does automatic **feature selection** as a side effect of
regularization.

**Geometric intuition**: rewriting this as minimizing the original loss
subject to a constraint \\(R(w) \le t\\), L2's constraint region is a
circle (or sphere), while L1's is a diamond (a polytope with sharp corners
sitting on the axes). When the loss function's contours meet this region at
an optimum, they're far more likely to meet at a corner (where some
coordinate is exactly 0) with the diamond than with the circle — that's the
geometric reason L1 produces exact zeros and L2 doesn't.

```python
def ridge_gradient_descent(X, y, lam, alpha, epochs):
    m, n = len(X), len(X[0])
    w = [0.0] * (n + 1)  # w[0] = bias
    for _ in range(epochs):
        grad = [0.0] * (n + 1)
        for i in range(m):
            pred = w[0] + sum(w[j+1] * X[i][j] for j in range(n))
            error = pred - y[i]
            grad[0] += error
            for j in range(n):
                grad[j+1] += error * X[i][j]
        for j in range(n + 1):
            reg = lam * w[j] if j > 0 else 0  # bias (w[0]) is conventionally excluded
            w[j] -= alpha * (grad[j] / m + reg)
    return w
```

Increasing `lam` (\\(\lambda\\)) from 0 upward, you can directly watch the
learned weights shrink toward 0 — feeding the same data through
\\(\lambda=0, 0.5, 5.0\\), the slope \\(w_1\\) shrinks from roughly
\\(2.0 \to 1.4 \to 0.4\\).

## 6.3 Cross-Validation: Letting the Data Choose \\(\lambda\\)

\\(\lambda\\) is a **hyperparameter** — unlike \\(w\\), which gets set by
training, a hyperparameter needs to be chosen before training even starts.
Choosing \\(\lambda\\) by looking only at training loss always favors
\\(\lambda=0\\) (no regularization), which defeats the purpose — you need
to measure performance on a **validation set**.

When data is scarce, **k-fold cross-validation** is used: split the data
into \\(k\\) parts, and \\(k\\) times, hold out one part for validation
while training on the rest, then average the performance — every data
point gets used for both training and validation at some point.

```python
def k_fold_split(data, k, fold_idx):
    n = len(data)
    fold_size = n // k
    start = fold_idx * fold_size
    end = start + fold_size if fold_idx < k - 1 else n
    val = data[start:end]
    train = data[:start] + data[end:]
    return train, val
```

**The principle of model selection**: use the training data to fit
parameters (\\(w\\)), use the validation data to choose hyperparameters
(\\(\lambda\\), \\(k\\) in kNN, tree depth, etc.), and use the **test
data** only once, at the very end, to check final performance — using the
test data to tune hyperparameters is no different from cheating (peeking
at the test data to pick your model). This three-way split
(train/validation/test) discipline applies just as strictly to ML2's team
project.

**Regularization is the tool that turns Chapter 4.4's bias-variance
principle — "make the model less flexible to reduce variance" — into
something you can actually dial in, just by adding a term to the loss
function. And that dial itself (\\(\lambda\\)) gets set by yet another
procedure: cross-validation.**

---

## Exercises

**1. (Coding)** Complete `ridge_gradient_descent` and `k_fold_split` above
(key lines left blank):

```python
def ridge_gradient_descent(X, y, lam, alpha, epochs):
    # ADD ADDITIONAL CODE HERE!!
    # Chapter 2's gradient_descent, plus an L2 regularization term (excluding bias)

def k_fold_split(data, k, fold_idx):
    # ADD ADDITIONAL CODE HERE!!
    # split data into k parts, return fold_idx as validation and the rest as training

X = [[1.0],[2.0],[3.0],[4.0]]
y = [3.0,5.0,7.0,9.0]
print(ridge_gradient_descent(X, y, lam=0.0, alpha=0.01, epochs=2000))  # approximately [1.0, 2.0]
print(ridge_gradient_descent(X, y, lam=5.0, alpha=0.01, epochs=2000))  # w[1] pushed toward 0

print(k_fold_split(list(range(10)), k=5, fold_idx=2))  # ([...], [4, 5])
```

**2. (Conceptual)** In gene-expression data with 10,000 features, where
only about 20 features are believed to actually affect the outcome, which
would you use — L1 or L2 — and why?

**3. (Hand derivation, Tier A — free derivation)** Starting from the L2-
regularized linear regression (Ridge regression) cost function \\(J(w) =
\frac{1}{2m}\|Xw-y\|^2 + \frac{\lambda}{2}\|w\|^2\\), differentiate with
respect to \\(w\\), set the result to zero, and derive that the closed-form
solution is

\\[w^* = (X^TX + \lambda I)^{-1}X^Ty\\]

(reuse Chapter 2's normal-equation derivation, adding the derivative of the
regularization term \\(\nabla_w \frac{\lambda}{2}\|w\|^2 = \lambda w\\)).
Confirm that as \\(\lambda \to 0\\), this reduces to Chapter 2's normal
equation \\(w^*=(X^TX)^{-1}X^Ty\\), and explain in one sentence what value
\\(w^*\\) approaches as \\(\lambda \to \infty\\).
