# Chapter 2. Regression Models: Linear & Logistic

In 1805, mathematician Adrien-Marie Legendre was trying to fit a comet's orbit to
noisy observational data. Since every measurement carries some error, no single
curve passes through every point exactly — what's needed instead is the curve
that fits the data best overall. His answer — **find the line that minimizes the
sum of squared errors** — became known as the method of least squares, and it's
still used everywhere from spreadsheet trendlines to large-scale recommender
systems. A hundred and eighty years later, the S-shaped "logistic" curve a
Belgian mathematician devised to describe population growth would end up
solving a completely different problem: "what percent chance is there that
this email is spam?" This chapter tells both stories as one — **the same
linear model \\(w^Tx\\), with the output interpreted differently and the
loss function swapped, gives you both regression and classification.**

## 2.1 Linear Regression: The Model

Linear regression predicts an output from an input \\(x = (x_1, \ldots, x_n)\\)
as follows:

\\[h_w(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n\\]

\\(w_0\\) is the intercept (bias), and \\(w_1, \ldots, w_n\\) are the weights on
each feature. To simplify notation, we always append \\(x_0 = 1\\), so we can
write \\(h_w(x) = w^Tx = \sum_{j=0}^n w_j x_j\\) — folding the bias into a single
dot product.

## 2.2 The Cost Function: Mean Squared Error

For \\(m\\) training examples \\((x^{(i)}, y^{(i)})\\), we define how wrong the
model is as:

\\[J(w) = \frac{1}{2m}\sum_{i=1}^m \left(h_w(x^{(i)}) - y^{(i)}\right)^2\\]

Why square the errors and add them up? If we just add the raw errors, positive
and negative errors cancel out, so even a terrible model could show an "average
error of 0." Taking absolute values avoids that cancellation but makes the
function non-differentiable at zero, which complicates optimization. Squaring
solves both problems at once: it's always positive, and it's smoothly
differentiable. The leading \\(\frac{1}{2}\\) is a conventional constant added
so it cancels with the exponent 2 during differentiation later — it doesn't
change where the minimum is.

## 2.3 Gradient Descent

To find the \\(w\\) that minimizes \\(J(w)\\), we take small steps in the
**opposite** direction of the gradient (the direction of steepest increase):

\\[w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j}, \qquad
\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^m \left(h_w(x^{(i)}) -
y^{(i)}\right) x_j^{(i)}\\]

\\(\alpha\\) is the **learning rate** — the size of each step. We update every
\\(w_j\\) simultaneously and repeat until the loss is small enough (or for a
fixed number of iterations).

```python
def gradient_descent(X, y, alpha, epochs):
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
            w[j] -= alpha * grad[j] / m
    return w
```

**The learning-rate trap**: if \\(\alpha\\) is too small, convergence is slow.
If \\(\alpha\\) is too large, it can overshoot near the minimum and diverge
instead — like trying to reach the bottom of a valley but taking a step so big
you end up on the opposite wall.

## 2.4 The Normal Equation: Solving It in One Shot

Since \\(J(w)\\) is a quadratic function of \\(w\\), we can also set its
derivative to zero and solve directly, without gradient descent, to get a
**closed-form** solution:

\\[w^* = (X^TX)^{-1}X^Ty\\]

where \\(X\\) is the \\(m \times (n+1)\\) matrix whose rows are the
\\(x^{(i)}\\) (including the bias term). Deriving this is the centerpiece of
this chapter's exercises.

**Gradient descent vs. the normal equation**: the normal equation requires no
iteration and is exact, but computing \\((X^TX)^{-1}\\) costs time proportional
to the cube of the number of features \\(n\\) — infeasible when \\(n\\) is in
the thousands or more, as in many real problems. Gradient descent, by
contrast, scales only linearly in \\(n\\) per iteration. That's why the normal
equation is used when features are few, and gradient descent when they're
many.

## 2.5 From Regression to Classification

From here on, we put the same linear model to work on **classification**.
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

\\[h_w(x) = \sigma(w^Tx) = \frac{1}{1+e^{-w^Tx}}\\]

\\(h_w(x)\\) is interpreted as "the probability that \\(x\\) belongs to the
positive class (class 1)." We predict class 1 if \\(h_w(x) \ge 0.5\\), class
0 otherwise — since \\(h_w(x)=0.5\\) is exactly the point where \\(w^Tx=0\\),
the decision boundary is still a **straight line** (or hyperplane).

## 2.6 The Loss Function: From Shannon's Information Theory to Cross-Entropy

If we apply mean squared error directly to the sigmoid, \\(J(w)\\) becomes
**non-convex** in \\(w\\), risking that gradient descent gets stuck in a local
minimum. Instead we use the **cross-entropy** loss — a name that comes from
the **information theory** Claude Shannon founded in 1948.

Shannon asked how to measure "information" mathematically. If an event with
probability \\(p\\) occurs, how much "surprise" (information) does that news
carry? He defined it as \\(-\log_2 p\\) (in bits) — a frequent event
(\\(p\\) near 1) carries almost no information, like "the sun rose in the
east," while a rare event (\\(p\\) near 0) carries a lot, like "I won the
lottery."

The **expected value** of this information content is **entropy**:

\\[H(p) = -\sum_k p_k \log_2 p_k\\]

It's the average information you get from observing events drawn from
distribution \\(p\\), and equivalently the average number of bits needed to
optimally encode that distribution (we'll meet it again in Chapter 7.3, as
the criterion decision trees use to pick "the best splitting question").

But what if the true distribution is \\(p\\), while we encode assuming a
different (possibly wrong) distribution \\(q\\)? The average number of bits
that takes is the **cross-entropy**:

\\[H(p, q) = -\sum_k p_k \log_2 q_k\\]

When \\(q\\) exactly matches \\(p\\), \\(H(p,q) = H(p)\\) — its **minimum**.
The further \\(q\\) drifts from \\(p\\) (the more wrong the assumed
distribution is), the larger \\(H(p,q)\\) grows — that excess,
\\(H(p,q) - H(p)\\), is called **KL divergence**
(\\(D_{KL}(p\|q)\\)), which we'll meet again in Chapter 15's ELBO for VAEs.

In logistic regression, the true label \\(y^{(i)}\\) is a "true distribution"
(100% probability on one class, 0% on the other), and \\(h_w(x^{(i)})\\) is
the distribution the model predicts. **Minimizing cross-entropy means
pushing the model's predicted distribution as close as possible to the true
distribution** — that's exactly why it makes a good loss function:

\\[J(w) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log h_w(x^{(i)}) +
(1-y^{(i)}) \log(1-h_w(x^{(i)}))\right]\\]

(Machine learning usually uses the natural log \\(\log = \ln\\) instead of
\\(\log_2\\) — the two differ only by a constant factor \\(1/\ln 2\\), which
doesn't change where the loss is minimized.)

![Sigmoid function (left) and cross-entropy loss vs. predicted probability (right) — the loss blows up as a confident prediction gets further from the true label](../images/ch03_sigmoid_crossentropy.svg)

Intuition: if the true label is \\(y=1\\) but the model confidently predicts
\\(h_w(x) \to 0\\), then \\(-\log h_w(x) \to \infty\\) — the loss blows up.
**Being confidently wrong is punished proportionally hard.** Remarkably, when
you differentiate this loss, you get **exactly the same form** of gradient as
the linear regression in section 2.3:

\\[\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^m \left(h_w(x^{(i)}) - y^{(i)}\right) x_j^{(i)}\\]

So the gradient descent code itself is nearly identical to section 2.3 — just
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

## 2.7 Why "Accuracy" Alone Isn't Enough

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
- **F1** \\(= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}\\):
  the harmonic mean of precision and recall — it drops if either one is low,
  so a model can't score well by excelling at only one.

**The precision-recall tradeoff**: raising the threshold from 0.5 to 0.9
increases precision (only predict positive when confident) but lowers recall
(ambiguous positives get missed). Lowering the threshold does the opposite.
**PR-AUC** (the area under the precision-recall curve) summarizes performance
across the entire tradeoff without depending on any single threshold —
especially trustworthy under heavy class imbalance, where accuracy is not.

**Linear and logistic regression look like different problems (predicting a
number vs. predicting a probability) on the surface, but underneath they're
the exact same pipeline: linear model -> define a loss function -> minimize
it with gradient descent. That pattern repeats all semester.**

---

## Exercises

**1. (Coding)** Complete the `gradient_descent` function above (the four key
lines are left blank):

```python
def gradient_descent(X, y, alpha, epochs):
    # ADD ADDITIONAL CODE HERE!!
    # initialize w to a zero vector, length len(X[0])+1 (including bias)

    for epoch in range(epochs):
        # ADD ADDITIONAL CODE HERE!!
        # compute predictions h_w(x) = w^T x, compute the gradient, update w

X = [[1.0], [2.0], [3.0], [4.0]]
y = [3.0, 5.0, 7.0, 9.0]
print(gradient_descent(X, y, alpha=0.01, epochs=1000))  # approximately [1.0, 2.0]
```

**2. (Hand derivation, Tier A — free derivation)** Starting from the linear
regression cost function \\(J(w) = \frac{1}{2m}\|Xw - y\|^2\\), differentiate
with respect to \\(w\\), set the result to zero, and derive

\\[w^* = (X^TX)^{-1}X^Ty\\]

**from start to finish, by hand.** (Hint: first show that
\\(\nabla_w \|Xw-y\|^2 = 2X^T(Xw-y)\\), then set this to zero and solve for
\\(w\\).) Then argue that the \\(w^*\\) you derived is in fact the **global
minimum** of \\(J(w)\\) (by showing the Hessian is positive semi-definite).

**3. (Coding)** Complete `logistic_gradient_descent` above (key lines left
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

**4. (Hand derivation, Tier B — hints provided)** For a single sample's
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
same form as section 2.3's linear regression gradient.
