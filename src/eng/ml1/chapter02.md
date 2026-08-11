# Chapter 2. Linear Regression

In 1805, mathematician Adrien-Marie Legendre was trying to fit a comet's orbit to
noisy observational data. Since every measurement carries some error, no single
curve passes through every point exactly — what's needed instead is the curve
that fits the data best overall. His answer — **find the line that minimizes the
sum of squared errors** — became known as the method of least squares, and it's
still used everywhere from spreadsheet trendlines to large-scale recommender
systems. This chapter develops that idea formally.

## 2.1 The Model

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
