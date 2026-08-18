# Chapter 5. Support Vector Machines & Kernels

In the 1990s, Vladimir Vapnik and Corinna Cortes at Bell Labs asked a
completely different kind of question. Logistic regression uses "how well
does it predict probability" as its loss, and GDA/Naive Bayes model "how
was the data generated." Vapnik and Cortes ignored both probability and
generative process, and asked a purely **geometric** question instead: "of
the countless possible boundaries separating two classes, which one is
safest for new data?" The answer to that question is the **Support Vector
Machine** (SVM) — before deep learning took over in the 2010s, it was the
most widely used classifier, from handwriting recognition to text
classification.

## 5.1 Maximizing the Margin, and Support Vectors

Consider data that's perfectly linearly separable into two classes. There
are infinitely many lines (hyperplanes) that separate them — but some lines
hug one class dangerously closely, while others leave generous room on
both sides. SVM picks the line with the largest **margin** (the distance
from the boundary to the nearest data point) — the boundary least likely
to misclassify new data that looks slightly different.

Writing the decision boundary as \\(w^Tx+b=0\\), the constraint that every
point sits on the correct side with at least margin 1 is:

\\[y^{(i)}(w^Tx^{(i)}+b) \ge 1, \qquad y^{(i)} \in \{-1, +1\}\\]

(Unlike logistic regression, SVM conventionally uses \\(-1/+1\\) labels
instead of \\(0/1\\).) Under this constraint, the margin's size works out to
\\(\frac{2}{\|w\|}\\), so maximizing the margin is equivalent to
**minimizing** \\(\|w\|\\). The data points that satisfy this constraint
with equality (closest to the boundary) are called **support vectors** —
they're the only points that "hold up" the boundary's position. Every other
point could be moved or deleted without changing the result at all — a
sharp contrast with logistic regression, where **every** data point
contributes to the loss.

## 5.2 The Soft Margin: When Data Isn't Perfectly Separable

Real data usually isn't perfectly linearly separable because of noise. The
**soft margin** SVM introduces a slack variable \\(\xi^{(i)} \ge 0\\) per
data point, allowing some amount of violation:

\\[y^{(i)}(w^Tx^{(i)}+b) \ge 1 - \xi^{(i)}\\]

The overall objective becomes a tradeoff between "I want to maximize the
margin" (minimize \\(\|w\|\\)) and "I want to minimize violations"
(minimize \\(\sum \xi^{(i)}\\)):

\\[J(w,b) = \frac{1}{2}\|w\|^2 + C\sum_{i=1}^m \xi^{(i)}\\]

This objective can actually be rewritten as a sum of **hinge losses**
\\(\max(0, 1-y^{(i)}(w^Tx^{(i)}+b))\\) — zero loss if the margin is
satisfied (\\(\ge 1\\)), and loss growing linearly with the violation
otherwise. \\(C\\) is the hyperparameter controlling "how wide a margin do
I want vs. how little violation will I tolerate": large \\(C\\) is stricter
about violations (risking overfitting), small \\(C\\) allows a wider margin
at the cost of tolerating more violations (risking underfitting) — exactly
the same kind of tradeoff we'll adjust with regularization strength in
Chapter 6.

```python
def hinge_loss_gradient_descent(X, y, C, alpha, epochs):
    # y[i] is either -1 or +1
    m, n = len(X), len(X[0])
    w, b = [0.0] * n, 0.0
    for _ in range(epochs):
        grad_w, grad_b = [0.0] * n, 0.0
        for i in range(m):
            margin = y[i] * (sum(w[j] * X[i][j] for j in range(n)) + b)
            if margin < 1:  # only margin-violating points contribute to the gradient
                for j in range(n):
                    grad_w[j] += -y[i] * X[i][j]
                grad_b += -y[i]
        for j in range(n):
            grad_w[j] = w[j] + C * grad_w[j] / m  # gradient of the ||w||^2 term + hinge term
        grad_b = C * grad_b / m
        for j in range(n):
            w[j] -= alpha * grad_w[j]
        b -= alpha * grad_b
    return w, b
```

## 5.3 The Kernel Trick: When Data Isn't Linearly Separable

Sometimes two classes are tangled together in a circular pattern in the
original feature space, and no line can separate them. Moving such data
into a **higher-dimensional** space (e.g., \\(x=(x_1,x_2) \to
\phi(x)=(x_1,x_2,x_1^2+x_2^2)\\)) can make it separable by a flat
hyperplane in that space — a "curved boundary in 2D" becomes a "flat
boundary in 3D."

The problem is that computing \\(\phi(x)\\) explicitly can blow up in
dimension beyond what's manageable. The **kernel trick** sidesteps this —
rewriting SVM's optimization problem in its (Lagrangian) dual form reveals
that the data only ever appears as \\(\phi(x^{(i)}) \cdot \phi(x^{(j)})\\)
(a dot product). If that's true, we just need a **kernel function**
\\(K(x,x') = \phi(x)\cdot\phi(x')\\) that computes this dot product
directly, without ever computing \\(\phi\\) itself.

The most widely used kernels:

- **Polynomial kernel**: \\(K(x,x') = (x \cdot x' + c)^d\\)
- **RBF (Gaussian) kernel**: \\(K(x,x') = \exp\left(-\frac{\|x-x'\|^2}{2\sigma^2}\right)\\)
  — a "similarity" function that's close to 1 when two points are near
  each other and close to 0 when they're far apart. This kernel is known
  to correspond to an **infinite-dimensional** \\(\phi\\), yet the kernel
  function itself can still be computed from nothing more than Euclidean
  distance.

```python
import math

def rbf_kernel(x1, x2, sigma):
    sq_dist = sum((x1[i] - x2[i]) ** 2 for i in range(len(x1)))
    return math.exp(-sq_dist / (2 * sigma ** 2))
```

**Intuition**: prediction with a kernel SVM ends up being something like a
weighted vote of "how similar is this new point to each support vector"
(the kernel value). A small \\(\sigma\\) weighs only very close support
vectors heavily (complex boundary, risk of overfitting); a large
\\(\sigma\\) weighs distant support vectors too (simpler boundary, risk of
underfitting). Actually training a kernel SVM (solving the dual problem via
quadratic programming) is beyond this book's scope, but the idea of
"sidestepping a high-dimensional transformation with a single kernel
function" left a deep mark on how nonlinear models were designed in the
era before neural networks.

**SVM solves exactly the same problem as logistic regression and GDA
(finding a linear decision boundary), but starts from an entirely
different principle — "maximize the margin" instead of "model a
probability." Even when the destination looks similar, a different
starting principle yields different properties (support vectors, the
kernel trick).**

---

## Exercises

**1. (Coding)** Complete `hinge_loss_gradient_descent` above (key lines
left blank):

```python
def hinge_loss_gradient_descent(X, y, C, alpha, epochs):
    # ADD ADDITIONAL CODE HERE!!
    # initialize w, b to zero; for each epoch, compute each sample's margin
    # y[i]*(w.x[i]+b) and accumulate the gradient when it's less than 1,
    # then add the ||w||^2 term's gradient (=w) and update w, b

X = [[3,3],[4,3],[3,4],[-3,-3],[-4,-3],[-3,-4]]
y = [1,1,1,-1,-1,-1]
w, b = hinge_loss_gradient_descent(X, y, C=1.0, alpha=0.01, epochs=2000)
print(w, b)  # approximately [0.17, 0.17], 0.0 -- a boundary symmetric through the origin
```

**2. (Conceptual)** In the following two scenarios, explain with reasons
whether \\(C\\) should be large or small: (a) you suspect the training data
contains some mislabeled examples; (b) the training data is very clean and
you're confident new data will follow a similar distribution.

**3. (Hand derivation, Tier B — hints provided)** Show that the margin
equals \\(\frac{2}{\|w\|}\\). (Hint: let \\(x_0\\) be a point on the
decision boundary, and \\(x_+\\) be the closest positive support vector, so
that \\(w^Tx_+ + b = 1\\) and \\(w^Tx_0+b=0\\). Subtract the two equations
to get \\(w^T(x_+-x_0)=1\\), and use the fact that \\(x_+-x_0\\) points in
the direction parallel to \\(w\\) (the shortest-distance direction
perpendicular to the boundary) to derive \\(\|x_+-x_0\| =
\frac{1}{\|w\|}\\).) Explain why the full margin width is twice this
distance (positive and negative support vectors on either side), and state
in one sentence why "minimizing \\(\|w\|\\)" is the same thing as
"maximizing the margin."
