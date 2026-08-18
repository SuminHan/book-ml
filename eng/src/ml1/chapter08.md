# Chapter 8. Neural Network Basics & Backpropagation

In 1969, Marvin Minsky and Seymour Papert proved mathematically, in their
book *Perceptrons*, that a single-layer perceptron cannot solve even
**XOR**. XOR is a very simple rule ("output 1 if the two inputs differ, 0 if
they're the same"), but its decision boundary cannot be drawn with a single
straight line — a problem fundamentally unsolvable by logistic regression
(a model that separates with one line), which we covered earlier. This
result sharply curtailed investment in neural network research and became
one cause of the decade-plus "AI winter" that followed.

## 8.1 From Perceptron to Multi-Layer Network

The fix turned out to be remarkably simple: if one line isn't enough,
**stack several lines to build a new space, then separate again in that new
space.** Logistic regression can actually be viewed as the simplest possible
"neural network," with only an input and output layer (no hidden layer):
\\(a = \sigma(w^Tx)\\). A **Multi-Layer Perceptron (MLP)** inserts one or
more hidden layers in between — adding just a single hidden layer is enough
to build a boundary that solves XOR perfectly.

## 8.2 Forward Propagation

For a two-layer network (input → hidden → output) with input \\(x\\),
hidden layer weights \\(W_1, b_1\\), and output layer weights \\(W_2,
b_2\\), the forward pass is:

\\[z_1 = W_1 x + b_1, \quad a_1 = \sigma(z_1), \quad z_2 = W_2^T a_1 + b_2,
\quad a_2 = \sigma(z_2)\\]

\\(z\\) is the value before applying the activation function
(pre-activation), \\(a\\) is after (activation). The final output \\(a_2\\)
is the prediction.

## 8.3 Backpropagation: Running the Chain Rule Backwards

Adding a hidden layer creates a new problem: how do we train that hidden
layer's weights? The output layer's gradient can be computed exactly as in
Chapter 2's logistic regression, but the hidden layer has no direct target
to compare against. The **backpropagation** algorithm, popularized in 1986
by David Rumelhart, Geoffrey Hinton, and Ronald Williams, solved this: flow
the output layer's error backward through the chain rule, computing exactly
how responsible each hidden-layer weight is for the final error.

Given a loss function \\(L\\), we need \\(\frac{\partial L}{\partial W}\\)
for every weight in order to apply gradient descent. The problem is that
\\(W_1\\)'s effect on the final loss travels through the long chain
\\(z_1 \to a_1 \to z_2 \to a_2 \to L\\). The chain rule tells us we can
follow this chain **backward, from output to input**, multiplying
derivatives one link at a time:

1. **Output layer error**: \\(\delta_2 = \frac{\partial L}{\partial z_2} =
   (a_2 - y) \sigma'(z_2)\\) (the same form we saw in Chapter 2 — the
   combination of cross-entropy and sigmoid always simplifies this
   cleanly.)
2. **Hidden layer error**: \\(\delta_1 = (W_2 \delta_2) \odot
   \sigma'(z_1)\\) — the output layer's error \\(\delta_2\\) is "sent back"
   through \\(W_2\\) to the hidden layer, then multiplied by the hidden
   layer's own derivative \\(\sigma'(z_1)\\). (\\(\odot\\) is elementwise
   multiplication.)
3. **Gradients**: \\(\frac{\partial L}{\partial W_2} = \delta_2 \cdot
   a_1^T\\), \\(\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T\\)

This is exactly where the name "backpropagation" comes from: the error
(\\(\delta\\)) is computed at the output layer, and that value propagates
**backward** toward the hidden layer, generating the gradient for each
layer along the way.

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def two_layer_forward(x, W1, b1, W2, b2):
    z1 = [sum(W1[i][j] * x[j] for j in range(len(x))) + b1[i] for i in range(len(b1))]
    a1 = [sigmoid(v) for v in z1]
    z2 = sum(W2[i] * a1[i] for i in range(len(a1))) + b2
    a2 = sigmoid(z2)
    return a2, (x, z1, a1, z2, a2)

def two_layer_backward(y_true, cache, W2):
    x, z1, a1, z2, a2 = cache
    delta2 = (a2 - y_true) * sigmoid_prime(z2)
    delta1 = [W2[i] * delta2 * sigmoid_prime(z1[i]) for i in range(len(z1))]
    grad_W2 = [delta2 * a1[i] for i in range(len(a1))]
    grad_b2 = delta2
    grad_W1 = [[delta1[i] * x[j] for j in range(len(x))] for i in range(len(delta1))]
    grad_b1 = delta1
    return grad_W1, grad_b1, grad_W2, grad_b2
```

## 8.4 Why Do This "Without Autograd," by Hand

PyTorch and TensorFlow compute all these derivatives automatically with a
single line, `.backward()` (automatic differentiation). But if you don't
know exactly what that automatic differentiation is doing internally,
there's no way to diagnose the cause when training diverges or you hit the
vanishing gradient problem (covered in Chapter 9). The point of this
chapter's exercises is to walk through, by hand, the exact chain rule that
the library normally does for you — just once.

**A neural network's ability to "learn anything" comes from stacking layers
to increase its expressive power; its ability to be "actually trained" comes
down to a single idea: the chain rule.**

---

## Exercises

**1. (Coding)** Complete the forward and backward pass (key lines left
blank) for a two-layer network with input (2) → hidden (2, sigmoid) →
output (1, sigmoid):

```python
def two_layer_nn_forward(x, W1, b1, W2, b2):
    # ADD ADDITIONAL CODE HERE!!
    # hidden pre-activation z1 = W1 @ x + b1, activation a1 = sigmoid(z1)
    # output pre-activation z2 = W2 . a1 + b2, activation a2 = sigmoid(z2)

    return a2, (x, z1, a1, z2, a2)

def two_layer_nn_backward(y_true, cache, W2):
    x, z1, a1, z2, a2 = cache
    # ADD ADDITIONAL CODE HERE!!
    # output error delta2, hidden error delta1, gradients for W2/b2/W1/b1

    return grads
```

**2. (Conceptual)** State exactly which Chapter 2 model a neural network
with **no** hidden layer (input directly to sigmoid output) reduces to,
and explain in two or three sentences why adding a hidden layer increases
expressive power (using the XOR example).

**3. (Hand derivation, Tier C — fallback prepared)** Starting from the loss
function \\(L = \frac{1}{2}(a_2 - y)^2\\) for the network above, derive
\\(\frac{\partial L}{\partial W_1}\\) and \\(\frac{\partial L}{\partial
W_2}\\) from start to finish, **using the chain rule alone**.

**Fill-in-the-blank fallback version** (if free derivation is too
difficult):

```
L = (1/2)(a2 - y)^2

Step 1: dL/da2 = ______________
Step 2: da2/dz2 = a2(1-a2)  [sigmoid derivative formula -- already given]
Step 3: dL/dz2 = dL/da2 * da2/dz2 = ______________  (this is delta2)
Step 4: dz2/dW2 = a1
Step 5: dL/dW2 = delta2 * ______________

Step 6: dz2/da1 = W2
Step 7: dL/da1 = delta2 * ______________
Step 8: da1/dz1 = a1(1-a1)
Step 9: dL/dz1 = dL/da1 * da1/dz1 = ______________  (this is delta1)
Step 10: dL/dW1 = delta1 * ______________  (outer product with x)
```

**Confirm correctness**: connect each line of your finished derivation to
the corresponding blank in the code above, one sentence each.
