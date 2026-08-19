# Chapter 9. Neural Network Basics, Backpropagation & Training Techniques

In 1969, Marvin Minsky and Seymour Papert proved mathematically, in their
book *Perceptrons*, that a single-layer perceptron cannot solve even
**XOR**. XOR is a very simple rule ("output 1 if the two inputs differ, 0 if
they're the same"), but its decision boundary cannot be drawn with a single
straight line — a problem fundamentally unsolvable by logistic regression
(a model that separates with one line), which we covered earlier. This
result sharply curtailed investment in neural network research and became
one cause of the decade-plus "AI winter" that followed. This chapter covers
how that problem was solved (hidden layers + backpropagation), and the new
problem that appears once you stack layers deep (vanishing gradients) and
how to tame it.

## 9.1 From Perceptron to Multi-Layer Network

The fix turned out to be remarkably simple: if one line isn't enough,
**stack several lines to build a new space, then separate again in that new
space.** Logistic regression can actually be viewed as the simplest possible
"neural network," with only an input and output layer (no hidden layer):
\\(a = \sigma(w^Tx)\\). A **Multi-Layer Perceptron (MLP)** inserts one or
more hidden layers in between — adding just a single hidden layer is enough
to build a boundary that solves XOR perfectly.

## 9.2 Forward Propagation

For a two-layer network (input → hidden → output) with input \\(x\\),
hidden layer weights \\(W_1, b_1\\), and output layer weights \\(W_2,
b_2\\), the forward pass is:

\\[z_1 = W_1 x + b_1, \quad a_1 = \sigma(z_1), \quad z_2 = W_2^T a_1 + b_2,
\quad a_2 = \sigma(z_2)\\]

\\(z\\) is the value before applying the activation function
(pre-activation), \\(a\\) is after (activation). The final output \\(a_2\\)
is the prediction.

## 9.3 Backpropagation: Running the Chain Rule Backwards

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

PyTorch and TensorFlow compute all these derivatives automatically with a
single line, `.backward()` (automatic differentiation). But if you don't
know exactly what that automatic differentiation is doing internally,
there's no way to diagnose the cause when you hit the vanishing gradient
problem below (Section 9.5). The point of this chapter's exercises is to
walk through, by hand, the exact chain rule that the library normally does
for you — just once.

## 9.4 The Sigmoid's Hidden Trap: Vanishing Gradients

Backpropagation can, in theory, compute gradients no matter how many layers
there are. But when you actually stack 10 or 20 layers deep, something
strange happens: the weights of layers near the input barely learn at all.
This is one reason neural networks stayed "shallow" (2-3 layers) for so
long — the paradox that going deeper actually makes training worse.

The cause lies in the sigmoid function itself. \\(\sigma'(z) =
\sigma(z)(1-\sigma(z))\\) has a maximum of only 0.25, at \\(z=0\\).
Backpropagation **multiplies** this derivative every time it climbs back up
a layer (think of Section 9.3's \\(\delta_1 = (W_2\delta_2) \odot
\sigma'(z_1)\\) repeating at every layer), so with 10 layers, the signal
that reaches the first layer is at best \\(0.25^{10} \approx 0.00000095\\)
of its original size — effectively zero. This is the **vanishing
gradient** problem. Conversely, if weights are initialized badly, the
gradient can instead grow explosively larger with each layer — the
**exploding gradient** problem.

In a network with \\(L\\) layers, the chain rule gives the gradient with
respect to the first layer's weights this form (a product of each layer's
activation derivative and weights):

\\[\frac{\partial L}{\partial W_1} \propto \prod_{l=2}^{L} \sigma'(z_l) \cdot W_l\\]

- If each \\(\sigma'(z_l) < 1\\) (as with sigmoid), this product shrinks
  exponentially toward 0 as the number of layers grows — **vanishing
  gradient**.
- If each term is greater than 1 (e.g., weights initialized too large), the
  product grows exponentially — **exploding gradient**.

## 9.5 Activation Functions

The activation functions and regularization techniques (dropout, batch
normalization) covered from here on look like unrelated tricks on the
surface, but most of them are really different angles on solving this one
problem: "how do we keep the gradient alive as it travels through a deep
network?"

| Function | Definition | Derivative | Notes |
|---|---|---|---|
| Sigmoid | \\(\frac{1}{1+e^{-z}}\\) | \\(\sigma(z)(1-\sigma(z))\\), max 0.25 | Output in (0,1) — useful for probabilities, but prone to vanishing gradients |
| Tanh | \\(\frac{e^z-e^{-z}}{e^z+e^{-z}}\\) | max 1 | Output in (-1,1), better than sigmoid but still saturates at both ends |
| ReLU | \\(\max(0, z)\\) | 1 if \\(z>0\\), 0 if \\(z<0\\) | Derivative is always 1 in the positive region — doesn't shrink the gradient across layers |

ReLU's derivative is either 0 or 1, so multiplying by it never shrinks
anything. It does have the "dying ReLU" problem — the derivative is exactly
0 when \\(z<0\\), so a neuron can get stuck — but it's simple to compute
and has no vanishing-gradient issue in the positive region, which is why
it's the most common default choice today. With ReLU, activated neurons
have \\(\sigma'(z_l)=1\\), so the product above is at least never shrunk by
the activation function itself — though the size of the weights \\(W_l\\)
themselves remains a separate concern (which is why the initialization and
regularization techniques below still matter).

## 9.6 Overfitting and Regularization

Overfitting is when a model fits the training data perfectly but performs
worse on new data. In the bias-variance terms from Chapter 4.4, a neural
network's huge number of parameters makes it an extremely flexible
(high-variance) model, which is exactly why it's so prone to overfitting —
Chapter 6's L1/L2 regularization applies to neural nets too, but neural
nets also lean on regularization techniques baked into their structure:

- **Dropout**: at every training step, randomly turn off some fraction
  \\(p\\) of neurons. This prevents neurons from becoming overly dependent
  on one another, forcing the network to learn more robust features rather
  than relying on a handful of specific neurons — similar to how "assuming
  some teammates might always be absent" tends to make everyone develop
  each other's skills to some degree.
- **Batch Normalization**: normalize each layer's input, per mini-batch, to
  mean 0 and variance 1, then multiply and add learnable scale/shift
  parameters. This reduces the phenomenon where a layer's input
  distribution keeps shifting during training (internal covariate shift),
  making training faster and more stable.

```python
def dropout(activations, p, training):
    if not training:
        return activations
    import random
    return [0.0 if random.random() < p else a / (1 - p) for a in activations]
```

(Why divide by `(1-p)`: during training, only \\((1-p)\\) of neurons are
alive on average, so this rescaling keeps the expected value consistent
with inference time, when all neurons are used.)

## 9.7 Weight Initialization

If all weights are initialized to zero, every neuron learns identically
(the symmetry-breaking problem), making stacking multiple layers
pointless. If weights are initialized too large, gradients explode; too
small, and vanishing becomes more likely. Methods like Xavier/He
initialization account for the number of input/output neurons at a layer
and initialize weights randomly with an appropriately scaled variance —
designed so the gradient's magnitude stays roughly constant across layers.

**A neural network's ability to "learn anything" comes from stacking layers
to increase its expressive power, and its ability to be "actually trained"
comes down to a single idea: the chain rule — but the intuition "deeper
network = smarter model" is only half right. The other half is the far
more practical question of whether that depth can actually be trained.**

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

**2. (Coding)** Complete `relu`, `relu_prime`, and
`gradient_norm_through_layers` below (key lines left blank):

```python
def relu(z):
    # ADD ADDITIONAL CODE HERE!!

def relu_prime(z):
    # ADD ADDITIONAL CODE HERE!!
    # 1 if z > 0, else 0 (by convention, 0 at z=0)

def gradient_norm_through_layers(n_layers, sigmoid_derivatives):
    # ADD ADDITIONAL CODE HERE!!
    # return the product of all values in sigmoid_derivatives

print(gradient_norm_through_layers(5, [0.2]*5))   # 0.2^5 = 0.00032
print(gradient_norm_through_layers(20, [0.2]*20)) # 0.2^20 -- effectively 0
```

**3. (Conceptual)** State exactly which Chapter 2 model a neural network
with **no** hidden layer (input directly to sigmoid output) reduces to.
Then explain how Batch Normalization and Dropout each behave differently
at inference time compared to training time.

**4. (Hand derivation, Tier C — fallback prepared)** Starting from the loss
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

**5. (Hand derivation, Tier B — hints provided)** Show that the sigmoid
derivative \\(\sigma'(z) = \sigma(z)(1-\sigma(z))\\) reaches a maximum of
0.25 at \\(z=0\\) (hint: differentiate \\(f(p)=p(1-p)\\) with respect to
\\(p\\) and set it to zero — the maximum occurs at \\(p=0.5\\), and
\\(\sigma(0)=0.5\\)).

For a 10-layer network where every layer's \\(\sigma'(z_l)\\) equals this
maximum of 0.25, compute what percentage of the original gradient's size
survives after passing through all 10 layers. Then compute the same for
ReLU (where \\(\sigma'(z_l)=1\\) in the positive region) and compare.

**Confirm correctness**: based on your calculations, explain in one
paragraph why sigmoid-based networks get harder to train as they get
deeper, and note that ReLU doesn't fully solve this problem either (its
derivative is 0 in the negative region).
