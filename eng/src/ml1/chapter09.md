# Chapter 9. Deep Learning Training Techniques

The backpropagation we covered last chapter can, in theory, compute
gradients no matter how many layers there are. But when you actually stack
10 or 20 layers deep, something strange happens: the weights of layers near
the input barely learn at all. This is one reason neural networks stayed
"shallow" (2-3 layers) for so long — the paradox that going deeper actually
makes training worse.

## 9.1 The Sigmoid's Hidden Trap

The cause lies in the sigmoid function itself. \\(\sigma'(z) =
\sigma(z)(1-\sigma(z))\\) has a maximum of only 0.25, at \\(z=0\\).
Backpropagation **multiplies** this derivative every time it climbs back up
a layer, so with 10 layers, the signal that reaches the first layer is at
best \\(0.25^{10} \approx 0.00000095\\) of its original size — effectively
zero. This is the **vanishing gradient** problem. Conversely, if weights
are initialized badly, the gradient can instead grow explosively larger
with each layer — the **exploding gradient** problem.

## 9.2 Vanishing/Exploding Gradients as a Formula

In a network with \\(L\\) layers, the chain rule gives the gradient with
respect to the first layer's weights this form (a product of each layer's
activation derivative and weights):

\\[\frac{\partial L}{\partial W_1} \propto \prod_{l=2}^{L} \sigma'(z_l) \cdot W_l\\]

- If each \\(\sigma'(z_l) < 1\\) (as with sigmoid), this product shrinks
  exponentially toward 0 as the number of layers grows — **vanishing
  gradient**.
- If each term is greater than 1 (e.g., weights initialized too large), the
  product grows exponentially — **exploding gradient**.

## 9.3 Activation Functions

The activation functions and regularization techniques (dropout, batch
normalization) covered in this chapter look like unrelated tricks on the
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

## 9.4 Overfitting and Regularization

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

## 9.5 Weight Initialization

If all weights are initialized to zero, every neuron learns identically
(the symmetry-breaking problem), making stacking multiple layers
pointless. If weights are initialized too large, gradients explode; too
small, and vanishing becomes more likely. Methods like Xavier/He
initialization account for the number of input/output neurons at a layer
and initialize weights randomly with an appropriately scaled variance —
designed so the gradient's magnitude stays roughly constant across layers.

**The intuition "deeper network = smarter model" is only half right — the
other half is the far more practical question of whether that depth can
actually be trained.**

---

## Exercises

**1. (Coding)** Complete `relu`, `relu_prime`, and
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

**2. (Conceptual)** Batch Normalization and Dropout both have a
regularizing effect, but they work differently. Explain how each one's
behavior changes at inference time compared to training time.

**3. (Hand derivation, Tier B — hints provided)** Show that the sigmoid
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
