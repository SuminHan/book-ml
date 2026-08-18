# Chapter 11. CNN Applications & Modern Architectures

In 2015, Kaiming He and colleagues at Microsoft noticed something odd —
stacking a CNN deeper was supposed to keep improving performance, but past
a certain point, it got **worse** instead. This wasn't the overfitting from
Chapter 9 (even training performance got worse) — adding more layers meant
the gradient couldn't fully get through, so the deep layers effectively
stopped learning. Their solution, **ResNet** (Residual Network), was
remarkably simple, and it became a basic ingredient in nearly every large
CNN and Transformer architecture used today. This chapter covers how CNNs
get reused and extended in practice.

## 11.1 Transfer Learning: Not Starting From Scratch

The early layers of a CNN pretrained on a large dataset like ImageNet
(millions of images) have learned to detect very general patterns — lines,
edges, textures — closer to "the ability to see an image" than "the
ability to tell cats apart." **Transfer learning** reuses these pretrained
early layers as-is, and only retrains the last few layers (or just the
final classification layer) for the new problem:

- **Feature extraction**: freeze all the pretrained layers' weights, and
  train only the final classification layer. Especially useful when data is
  scarce — there are far fewer new parameters to learn.
- **Fine-tuning**: use the pretrained weights only as a starting point, and
  continue training the whole network (or the later layers) at a low
  learning rate. Tends to outperform feature extraction once you have a
  reasonable amount of new data.

**Why this works**: Chapter 6 taught us that regularization "reduces
variance by making the model less flexible." Transfer learning has a
similar effect — instead of starting from a random initialization, you
start from an already-validated good starting point, so you can train
without overfitting even on much less data.

## 11.2 Beyond Classification: Detection and Segmentation

So far our CNNs have only answered "what class is the entire image" —
classification. Real problems often demand a more fine-grained answer:

- **Object Detection**: "where are the objects in this image, and what are
  they" — predicting each object's location as a bounding box along with
  its class.
- **Semantic Segmentation**: "what class does each pixel belong to" —
  producing output the same size as the input image, with a class label
  attached to every pixel (used in self-driving cars to tell "this pixel is
  road/pedestrian/vehicle," for example).

Both problems still use convolution and pooling from this book as their
basic ingredients, but add structure to change the shape of the output
(e.g., in segmentation, layers that expand the spatial size back to the
original resolution) — the detailed architecture is beyond this semester's
scope, but the underlying principle — "attach a different output head on
top of the features convolution extracts, and you can change the
problem" — is worth remembering.

## 11.3 ResNet and Skip Connections: Stacking Deeper

The vanishing gradient problem from Chapter 9 came from multiplying the
activation function's derivative at every layer. ResNet's fix, the **skip
connection** (or residual connection), cuts a shortcut through that chain
of multiplication — instead of a block's output being just \\(F(x)\\), it's
defined as the input added back in: \\(y = F(x) + x\\):

```python
def residual_block(x, F):
    # F: a function made of a few convolutional layers (e.g., Conv-ReLU-Conv)
    return [F(x)[i] + x[i] for i in range(len(x))]  # F(x) + x
```

**Why this keeps the gradient alive** (intuition): differentiating
\\(y=F(x)+x\\) with respect to \\(x\\) gives \\(\frac{\partial y}{\partial
x} = \frac{\partial F}{\partial x} + 1\\). Even if
\\(\frac{\partial F}{\partial x}\\) shrinks toward zero (vanishing
gradient) in Chapter 9's chain of products \\(\prod_l \sigma'(z_l) \cdot
W_l\\), **the added "+1" term guarantees a path along which the gradient
stays at least 1** as it passes through — a shortcut that skips over the
input and flows straight to later layers unchanged. This idea is what
made it possible to train CNNs over 100 layers deep, and it shows up
(in a different form) in the Transformer architecture we'll cover in ML2.

**Transfer learning, detection, segmentation, and ResNet all solve
different problems, but they share something: they're all built on top of
Chapter 10's basic ingredient — extracting local patterns with convolution
— and answer "how do we reuse, recombine, or stabilize that ingredient?"**

---

## Exercises

**1. (Coding)** Complete `residual_block` above (key lines left blank), and
`transfer_learning_predict`, which mimics "a frozen feature extractor plus
a newly-trained classification layer":

```python
def residual_block(x, F):
    # ADD ADDITIONAL CODE HERE!!
    # compute F(x), then add it elementwise to x and return

def transfer_learning_predict(x, frozen_features_fn, new_w, new_b):
    # ADD ADDITIONAL CODE HERE!!
    # extract features via frozen_features_fn(x) (no weight updates),
    # then compute new_w . features + new_b as the new classification result

features_fn = lambda x: [x[0]+x[1], x[0]-x[1]]  # a fixed (pretrained) feature extractor
print(transfer_learning_predict([3, 1], features_fn, new_w=[2, 1], new_b=0.5))
# features = [4, 2], prediction = 2*4 + 1*2 + 0.5 = 10.5
```

**2. (Conceptual)** When the new problem's data is (a) very scarce, and (b)
fairly plentiful (though not as much as the pretraining data), explain with
reasons which of feature extraction or fine-tuning would be the better
choice in each case.

**3. (Hand derivation, Tier B — hints provided)** For a neural network made
of \\(L\\) residual blocks stacked with skip connections, show that the
gradient of the final output \\(x_L\\) with respect to the first block's
input \\(x_0\\), \\(\frac{\partial x_L}{\partial x_0}\\), contains "a term
that's at least 1."

**Hint**: let each block be \\(x_{l} = F_l(x_{l-1}) + x_{l-1}\\). By the
chain rule, \\(\frac{\partial x_L}{\partial x_0} = \prod_{l=1}^L
\frac{\partial x_l}{\partial x_{l-1}}\\), and each factor is
\\(\frac{\partial x_l}{\partial x_{l-1}} = \frac{\partial F_l}{\partial
x_{l-1}} + 1\\). Expanding this product (every term in the expansion is
either "1" or "a term involving \\(\partial F_l/\partial x_{l-1}\\)"),
show that **the single term formed by picking "1" from every block**
survives the product exactly equal to 1. Compare this to Chapter 9's
\\(\prod_l \sigma'(z_l)\cdot W_l\\) (without skip connections), and explain
in one paragraph why this "+1" term fundamentally prevents vanishing
gradients.
