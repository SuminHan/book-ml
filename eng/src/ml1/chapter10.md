# Chapter 10. CNN Basics & Applications

In 1959, neurophysiologists David Hubel and Torsten Wiesel ran an experiment
where they implanted electrodes in a cat's visual cortex and showed it
various visual stimuli. They found that certain neurons respond not to the
**whole** screen, but only to a specific line orientation within a tiny
region (a receptive field) — meaning the brain doesn't look at an image as
a whole, but stacks layers of neurons that each detect small local
patterns, combining them into increasingly complex patterns (lines →
edges → shapes → objects). This discovery, via Kunihiko Fukushima's
Neocognitron in the 1980s, became the direct inspiration for what is now
the **Convolutional Neural Network (CNN)**. This chapter covers the
convolution operation itself (the first half) together with how it gets
reused and extended in practice (the second half).

## 10.1 Why You Shouldn't Feed Images to an MLP

To feed an image into the Multi-Layer Perceptron (MLP) from Chapter 9, you'd
have to flatten every pixel into one long vector. A 200×200 grayscale image
alone gives 40,000 inputs, and if the first hidden layer has 1000 neurons,
that's already 40 million weights just for that connection — and this
approach doesn't exploit the fact that "a cat is still a cat whether it's
in the top-left or bottom-right of the photo" at all. Shift a single pixel
and the whole thing is treated as a completely different input.

## 10.2 Convolution: Reusing a Small Filter Across the Whole Image

CNN's core idea is the same thing Hubel and Wiesel discovered: instead of
looking at the **entire** image at once, slide a small filter (e.g. 3×3)
across the whole image, reusing it **identically** everywhere. A \\(k
\times k\\) filter (kernel) slides over the input, and at each position
computes the sum of elementwise products with the overlapping region:

```python
def conv2d(image, kernel):
    img_h, img_w = len(image), len(image[0])
    k = len(kernel)
    out_h, out_w = img_h - k + 1, img_w - k + 1
    output = [[0.0] * out_w for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            for di in range(k):
                for dj in range(k):
                    total += image[i+di][j+dj] * kernel[di][dj]
            output[i][j] = total
    return output
```

Each filter is **trained** to respond strongly to a particular pattern
(vertical lines, edges, etc.) — the filter's values themselves are the
parameters being learned. An "edge-detecting filter" needs to detect edges
the same way no matter where it is in the image, so there's no need to
learn separate weights for each position — this reuse (**parameter
sharing**) is why CNNs can handle much larger images with far fewer
parameters than an MLP.

## 10.3 The Output Size Formula

For an input of size \\(n \times n\\), filter size \\(k \times k\\),
padding \\(p\\) (the number of zero pixels added around the input's edge),
and stride \\(s\\) (how many cells the filter moves each step), the output
size is:

\\[n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1\\]

Example: with \\(n=28, k=3, p=0, s=1\\), \\(n_{\text{out}} = 28-3+1 = 26\\).
Without padding, the image shrinks a little every time a filter passes
over it — setting \\(p = (k-1)/2\\) (**"same" padding**) keeps the output
the same size as the input.

## 10.4 Counting Parameters and Computation

For a convolutional layer with \\(C_{\text{in}}\\) input channels,
\\(C_{\text{out}}\\) output channels (i.e., filters), and filter size
\\(k \times k\\), the number of parameters (including bias) is:

\\[\text{params} = (k \times k \times C_{\text{in}} + 1) \times C_{\text{out}}\\]

Comparing this to a fully-connected layer of the same size makes the
efficiency of CNNs obvious. Example: for a 32×32×3 image, a convolutional
layer with a 3×3 filter and 16 output channels uses only
\\((3 \times 3 \times 3 + 1) \times 16 = 448\\) parameters. Connecting the
same input to a fully-connected layer (assuming 32×32×16 output neurons
too) would need millions.

**Computation (FLOPs) is a different question from parameter count**:
parameter count measures "how many numbers do we need to store," while the
actual computational cost (speed) is measured by "how many
multiply-accumulate operations happen." Computing one output position, one
channel, takes \\(k \times k \times C_{\text{in}}\\) multiplications, and
this repeats for every output position (\\(n_{\text{out}} \times
n_{\text{out}}\\) of them) and every filter (\\(C_{\text{out}}\\) of them):

\\[\text{FLOPs} \approx n_{\text{out}}^2 \times C_{\text{out}} \times (k
\times k \times C_{\text{in}})\\]

This looks almost identical to the parameter-count formula, but **it has an
extra factor of \\(n_{\text{out}}^2\\) that the parameter count doesn't** —
the same filter (the same parameters) gets reused at every output position,
which is why parameters stay few, but the actual computation repeats once
per position.

Example: apply a single 3×3 filter (\\(C_{\text{out}}=1\\), no padding,
stride 1) to a 224×224 grayscale image (\\(C_{\text{in}}=1\\)). Then
\\(n_{\text{out}}=224-3+1=222\\), and:

\\[\text{FLOPs} = 222^2 \times 1 \times (3\times3\times1) = 49{,}284
\times 9 = 443{,}556\\]

About 440,000 multiply-accumulate operations. Real CNNs use dozens to
hundreds of filters, not just one — bump the same input up to 64 filters
(\\(C_{\text{out}}=64\\)) and computation scales up by exactly 64×, to about
28.39 million. **Adding more kernels (filters) scales parameter count and
computation in exact proportion.** Real CNNs stack dozens to hundreds of
such convolutional layers, which is why total computation is tracked just
as closely as parameter count when designing a model.

![How spatial size (width×height) shrinks and channel count (depth) grows across convolutions — visualizing the exact network from Section 10.4's exercise (32×32×3 → Conv5×5,6 → Pool2×2 → Conv5×5,16)](../images/ch09_cnn_structure.svg)

## 10.5 Pooling

After convolution, a pooling layer usually reduces the spatial size.
**Max pooling** keeps only the maximum value from each \\(2\times2\\)
region — if a feature shifts by a pixel or two, the same maximum is likely
to still be picked, creating features that are insensitive to small shifts
(translation-invariant). Pooling has no trainable parameters — it's a pure
downsampling operation.

## 10.6 The Typical CNN Structure

Stack `[convolution → activation (ReLU) → pooling]` several times, shrinking
spatial size while growing the number of channels (features), then finish
with one or two fully-connected layers to produce the classification
output. Shallow layers learn to detect low-level patterns like lines and
edges, while deeper layers learn increasingly high-level patterns like
eyes, noses, and wheels — the same hierarchical structure, from simple
cells to complex cells, that Hubel and Wiesel observed.

## 10.7 Transfer Learning: Not Starting From Scratch

From here on, we cover how the convolutional structure above gets reused
and extended in practice. The early layers of a CNN pretrained on a large
dataset like ImageNet (millions of images) have learned to detect very
general patterns — lines, edges, textures — closer to "the ability to see
an image" than "the ability to tell cats apart." **Transfer learning**
reuses these pretrained early layers as-is, and only retrains the last few
layers (or just the final classification layer) for the new problem:

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

## 10.8 Beyond Classification: Detection and Segmentation

So far our CNNs have only answered "what class is the entire image" —
classification. Real problems often demand a more fine-grained answer:

- **Object Detection**: "where are the objects in this image, and what are
  they" — predicting each object's location as a bounding box along with
  its class.
- **Semantic Segmentation**: "what class does each pixel belong to" —
  producing output the same size as the input image, with a class label
  attached to every pixel (used in self-driving cars to tell "this pixel is
  road/pedestrian/vehicle," for example).

Both problems still use Section 10.2's convolution and 10.5's pooling as
their basic ingredients, but add structure to change the shape of the
output (e.g., in segmentation, layers that expand the spatial size back to
the original resolution) — the detailed architecture is beyond this
semester's scope, but the underlying principle — "attach a different output
head on top of the features convolution extracts, and you can change the
problem" — is worth remembering.

## 10.9 ResNet and Skip Connections: Stacking Deeper

In 2015, Kaiming He and colleagues at Microsoft noticed something odd —
stacking a CNN deeper was supposed to keep improving performance, but past
a certain point, it got **worse** instead. This wasn't ordinary overfitting
(even training performance got worse) — adding more layers meant the
gradient couldn't fully get through, so the deep layers effectively stopped
learning (Chapter 9's vanishing gradient problem, resurfacing in CNNs).
Their solution, **ResNet** (Residual Network), was remarkably simple, and
it became a basic ingredient in nearly every large CNN and Transformer
architecture used today.

The **skip connection** (or residual connection) cuts a shortcut through
that chain of multiplication — instead of a block's output being just
\\(F(x)\\), it's defined as the input added back in: \\(y = F(x) + x\\):

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
(in a different form) in the Transformer architecture we'll cover in
Chapter 12.

**A CNN implements a single insight — that an image's local patterns can be
reused regardless of position — through a single operation: convolution.
Transfer learning, detection, segmentation, and ResNet all solve different
problems, but they're all built on top of that ingredient, answering "how
do we reuse, recombine, or stabilize it?"**

---

## Exercises

**1. (Coding)** Complete `conv2d` above and the following `max_pool` (key
lines left blank):

```python
def max_pool(feature_map, pool_size):
    # ADD ADDITIONAL CODE HERE!!

fm = [[1,3,2,4],
      [5,6,7,8],
      [9,1,2,3],
      [4,5,6,7]]
print(max_pool(fm, 2))  # [[6,8],[9,7]]
```

**2. (Coding)** Complete `residual_block` above, and
`transfer_learning_predict`, which mimics "a frozen feature extractor plus
a newly-trained classification layer" (key lines left blank):

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

**3. (Conceptual)** (a) Holding the input size and filter size fixed,
explain using the output-size formula how the output size and computation
(FLOPs) change if you increase the stride \\(s\\) from 1 to 2. (b) When
the new problem's data is very scarce versus fairly plentiful (though not
as much as the pretraining data), explain with reasons which of feature
extraction or fine-tuning would be the better choice in each case.

**4. (Hand derivation, Tier A — free derivation)** For an input of size
\\(n \times n\\), filter size \\(k \times k\\), padding \\(p\\), and stride
\\(s\\), derive the output size formula

\\[n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1\\]

by directly counting the number of positions a filter can occupy on the
image. (Hint: after padding, the effective input size is \\(n+2p\\). Count
how many hops of size \\(s\\) are needed for the filter to move from one
end to the other.)

Then compute the output size, parameter count, and computation (FLOPs) at
each layer of the following CNN — input: 32×32×3, Layer 1: 5×5 filter, 6
output channels, padding 0, stride 1. Layer 2 (pooling): 2×2 max pooling
(no parameters or FLOPs to compute). Layer 3: 5×5 filter, 16 output
channels, padding 0, stride 1.

**5. (Hand derivation, Tier B — hints provided)** For a neural network made
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
