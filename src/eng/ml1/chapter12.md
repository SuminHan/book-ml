# Chapter 12. Generative Models Preview & Review

In 2006, Geoffrey Hinton and Ruslan Salakhutdinov showed that neural
networks could perform dimensionality reduction far more flexibly than
PCA. The core idea is simple: train a neural network to compress its input
and then reconstruct the original, but force it through a **bottleneck** —
a hidden layer much narrower than the input — along the way. If the
network can reconstruct the original even after passing through that
bottleneck, then the bottleneck's values must be a useful compressed
representation of the original information. This structure is called an
**Autoencoder**.

## 12.1 The Structure of an Autoencoder

It has two parts:

- **Encoder** \\(f\\): compresses the input \\(x\\) into a low-dimensional
  latent representation \\(z = f(x)\\).
- **Decoder** \\(g\\): reconstructs the original from \\(z\\),
  \\(\hat{x} = g(z)\\).

The loss function is the reconstruction error, usually mean squared error:

\\[J = \frac{1}{m}\sum_{i=1}^m \|x^{(i)} - g(f(x^{(i)}))\|^2\\]

```python
def autoencoder_forward(x, encoder_weights, decoder_weights):
    z = apply_layer(x, encoder_weights)   # compress: narrower hidden layer than input
    x_hat = apply_layer(z, decoder_weights)  # reconstruct: back to original dimension
    return x_hat, z
```

## 12.2 Why a Bottleneck Is Necessary

If the hidden layer were the same size as the input or larger, the network
could simply learn the identity function — copying the input straight
through — giving zero reconstruction error while learning nothing useful.
Forcing a bottleneck (a hidden layer much narrower than the input) means
the network cannot store the original wholesale, so it's forced to
**compress only the information that really matters**.

## 12.3 Relationship to PCA

PCA, covered in Chapter 10, shared the same goal: compress high-dimensional
data into fewer dimensions while losing as little information as possible.
The difference is that PCA can only do **linear** (straight-line)
projections, while an autoencoder, being a neural network, can do
**nonlinear** compression as well — in fact, if an autoencoder's encoder
uses only linear activation functions, it's known to learn mathematically
the exact same thing as PCA:

\\[\text{linear autoencoder} \subset \text{nonlinear autoencoder}, \qquad
\text{linear autoencoder} \approx \text{PCA}\\]

In other words, an autoencoder can be seen as a generalization of PCA.
Adding nonlinear activation functions (ReLU, sigmoid, etc.) to the
encoder/decoder lets it compress curved data structures that PCA simply
cannot represent.

## 12.4 From Compression to Generation

An autoencoder is originally built for compression (encoding), but one
interesting question remains: what happens if you pick a random value in
the bottleneck and feed it to the decoder? Could it produce something new —
similar to the original data, but that **never actually existed**? This
question is exactly where "generative models" begin.

Trying to use an autoencoder directly as a "generative model" runs into a
problem: we have no idea which values in the latent space \\(z\\)
reconstruct into "plausible" data — the region occupied by the training
data in latent space can be sparse or irregular. The **Variational
Autoencoder (VAE)** solves this by forcing the latent space to follow a
smooth probability distribution (usually Gaussian) — covered in ML2
Chapter 9. GAN and Diffusion achieve the same goal (generating plausible
new data) through completely different principles (adversarial training,
gradual noise removal) — covered in ML2 Chapter 10.

## 12.5 ML1 Review

This semester started with linear regression's gradient descent and moved
through classification (logistic regression), nonlinear models (kNN,
trees, GBDT), neural networks (backpropagation, CNNs), and unsupervised
learning (k-means, PCA).

| Chapter | What We Learned | The Key Question |
|---|---|---|
| Ch02 | Linear Regression | How do we predict a continuous value? |
| Ch03 | Logistic Regression | How do we output a probability, and measure classification performance? |
| Ch04 | kNN | Can we predict using "closeness" alone, and what are its limits? |
| Ch05 | Decision Trees/Random Forest | How do we choose the question that best splits the data? |
| Ch06 | GBDT/SHAP | How do we make trees stronger, and explain their predictions? |
| Ch07 | Neural Networks/Backprop | How do we train a layer? |
| Ch08 | Training Techniques | Why and how does a deep network fail to train? |
| Ch09 | CNN | How do we handle an image's local structure efficiently? |
| Ch10 | k-means/PCA | How do we discover the structure of data without correct answers? |
| Ch11 | RL Preview | How do we learn from a reward instead of a correct answer? |
| Ch12 | Autoencoders | How does compression lead to generation? |

Looking back, one structure repeated across almost every chapter: **define
a model → quantify how wrong it is with a loss function → adjust
parameters to reduce that loss.** ML2 builds on this neural-network
foundation and extends it to sequences (RNN/Transformer), reinforcement
learning, and generative models.
