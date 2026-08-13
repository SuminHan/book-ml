# Chapter 9. Generative Models I: Likelihood-Based

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter09_vae_elbo.ipynb)

Remember the question ML1's last chapter left us with — "what if we pick a
random value in the bottleneck and feed it to the decoder? Could it
produce new data that never actually existed?" The **Variational
Autoencoder (VAE)**, published in 2013 by Diederik Kingma and Max Welling,
is the formal answer to that question.

## 9.1 Why an Autoencoder Alone Doesn't Work

A regular autoencoder's latent space comes with no guarantee about "which
values reconstruct into plausible data" — the training data can occupy a
sparse or irregular region of latent space, and sampling randomly from the
empty space in between gives no idea what will come out. VAE's solution is
to force the encoder to output a **probability distribution** (usually the
mean and variance of a Gaussian) instead of a single point \\(z\\). An
additional constraint pushes that distribution to be close to a standard
normal distribution (mean 0, variance 1) — this makes the entire latent
space smooth and densely filled, so sampling from any point is likely to
produce plausible data.

## 9.2 The Structure of a VAE

- **Encoder** \\(q_\phi(z|x)\\): takes input \\(x\\) and outputs a
  probability distribution over the latent variable \\(z\\) (usually the
  mean and variance of a Gaussian \\(\mathcal{N}(\mu(x),
  \sigma^2(x))\\)).
- **Sampling**: draw a \\(z\\) from that distribution.
- **Decoder** \\(p_\theta(x|z)\\): reconstructs (or generates) \\(x\\)
  from \\(z\\).

## 9.3 The Likelihood Perspective

VAE is the first of "the three principles of generative models" — the
**likelihood-based** approach: train the model to directly maximize (or
maximize an approximation of) the probability (likelihood) \\(P(x)\\) that
it generates the training data. The problem is that
\\(p_\theta(x) = \int p_\theta(x|z)p(z)\,dz\\) is an integral over every
possible \\(z\\), which is generally intractable to compute.

## 9.4 ELBO: Working Around It With a Computable Lower Bound

Instead of \\(\log p_\theta(x)\\), which can't be computed directly, the
following can be shown to hold:

\\[\log p\_\theta(x) \ge \mathbb{E}\_{z \sim q\_\phi(z|x)}[\log p\_\theta(x|z)] -
D\_{KL}\big(q\_\phi(z|x) \\,\\|\\, p(z)\big)\\]

The right-hand side is called the **ELBO** (Evidence Lower BOund). What
its two terms mean:

- The first term \\(\mathbb{E}[\log p_\theta(x|z)]\\): the
  **reconstruction term** — how well the decoder reconstructs the
  original \\(x\\) from the encoder's \\(z\\) (essentially the same as an
  ordinary autoencoder's reconstruction error).
- The second term \\(D_{KL}(q_\phi(z|x)\|p(z))\\): the **regularization
  term** — how far the encoder's distribution \\(q_\phi(z|x)\\) strays
  from the target prior \\(p(z)\\) (usually a standard normal
  distribution), measured by KL divergence — the same concept ML1 Chapter
  3.3 introduced via Shannon's information theory, a "distance" between two
  probability distributions. This term is exactly what makes the latent
  space smooth.

Since \\(\log p_\theta(x) \ge \text{ELBO}\\), **maximizing the ELBO also
raises the actual likelihood's lower bound** — swapping a goal we can't
compute directly for a computable surrogate goal. Deriving this inequality
itself using Jensen's inequality is the centerpiece of this chapter's
exercises.

```python
def vae_loss(x, x_reconstructed, mu, log_var):
    # reconstruction loss (approximated here with MSE)
    recon_loss = sum((x[i] - x_reconstructed[i]) ** 2 for i in range(len(x)))
    # KL divergence: closed-form formula between Gaussian q(mu, sigma^2) and standard normal N(0,1)
    kl_loss = -0.5 * sum(1 + log_var[i] - mu[i]**2 - math.exp(log_var[i])
                          for i in range(len(mu)))
    return recon_loss + kl_loss  # maximizing ELBO = minimizing this loss (negative ELBO)
```

## 9.5 The Reparameterization Trick (For Reference)

Sampling \\(z\\) directly from a probability distribution is a
non-differentiable operation, so backpropagation can't flow back through
it to the encoder. VAE works around this with the reparameterization
trick, rewriting the sampling as \\(z = \mu + \sigma \odot \epsilon\\)
(treating \\(\epsilon \sim \mathcal{N}(0,1)\\) as a constant that pulls
the randomness outside) — now \\(\mu\\) and \\(\sigma\\) are
differentiable, so backpropagation flows normally. The detailed derivation
goes beyond this semester's scope, but the question itself — "why can't we
just sample directly" — is worth remembering.

**Next chapter covers two completely different principles (adversarial,
score-based) — comparing all three side by side makes clear just how
differently the same goal ("generate plausible new data") can be
approached.**

---

## Exercises

**1. (Coding)** Complete `kl_divergence_gaussian` (the KL divergence
between Gaussian \\(\mathcal{N}(\mu, \sigma^2)\\) and the standard normal,
\\(D_{KL} = -\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)\\)) and
`vae_loss` below (key lines left blank):

```python
import math

def kl_divergence_gaussian(mu, log_var):
    # log_var = log(sigma^2)
    # ADD ADDITIONAL CODE HERE!!

print(kl_divergence_gaussian(mu=0.0, log_var=0.0))  # 0.0
print(kl_divergence_gaussian(mu=2.0, log_var=0.0))  # 2.0

def vae_loss(recon_loss, mu_list, log_var_list):
    # ADD ADDITIONAL CODE HERE!!
    # sum the KL divergence across all latent dimensions, then add to the reconstruction loss

print(vae_loss(recon_loss=5.0, mu_list=[0.5, -0.3], log_var_list=[0.1, -0.2]))
```

**2. (Hand derivation, Tier C — fallback prepared)** Starting from
\\(\log p_\theta(x) = \log \int p_\theta(x,z)\,dz\\), use Jensen's
inequality (\\(\log \mathbb{E}[X] \ge \mathbb{E}[\log X]\\)) to derive

\\[\log p\_\theta(x) \ge \mathbb{E}\_{z \sim q\_\phi(z|x)}[\log p\_\theta(x|z)] -
D\_{KL}(q\_\phi(z|x)\\|p(z))\\]

(Hint: first rewrite \\(\log p\_\theta(x) = \log \mathbb{E}\_{q\_\phi}
\left[\frac{p\_\theta(x,z)}{q\_\phi(z|x)}\right]\\), then apply Jensen's
inequality, and decompose \\(p_\theta(x,z) = p_\theta(x|z)p(z)\\) to
arrive at the definition of KL divergence.)

**Fill-in-the-blank fallback version** (if free derivation is too
difficult):

```
Step 1: log p(x) = log integral[ q(z|x) * (p(x,z) / q(z|x)) dz ]
                  = log E_q[ ______________ ]

Step 2: apply Jensen's inequality (log is concave):
  log E_q[ p(x,z)/q(z|x) ] >= E_q[ log(______________) ]

Step 3: decomposing p(x,z) = p(x|z) * p(z):
  = E_q[ log p(x|z) ] + E_q[ ______________ ]

Step 4: E_q[ log p(z) - log q(z|x) ] = -D_KL(q(z|x) || p(z))

Conclusion: log p(x) >= E_q[log p(x|z)] - D_KL(q(z|x) || p(z))   [ELBO]
```

**Confirm correctness**: explain in one sentence why Step 2 is an
inequality (`>=`) rather than an equality (hint: when does Jensen's
inequality become an equality? When \\(X\\) is constant), and explain why
maximizing the ELBO amounts to "indirectly" maximizing the true
likelihood \\(\log p(x)\\).
