# Chapter 15. Latent-Variable Generative Models: From EM/GMM to VAE, GAN, and Diffusion

[![Open In Colab: VAE](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_vae_elbo.ipynb)
[![Open In Colab: GAN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml1/chapter15_gan.ipynb)

In 1977, statisticians Arthur Dempster, Nan Laird, and Donald Rubin
published "Maximum Likelihood from Incomplete Data via the EM Algorithm,"
showing that several seemingly unrelated statistical problems actually
share one common structure: **when part of the information is unobserved
(a latent variable), repeatedly approximate the calculation that would
have been easy if you'd known it.** This chapter starts from their **EM**
(Expectation-Maximization) algorithm, extends the exact same question
("how does an unobserved latent variable generate the data?") to neural
networks with VAE, and closes with GAN and Diffusion, which reach the same
goal through two completely different principles (adversarial training,
gradual noise removal) — four faces of generative modeling in one chapter.

## 15.1 Chicken or Egg: The Latent Variable Problem

Chapter 14's k-means assigned each point to **exactly one** cluster (a
hard assignment). But real data often has fuzzy cluster boundaries — it's
often more natural to say a point belongs 60% to cluster A and 40% to
cluster B. A **Gaussian Mixture Model** (GMM) represents each cluster as a
normal distribution (the same assumption as Chapter 3's GDA), and
represents which cluster each point came from as a **probability**.

Here's the catch — the variable \\(z\\) representing which cluster each
point truly came from is **never observed** in the data (a latent
variable):

- If we **knew** each point's true \\(z\\) (its cluster), estimating each
  cluster's mean and variance would be easy (exactly like Chapter 3's GDA
  — just compute the mean/variance of the points belonging to that
  cluster).
- If we **knew** each cluster's mean and variance, computing the
  probability of each point's cluster membership (\\(z\\)'s posterior)
  would also be easy (Bayes' rule).

The problem is that we know **neither**. It's a chicken-and-egg loop. EM
breaks this loop with iteration: fix one (at some arbitrary starting
value), compute the other, use that to refix the first, and repeat until
the values stop changing.

## 15.2 The EM Algorithm: E-Step and M-Step

**E-step (Expectation)**: assuming the current cluster parameters (means
\\(\mu_k\\), variances \\(\sigma_k^2\\), mixture weights \\(\pi_k\\)) are
correct, compute each data point \\(x_i\\)'s posterior probability of
belonging to cluster \\(k\\) (the **responsibility**) \\(\gamma_{ik}\\)
via Bayes' rule:

\\[\gamma_{ik} = P(z_i{=}k \mid x_i) = \frac{\pi_k \, \mathcal{N}(x_i;
\mu_k,\sigma_k^2)}{\sum_{j} \pi_j \, \mathcal{N}(x_i;\mu_j,\sigma_j^2)}\\]

**M-step (Maximization)**: using the responsibilities just computed as
weights, re-estimate each cluster's parameters — the higher a point's
responsibility for cluster \\(k\\), the more it contributes to estimating
\\(\mu_k, \sigma_k^2\\):

\\[\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \qquad
\sigma_k^2 = \frac{\sum_i \gamma_{ik}(x_i-\mu_k)^2}{\sum_i \gamma_{ik}},
\qquad \pi_k = \frac{\sum_i \gamma_{ik}}{m}\\]

```python
import math

def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def em_step(data, means, sigmas, weights):
    K, n = len(means), len(data)
    # E-step: compute responsibilities
    resp = []
    for x in data:
        probs = [weights[k] * gaussian_pdf(x, means[k], sigmas[k]) for k in range(K)]
        total = sum(probs)
        resp.append([p / total for p in probs])
    # M-step: re-estimate parameters as responsibility-weighted averages
    Nk = [sum(resp[i][k] for i in range(n)) for k in range(K)]
    new_means = [sum(resp[i][k] * data[i] for i in range(n)) / Nk[k] for k in range(K)]
    new_sigmas = [math.sqrt(sum(resp[i][k] * (data[i] - new_means[k]) ** 2
                                  for i in range(n)) / Nk[k]) for k in range(K)]
    new_weights = [Nk[k] / n for k in range(K)]
    return new_means, new_sigmas, new_weights
```

Alternating between the E-step and M-step, the two clusters' means,
variances, and weights converge toward their true values. Placed side by
side with k-means, the structures are identical — in fact, k-means can be
viewed as the special case of GMM where every cluster's variance is fixed
to be very small (so the responsibilities effectively collapse to 0 or 1).

## 15.3 From EM to VAE: Making the Latent Variable Continuous, With a Neural Network

GMM is the simplest case, where the latent variable \\(z\\) (which
cluster) is finite and discrete. The **VAE** (Variational Autoencoder,
2013, by Diederik Kingma and Max Welling) that we cover from here asks
exactly the same question — "how does an unobserved latent variable
generate the data?" — but makes \\(z\\) a **continuous** vector instead of
a finite cluster number, and approximates the E-step/M-step's explicit
iteration with a **neural network** instead.

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

**Structure**: the encoder \\(q_\phi(z|x)\\) takes input \\(x\\) and
outputs a probability distribution over \\(z\\) (mean and variance), a
\\(z\\) is drawn from that distribution, and the decoder
\\(p_\theta(x|z)\\) reconstructs (or generates) \\(x\\) from \\(z\\).

VAE is the first of "the ways to build a generative model" — the
**likelihood-based** approach: train the model to directly maximize (or
maximize an approximation of) the probability (likelihood) \\(P(x)\\) that
it generates the training data. The problem is that
\\(p_\theta(x) = \int p_\theta(x|z)p(z)\,dz\\) is an integral over every
possible \\(z\\), which is generally intractable to compute.

## 15.4 ELBO: Working Around It With a Computable Lower Bound

Instead of \\(\log p_\theta(x)\\), which can't be computed directly, the
following can be shown to hold:

\\[\log p_\theta(x) \ge \mathbb{E}\_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] -
D\_{KL}\big(q_\phi(z|x) \,\|\, p(z)\big)\\]

The right-hand side is called the **ELBO** (Evidence Lower BOund). What
its two terms mean:

- The first term \\(\mathbb{E}[\log p_\theta(x|z)]\\): the
  **reconstruction term** — how well the decoder reconstructs the
  original \\(x\\) from the encoder's \\(z\\) (essentially the same as an
  ordinary autoencoder's reconstruction error).
- The second term \\(D_{KL}(q_\phi(z|x)\|p(z))\\): the **regularization
  term** — how far the encoder's distribution \\(q_\phi(z|x)\\) strays
  from the target prior \\(p(z)\\) (usually a standard normal
  distribution), measured by KL divergence — the same concept Chapter
  2.6 introduced via Shannon's information theory. This term is exactly
  what makes the latent space smooth.

Since \\(\log p_\theta(x) \ge \text{ELBO}\\), **maximizing the ELBO also
raises the actual likelihood's lower bound** — swapping a goal we can't
compute directly for a computable surrogate goal.

```python
def vae_loss(x, x_reconstructed, mu, log_var):
    # reconstruction loss (approximated here with MSE)
    recon_loss = sum((x[i] - x_reconstructed[i]) ** 2 for i in range(len(x)))
    # KL divergence: closed-form formula between Gaussian q(mu, sigma^2) and standard normal N(0,1)
    kl_loss = -0.5 * sum(1 + log_var[i] - mu[i]**2 - math.exp(log_var[i])
                          for i in range(len(mu)))
    return recon_loss + kl_loss  # maximizing ELBO = minimizing this loss (negative ELBO)
```

**The reparameterization trick (for reference)**: sampling \\(z\\) directly
from a probability distribution is a non-differentiable operation, so
backpropagation can't flow back through it to the encoder. VAE works
around this by rewriting the sampling as \\(z = \mu + \sigma \odot
\epsilon\\) (treating \\(\epsilon \sim \mathcal{N}(0,1)\\) as a constant
that pulls the randomness outside).

## 15.5 GAN: The Generator-Discriminator Min-Max Game

In 2014, Ian Goodfellow, then a graduate student, has said in several
later interviews that he came up with an idea during a bar argument with
friends: instead of training a model to produce realistic fake images
directly, **what if two neural networks competed against each other?** Set
one up as a forger creating fakes (the Generator) and the other as an
appraiser distinguishing real from fake (the Discriminator), and have the
forger try to fool the appraiser while the appraiser tries to catch the
forger. Unlike VAE, which had a clear objective function ("maximize the
likelihood"), GAN has no such single objective — instead, two networks are
trained simultaneously with different, even **opposing**, goals.

- **Generator** \\(G\\): takes random noise \\(z \sim p(z)\\) and produces
  a fake sample \\(G(z)\\).
- **Discriminator** \\(D\\): takes data as input and outputs the
  probability \\(D(x) \in [0,1]\\) that it's real (exactly the same form
  as Chapter 2's logistic regression).

Objective function (min-max game):

\\[\min_G \max_D V(D,G) = \mathbb{E}\_{x \sim p_{\text{data}}}[\log D(x)] +
\mathbb{E}\_{z \sim p(z)}[\log(1 - D(G(z)))]\\]

```python
def discriminator_loss(D_real, D_fake):
    return -(math.log(D_real) + math.log(1 - D_fake))  # loss the discriminator minimizes

def generator_loss(D_fake):
    return -math.log(D_fake)  # loss the generator minimizes (pushes D_fake toward 1)
```

**Why we need to discuss "equilibrium"**: since two networks train
simultaneously with opposing goals, we first have to redefine what
"training is done" even means. In game theory, the stable state of a
situation like this (each player choosing their best strategy given the
other's) is called a **Nash equilibrium** — a state where neither side can
improve by changing their own strategy alone. Theoretically, this game's
Nash equilibrium is proven to be the point where \\(D(x) = 0.5\\) (the
discriminator cannot tell real from fake at all) and the distribution
\\(G\\) produces exactly matches the real data distribution.

Unlike the theory, actual GAN training is often unstable — a common
problem is **mode collapse**, where the generator crowds around just a few
patterns that fool the discriminator and loses diversity. VAE tends to be
theoretically stable but produces somewhat blurry results, while GAN
produces sharp results but is finicky to train.

## 15.6 Diffusion: Gradually Reversing Noise

**Diffusion models** are a third, completely different approach.
**Forward process**: add tiny amounts of noise to a real image \\(x_0\\),
repeated \\(T\\) times, until \\(x_T\\) becomes pure random noise (this
process is a fixed procedure, not something that's trained). **Reverse
process**: a neural network is trained to predict "what the noise looked
like one step earlier" — once trained, a new image is created by starting
from pure noise \\(x_T\\) and repeating this reverse step \\(T\\) times.

Unlike GAN, which tries to turn noise into an image "all at once,"
Diffusion breaks that hard problem into \\(T\\) tiny steps — each step is
the much easier problem of "removing just a tiny bit of noise," so
training is far more stable overall. The tradeoff is that generating a
single image requires \\(T\\) repeated steps, making it slower than GAN.
The most widely used image generation models today (Stable Diffusion and
similar) use this principle.

## 15.7 Comparing the Four Principles

| | EM/GMM | VAE (likelihood-based) | GAN (adversarial) | Diffusion (score-based) |
|---|---|---|---|---|
| Latent variable | Discrete (cluster number) | Continuous vector | None (transforms noise directly) | Continuous (noise steps) |
| Training method | Alternating E-step/M-step | Maximize ELBO (backprop) | Equilibrium of a min-max game | Minimize noise-prediction error at each step |
| Training stability | Stable | Relatively stable | Prone to instability | Stable |
| Generation speed | — | Fast (one shot) | Fast (one shot) | Slow (requires iteration) |

**If we had learned only one of these principles, it would be easy to
misunderstand what "generative model" means narrowly. Only by seeing all
four side by side does it become clear how many different ways the
question "how does unobserved structure produce new data" can be solved.**

---

## Exercises

**1. (Coding)** Complete `em_step` above and `kl_divergence_gaussian` (the
KL divergence between Gaussian \\(\mathcal{N}(\mu, \sigma^2)\\) and the
standard normal) below (key lines left blank):

```python
def em_step(data, means, sigmas, weights):
    # ADD ADDITIONAL CODE HERE!!
    return new_means, new_sigmas, new_weights

data = [1.0, 1.5, 0.8, 9.2, 10.1, 9.8]
means, sigmas, weights = [2.0, 8.0], [1.0, 1.0], [0.5, 0.5]
for _ in range(20):
    means, sigmas, weights = em_step(data, means, sigmas, weights)
print(means)  # converges to roughly [1.1, 9.7]

def kl_divergence_gaussian(mu, log_var):
    # log_var = log(sigma^2), D_KL = -0.5*(1 + log_var - mu^2 - exp(log_var))
    # ADD ADDITIONAL CODE HERE!!

print(kl_divergence_gaussian(mu=0.0, log_var=0.0))  # 0.0
print(kl_divergence_gaussian(mu=2.0, log_var=0.0))  # 2.0
```

**2. (Coding)** Complete `discriminator_loss` and `generator_loss` above
(key lines left blank):

```python
def discriminator_loss(D_real, D_fake):
    # ADD ADDITIONAL CODE HERE!!

def generator_loss(D_fake):
    # ADD ADDITIONAL CODE HERE!!

print(discriminator_loss(D_real=0.9, D_fake=0.1))
print(generator_loss(D_fake=0.9))
```

**3. (Conceptual)** Explain why GMM becomes equivalent to k-means if each
cluster's variance is fixed to a very small value and never learned, and
compare, in two or three sentences, the generation-quality vs.
training-stability tradeoff between VAE (likelihood-based) and GAN
(adversarial).

**4. (Hand derivation, Tier B — hints provided)** With 3 data points
\\(x_1=1, x_2=9, x_3=10\\), 2 clusters, and current parameters
\\(\mu_1=0, \sigma_1=1, \mu_2=9, \sigma_2=1, \pi_1=\pi_2=0.5\\),
hand-compute the responsibilities \\(\gamma_{1,1}\\) and \\(\gamma_{1,2}\\)
for \\(x_1\\).

**Hint**: compute the normal density \\(\mathcal{N}(x;\mu,\sigma^2) =
\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}\\) at \\(x_1=1\\) for
each cluster and substitute into the responsibility formula. Note that
since \\(\pi_1=\pi_2\\), the \\(\pi\\) terms actually cancel out.

**5. (Hand derivation, Tier C — fallback prepared)** Starting from
\\(\log p_\theta(x) = \log \int p_\theta(x,z)\,dz\\), use Jensen's
inequality (\\(\log \mathbb{E}[X] \ge \mathbb{E}[\log X]\\)) to derive
Section 15.4's ELBO inequality.

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
inequality rather than an equality (hint: when does Jensen's inequality
become an equality? When \\(X\\) is constant).
