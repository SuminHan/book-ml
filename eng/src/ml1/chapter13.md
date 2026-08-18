# Chapter 13. EM Algorithm & Gaussian Mixture Models

In 1977, statisticians Arthur Dempster, Nan Laird, and Donald Rubin
published "Maximum Likelihood from Incomplete Data via the EM Algorithm,"
showing that several seemingly unrelated statistical problems actually
share one common structure: **when part of the information is
unobserved (a latent variable), repeatedly approximate the calculation
that would have been easy if you'd known it.** Their algorithm, **EM**
(Expectation-Maximization), still underlies nearly every unsupervised
learning method from Chapter 12's k-means to ML2 Chapter 10's VAE — the
idea of "assuming unobserved structure explains the data."

## 13.1 Chicken or Egg: The Latent Variable Problem

Chapter 12's k-means assigned each point to **exactly one** cluster (a
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

## 13.2 The EM Algorithm: E-Step and M-Step

**E-step (Expectation)**: assuming the current cluster parameters (means
\\(\mu_k\\), variances \\(\sigma_k^2\\), mixture weights \\(\pi_k\\)) are
correct, compute each data point \\(x_i\\)'s posterior probability of
belonging to cluster \\(k\\) (the **responsibility**) \\(\gamma_{ik}\\)
via Bayes' rule:

\\[\gamma_{ik} = P(z_i{=}k \mid x_i) = \frac{\pi_k \, \mathcal{N}(x_i;
\mu_k,\sigma_k^2)}{\sum_{j} \pi_j \, \mathcal{N}(x_i;\mu_j,\sigma_j^2)}\\]

(This is exactly the same calculation as GDA's posterior in Chapter 3 —
the difference is that there, \\(z\\) was a known label, while here it's
this very probability we're trying to compute.)

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
variances, and weights converge toward their true values — running this
loop just 30 times on toy data drawn from two normal distributions
centered near 0 and 10, the estimated means converge almost exactly to 0
and 10, and the variances converge to nearly 1.

## 13.3 GMM Is "Soft k-means"

Placed side by side, the structures are identical:

| | k-means | GMM/EM |
|---|---|---|
| Assignment step | Fixed to the **single** nearest centroid (hard) | Probability across all clusters (soft) |
| Update step | Plain **average** of assigned points | Responsibility-**weighted** average for mean/variance |
| Iteration | Until assignments stop changing | Until parameters stop changing |

In fact, k-means can be viewed as the special case of GMM where every
cluster's variance is fixed to be very small (so the responsibilities
effectively collapse to 0 or 1). Chapter 12's pattern — "repeatedly
updating a value converges to a fixed point" — shows up here again.

**From EM to VAE**: GMM is the simplest case, where the latent variable
\\(z\\) (which cluster) is finite and discrete. ML2 Chapter 10's **VAE**
(Variational Autoencoder) asks exactly the same question — "how does an
unobserved latent variable generate the data?" — but makes \\(z\\) a
**continuous** vector instead of a finite cluster number, and approximates
the E-step/M-step's explicit iteration with a **neural network** instead.
The iterative structure you learn here — assume an unobserved variable,
compute its posterior, update parameters — is the best warmup for
understanding VAE.

**EM is a general-purpose tool that gets reused any time you face a
situation where "the problem would have been easy if only we knew the
latent variable." GMM is just the most hands-on, tangible form of it — the
same idea reappears, in different shapes, all the way through the rest of
this semester.**

---

## Exercises

**1. (Coding)** Complete `em_step` above (key lines left blank):

```python
def em_step(data, means, sigmas, weights):
    # ADD ADDITIONAL CODE HERE!!
    # E-step: for each x in data, compute its responsibility for each cluster
    # M-step: compute new means, sigmas, weights as responsibility-weighted averages

    return new_means, new_sigmas, new_weights

data = [1.0, 1.5, 0.8, 9.2, 10.1, 9.8]
means, sigmas, weights = [2.0, 8.0], [1.0, 1.0], [0.5, 0.5]
for _ in range(20):
    means, sigmas, weights = em_step(data, means, sigmas, weights)
print(means)  # converges to roughly [1.1, 9.7]
```

**2. (Conceptual)** If GMM's cluster variances \\(\sigma_k^2\\) are fixed
at a very small value and never learned, describe what shape the
E-step's responsibilities \\(\gamma_{ik}\\) would take (spread across
several values between 0 and 1, vs. nearly 0 or 1), and explain why this
extreme case becomes equivalent to k-means.

**3. (Hand derivation, Tier B — hints provided)** With 3 data points
\\(x_1=1, x_2=9, x_3=10\\), 2 clusters, and current parameters
\\(\mu_1=0, \sigma_1=1, \mu_2=9, \sigma_2=1, \pi_1=\pi_2=0.5\\),
hand-compute the responsibilities \\(\gamma_{1,1}\\) and \\(\gamma_{1,2}\\)
for \\(x_1\\).

**Hint**: compute the normal density \\(\mathcal{N}(x;\mu,\sigma^2) =
\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}\\) at \\(x_1=1\\) for
each cluster (with \\(\sigma=1\\), the exponent's denominator is 2), then
substitute into \\(\gamma_{1,1} = \frac{\pi_1 \mathcal{N}(x_1;\mu_1,1)}
{\pi_1\mathcal{N}(x_1;\mu_1,1) + \pi_2\mathcal{N}(x_1;\mu_2,1)}\\). Note
that since \\(\pi_1=\pi_2\\), the \\(\pi\\) terms actually cancel out.

**Confirm correctness**: since \\(x_1=1\\) is much closer to
\\(\mu_1=0\\), \\(\gamma_{1,1}\\) should be much larger than
\\(\gamma_{1,2}\\) — check that your computed result matches this
intuition.
