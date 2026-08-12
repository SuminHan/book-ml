# Chapter 10. Generative Models II: Adversarial & Score-Based

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter10_gan.ipynb)

In 2014, Ian Goodfellow, then a graduate student, has said in several later
interviews that he came up with an idea during a bar argument with
friends: instead of training a model to produce realistic fake images
directly, **what if two neural networks competed against each other?**
Set one up as a forger creating fakes (the Generator) and the other as an
appraiser distinguishing real from fake (the Discriminator), and have the
forger try to fool the appraiser while the appraiser tries to catch the
forger — wouldn't the forger's skill keep improving until it approached
the real thing? This idea is the **GAN** (Generative Adversarial Network).

## 10.1 A Completely Different Principle From Last Chapter

Chapter 9's VAE had a clear objective function: "maximize the likelihood
(the probability of generating the data)." GAN has no such single
objective — instead, two networks are trained simultaneously with
different, even **opposing**, goals. This is a fundamentally different
game-theoretic setup from the "minimize one loss function" framework
we've used so far.

## 10.2 GAN: The Generator-Discriminator Min-Max Game

- **Generator** \\(G\\): takes random noise \\(z \sim p(z)\\) (usually
  standard normal) and produces a fake sample \\(G(z)\\).
- **Discriminator** \\(D\\): takes data as input and outputs the
  probability \\(D(x) \in [0,1]\\) that it's real (exactly the same form
  as Chapter 3's logistic regression).

Objective function (min-max game):

\\[\min\_G \max\_D V(D,G) = \mathbb{E}\_{x \sim p\_{\text{data}}}[\log D(x)] +
\mathbb{E}\_{z \sim p(z)}[\log(1 - D(G(z)))]\\]

- **The discriminator's perspective (\\(\max_D\\))**: wants \\(D(x)\\)
  close to 1 for real data and \\(D(G(z))\\) close to 0 for fake data —
  this value grows the better it distinguishes real from fake.
- **The generator's perspective (\\(\min_G\\))**: wants \\(D(G(z))\\)
  close to 1 (wants to fool the discriminator).

```python
def discriminator_loss(D_real, D_fake):
    # D_real = D(real data), D_fake = D(fake data) -- both are (0,1) probabilities
    return -(math.log(D_real) + math.log(1 - D_fake))  # loss the discriminator minimizes

def generator_loss(D_fake):
    return -math.log(D_fake)  # loss the generator minimizes (pushes D_fake toward 1)
```

## 10.3 Why We Need to Discuss "Equilibrium"

In ordinary supervised learning, minimizing a single loss function was the
whole story. GAN trains two networks simultaneously with **opposing**
goals, so we first have to redefine what "training is done" even means.
In game theory, the stable state of a situation like this (each player
choosing their best strategy given the other's) is called a **Nash
equilibrium** — a state where neither side can improve by changing their
own strategy alone.

Theoretically, this game's Nash equilibrium is proven to be the point
where \\(D(x) = 0.5\\) (the discriminator cannot tell real from fake at
all) and the distribution \\(G\\) produces exactly matches the real data
distribution — the theoretical endpoint is "the generator is so perfect
that the discriminator is left just guessing." This chapter's exercises
argue this equilibrium directly.

## 10.4 Instability in Practice

Unlike the theory, actual GAN training is often unstable — a common
problem is **mode collapse**, where the generator crowds around just a
few patterns that fool the discriminator and loses diversity. VAE tends to
be theoretically stable but produces somewhat blurry results, while GAN
produces sharp results but is finicky to train — this tradeoff is the
practical difference between the two principles.

## 10.5 Another Principle: Diffusion — Gradually Reversing Noise

**Diffusion models** are a third, completely different approach.

**Forward process**: add tiny amounts of noise to a real image \\(x_0\\),
repeated \\(T\\) times, until \\(x_T\\) becomes pure random noise (this
process is a fixed procedure, not something that's trained).

**Reverse process**: a neural network is trained to predict "what the
noise looked like one step earlier" — that is, it repeatedly learns the
small step of reconstructing \\(x_{t-1}\\) from \\(x_t\\). Once trained, a
new image is created by starting from pure noise \\(x_T\\) and repeating
this reverse step \\(T\\) times.

Unlike GAN, which tries to turn noise into an image "all at once,"
Diffusion breaks that hard problem into \\(T\\) tiny steps — each step is
the much easier problem of "removing just a tiny bit of noise," so
training is far more stable overall. The tradeoff is that generating a
single image requires \\(T\\) repeated steps, making it slower than GAN.
The most widely used image generation models today (Stable Diffusion and
similar) use this principle.

## 10.6 Comparing the Three Principles

| | VAE (likelihood-based) | GAN (adversarial) | Diffusion (score-based) |
|---|---|---|---|
| Training objective | Maximize ELBO | Equilibrium of a min-max game | Minimize noise-prediction error at each step |
| Training stability | Relatively stable | Prone to instability | Stable |
| Generation quality | Somewhat blurry | Sharp | Sharp |
| Generation speed | Fast (one shot) | Fast (one shot) | Slow (requires iteration) |

**If we had learned only one principle (VAE), it would be easy to
misunderstand "generative model = likelihood maximization." Only by seeing
all three side by side does it become clear how many different ways the
problem of generative modeling itself can be solved.**

---

## Exercises

**1. (Coding)** Complete `discriminator_loss` and `generator_loss` below
(key lines left blank):

```python
import math

def discriminator_loss(D_real, D_fake):
    # ADD ADDITIONAL CODE HERE!!
    # -(log(D_real) + log(1 - D_fake))

def generator_loss(D_fake):
    # ADD ADDITIONAL CODE HERE!!
    # -log(D_fake)

print(discriminator_loss(D_real=0.9, D_fake=0.1))  # the better it distinguishes, the smaller the loss
print(generator_loss(D_fake=0.9))                  # the better it fools the discriminator, the smaller the loss
```

**2. (Hand derivation, Tier C — fallback prepared)** For a fixed generator
\\(G\\), the discriminator's optimal solution is known to be

\\[D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}\\]

Accepting this fact, show that if the generator becomes theoretically
perfect, \\(p_G = p_{\text{data}}\\), then \\(D^*(x) = 0.5\\) (for all
\\(x\\)). Then argue, in one paragraph, why neither the discriminator nor
the generator can improve further at this point, and explain why this is
a Nash equilibrium.

**Fill-in-the-blank fallback version** (if free-form argument is too
difficult):

```
Step 1: assume D*(x) = p_data(x) / (p_data(x) + p_G(x)).

Step 2: if p_G(x) = p_data(x):
  D*(x) = p_data(x) / (______________) = ______________

Step 3: this means the discriminator is ______________ (good at telling them apart / completely confused).

Step 4: if the generator changes strategy here, it ______________ (gets better / gets worse)
Step 5: if the discriminator changes strategy here, it ______________ (gets better / gets worse)

Conclusion: the state where neither side can benefit from changing strategy alone = ______________ (name of the equilibrium)
```

**Confirm correctness**: check that the loss values computed in Exercise 1
match the values at this equilibrium point (\\(D=0.5\\)), and note in one
sentence why reaching this equilibrium in actual GAN training isn't as
easy as the theory suggests (e.g., mode collapse).
