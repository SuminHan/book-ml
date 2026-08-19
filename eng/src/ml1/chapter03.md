# Chapter 3. Generative Classifiers: Naive Bayes & GDA

In 2002, programmer Paul Graham wrote an essay called "A Plan for Spam,"
arguing that most spam filters of the time — which relied on hand-written
rules ("if the subject contains 'free,' it's spam") — were easily fooled,
and that a much simpler statistical method could do better: count how
often words appear in spam versus legitimate email, then flip that with
Bayes' rule to compute "the probability this is spam given these words."
This method (Naive Bayes) was widely adopted and became the standard for
early spam filters. This chapter approaches classification from a
fundamentally different direction than Chapter 2.

## 3.1 Bayes' Rule, and Generative vs. Discriminative

Chapter 2's logistic regression modeled \\(P(y|x)\\) (the probability of
the answer given the input) **directly** — this kind of approach is
called a **discriminative** model. This chapter's approach is the
opposite: model how each class "generates" data (\\(P(x|y)\\)) first, and
flip it via **Bayes' rule** to get the \\(P(y|x)\\) we actually want —
this is called a **generative** model.

\\[P(y|x) = \frac{P(x|y)P(y)}{P(x)}\\]

In classification, \\(x\\) is fixed and we compare across values of
\\(y\\) (the classes), so the denominator \\(P(x)\\) (the same across all
classes) can be ignored, and we just maximize the numerator:

\\[\hat{y} = \arg\max_y P(x|y)P(y)\\]

Here \\(P(y)\\) is the **prior** (e.g., the fraction of all email that's
spam), and \\(P(x|y)\\) is the **likelihood** (e.g., the probability that
a spam email contains these particular words). The name "generative"
comes from the fact that this model, in effect, learns "given class
\\(y\\), how is data \\(x\\) generated" — you can even sample from the
learned \\(P(x|y)\\) to produce fake "typical" data for that class (the
same family of idea as Chapter 15's EM/GMM and generative models).

## 3.2 Gaussian Discriminant Analysis (GDA)

When the input \\(x\\) is continuous (a real-valued vector), the most
natural choice is to assume each class's data follows a **normal
distribution** — this is **Gaussian Discriminant Analysis** (GDA). For
binary classification:

\\[y \sim \text{Bernoulli}(\phi), \qquad x \mid y{=}0 \sim
\mathcal{N}(\mu_0, \Sigma), \qquad x \mid y{=}1 \sim \mathcal{N}(\mu_1, \Sigma)\\]

We assume the two classes share the **same covariance matrix**
\\(\Sigma\\), differing only in their means \\(\mu_0, \mu_1\\) (meaning
the two classes' data is spread in the same "shape," just centered at
different locations). The parameters \\(\phi, \mu_0, \mu_1, \Sigma\\) are
estimated by simply computing the training data's empirical mean and
covariance (maximum likelihood estimation) — solved in closed form with
no gradient descent needed, which is a practical difference from logistic
regression.

**A remarkable fact**: expanding \\(P(y=1|x)\\) computed this way via
Bayes' rule gives exactly the same sigmoid-linear form as Chapter 2's
logistic regression, \\(P(y=1|x) = \sigma(w^Tx+b)\\) (derived in this
chapter's Exercise 2). In other words, GDA is "another road to the same
decision boundary (a straight line) as logistic regression" — just taking
a different path there (assuming the data is Gaussian first, versus
learning the decision boundary directly). If the data really is close to
Gaussian, GDA tends to fit well with less data; if that assumption is
wrong, the discriminative model (logistic regression) tends to be more
robust — this is the fundamental tradeoff between generative and
discriminative models: fit the model to the data, or assume the data
follows the model.

## 3.3 Naive Bayes: Assuming Independence to Dodge the Curse of Dimensionality

GDA works well when \\(x\\) is low-dimensional and continuous, but for
something like a spam filter — where \\(x\\) is "whether each word in a
vocabulary of tens of thousands appears" — estimating the covariance
matrix \\(\Sigma\\) (vocabulary size by vocabulary size) alone is
infeasible. **Naive Bayes** sidesteps this with a bold simplification: it
assumes that, given the class \\(y\\), **each feature (word) is
independent** of the others:

\\[P(x|y) = \prod_{j=1}^n P(x_j|y)\\]

The name "naive" reflects that this assumption is nearly always false in
reality — words like "free" and "prize" really do tend to co-occur in
spam, so they aren't independent. And yet Naive Bayes works surprisingly
well in practice (especially for text classification) — each
\\(P(x_j|y)\\) is estimated just by counting word frequencies in the
data, so the number of parameters grows only **linearly** with vocabulary
size (unlike GDA's covariance matrix, which grows quadratically).

**Laplace smoothing**: if a word never appeared in the training data for
some class but shows up in a new email, \\(P(x_j|y)=0\\) makes the entire
product collapse to zero. Adding 1 to every count sidesteps this:

\\[P(x_j{=}1|y{=}k) = \frac{(\text{documents in class } k\text{ containing word } j) + 1}{(\text{total documents in class } k) + 2}\\]

```python
import math

def train_naive_bayes(emails, labels):
    # emails: a list of word lists; labels: a list of 0(ham)/1(spam)
    vocab = set(w for email in emails for w in email)
    n_spam = sum(1 for l in labels if l == 1)
    n_ham = len(labels) - n_spam
    word_counts = {0: {}, 1: {}}
    for email, label in zip(emails, labels):
        for w in set(email):  # Bernoulli Naive Bayes: only presence/absence matters
            word_counts[label][w] = word_counts[label].get(w, 0) + 1
    return {"vocab": vocab, "word_counts": word_counts,
            "n_spam": n_spam, "n_ham": n_ham, "n_total": len(labels)}

def classify(email, model):
    words = set(email)
    log_prob = {}
    for label, n_docs in [(0, model["n_ham"]), (1, model["n_spam"])]:
        log_p = math.log(n_docs / model["n_total"])  # log P(y)
        for w in model["vocab"]:
            p_present = (model["word_counts"][label].get(w, 0) + 1) / (n_docs + 2)
            log_p += math.log(p_present) if w in words else math.log(1 - p_present)
        log_prob[label] = log_p
    return 1 if log_prob[1] > log_prob[0] else 0
```

**Why add logs**: with thousands of words, multiplying \\(P(x_j|y)\\)
thousands of times produces an extremely small number, and floating-point
precision can't tell the difference (underflow). For exactly the same
reason Chapter 2's cross-entropy used a sum of logs instead of a product
of probabilities, we again turn the product into a sum here:
\\(\log \prod_j P(x_j|y) = \sum_j \log P(x_j|y)\\).

**GDA and Naive Bayes take different approaches (a normal-distribution
assumption for continuous values vs. an independence assumption for
discrete ones), but they share the same generative philosophy: model how
each class produces data first, then flip it with Bayes' rule.**

---

## Exercises

**1. (Coding)** Complete `train_naive_bayes` and `classify` above (key
lines left blank):

```python
def train_naive_bayes(emails, labels):
    # ADD ADDITIONAL CODE HERE!!
    # build the vocab, count documents per class, count word-appearance-per-document by class

def classify(email, model):
    # ADD ADDITIONAL CODE HERE!!
    # for classes 0 and 1, compute log P(y) + sum(log P(x_j|y)) and compare

emails = [["free", "money", "now"], ["meeting", "tomorrow", "project"],
          ["free", "prize", "click"], ["project", "deadline", "meeting"]]
labels = [1, 0, 1, 0]
model = train_naive_bayes(emails, labels)
print(classify(["free", "prize"], model))  # 1 (spam)
print(classify(["project", "meeting"], model))  # 0 (ham)
```

**2. (Hand derivation, Tier B — hints provided)** Given \\(x|y{=}0 \sim
\mathcal{N}(\mu_0, \Sigma)\\), \\(x|y{=}1 \sim \mathcal{N}(\mu_1,
\Sigma)\\) (shared covariance), and \\(y \sim \text{Bernoulli}(\phi)\\),
derive that

\\[P(y{=}1|x) = \sigma(w^Tx+b), \qquad w = \Sigma^{-1}(\mu_1-\mu_0)\\]

(i.e., show that GDA's posterior probability has exactly the same
sigmoid-of-a-linear-function form as Chapter 2's logistic regression).

**Hint** (in three steps): (1) Define \\(z = \log\frac{P(x|y{=}1)P(y{=}1)}
{P(x|y{=}0)P(y{=}0)}\\), and first confirm that \\(P(y{=}1|x) =
\sigma(z)\\) (using Bayes' rule and the sigmoid's definition). (2)
Expanding the log of the multivariate normal density gives
\\(\log P(x|y{=}k) = -\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k) +
\text{constant}\\) — substitute this into \\(z\\) for \\(k=0,1\\) and
expand. (3) Since both classes share the **same** \\(\Sigma\\), the
\\(x^T\Sigma^{-1}x\\) quadratic term cancels exactly — figure out what's
left, confirm that \\(z\\) is a **linear** function of \\(x\\), and
express \\(w\\) and \\(b\\) in terms of \\(\mu_0, \mu_1, \Sigma, \phi\\).

**Confirm correctness**: if the two classes instead had different
covariances (\\(\Sigma_0 \ne \Sigma_1\\)), the quadratic term would *not*
cancel — explain in one sentence what shape the decision boundary would
become in that case (hint: a quadratic form).
