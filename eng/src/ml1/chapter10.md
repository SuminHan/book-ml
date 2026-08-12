# Chapter 10. Unsupervised Learning & Representation Learning

In 1901, statistician Karl Pearson posed an interesting question: given
data with several measured variables, what's the "best line (or plane)"
that reduces the number of variables while preserving as much information
as possible? His answer — find the direction along which the data is most
spread out — is the origin of what's now called **Principal Component
Analysis (PCA)**. An idea from more than 120 years ago is still used today
to compress hundred-dimensional image embeddings down to a 2D space that
humans can actually look at.

## 10.1 Learning Without Correct Answers

Every model we've covered so far had a "correct answer" (\\(y\\)) — house
price, spam or not, a class label. **Unsupervised learning** has none of
that. Instead, it discovers the structure of the data \\(x\\) itself: "how
many natural groups do these customers fall into?" (clustering), "how many
dimensions of these 1000 features actually carry real information?"
(dimensionality reduction).

## 10.2 k-means Clustering (Recap)

An algorithm that groups data by distance to \\(k\\) centroids — we
already covered the procedure and code in section 4.6 (k-means, too, is a
distance-based model in that sense: it finds the closest thing). What
matters here isn't the algorithm itself but **why this counts as
unsupervised learning**: kNN and the regression/classification models
we've covered so far all fit toward a correct label \\(y\\), but k-means
never sees a label — it forms groups purely from the structure of the
data \\(x\\)'s positions, using nothing but "group nearby points
together." Choosing \\(k\\) (the elbow method) works the same way as in
section 4.6.

## 10.3 Why We Want to Reduce Dimensionality

Recall the curse of dimensionality from Chapter 4 — with too many
features, the very notion of "closeness" breaks down. PCA compresses
features that are highly correlated (and thus effectively redundant) into
a handful of new axes (principal components), while preserving as much of
the original data's variance (information) as possible.

## 10.4 PCA: Finding the Axes That Preserve the Most Variance

The goal is to project the data into lower dimensions while making the
**variance of the projected data as large as possible** — meaning the
direction that loses the least information.

**Procedure**:

1. Center the data so the mean is zero.
2. Compute the covariance matrix \\(\Sigma = \frac{1}{m}X^TX\\).
3. Find the eigenvectors and eigenvalues of \\(\Sigma\\).
4. Sort the eigenvectors by decreasing eigenvalue — each eigenvector is a
   "principal component," and its corresponding eigenvalue is the amount
   of variance along that direction.
5. Projecting the data onto the top \\(d\\) eigenvectors gives a
   \\(d\\)-dimensional compressed representation.

**Why eigenvectors** (intuition): a vector \\(v\\) satisfying \\(\Sigma v =
\lambda v\\) (an eigenvector) is special in that projecting the data onto
that direction gives a result whose variance is exactly \\(\lambda\\) (the
eigenvalue). Solving the variance-maximization problem with Lagrange
multipliers leads exactly to this eigenvalue problem — that derivation is
the centerpiece of this chapter's exercises.

## 10.5 A Taste of Embeddings: Node2Vec

While PCA compresses data that's already a vector, **embedding** is a way
of learning a low-dimensional vector representation for things that
weren't originally vectors at all (words, nodes in a graph). Node2Vec
trains a graph so that "nodes with similar neighborhood structure end up
with similar vectors" — for example, two people in similar friend groups
in a social network end up with vectors that are close together. The
mechanism differs, but the goal is the same as PCA's: **compress a
high-dimensional (or non-vector) original into a low-dimensional vector
while preserving useful structure.**

**Supervised learning learns "how to predict the right answer"; unsupervised
learning learns "what shape the data has on its own" — they answer
different questions.**

---

## Exercises

**1. (Coding)** Complete `kmeans_assign` (the assignment step of k-means)
and `center_data` (PCA's preprocessing step) below (key lines left blank):

```python
def kmeans_assign(X, centroids):
    # ADD ADDITIONAL CODE HERE!!

X = [[1,1],[1,2],[8,8],[9,9]]
centroids = [[1,1],[9,9]]
print(kmeans_assign(X, centroids))  # [0, 0, 1, 1]

def center_data(X):
    # ADD ADDITIONAL CODE HERE!!

X2 = [[1,2],[3,4],[5,6]]
print(center_data(X2))  # [[-2,-2],[0,0],[2,2]]
```

**2. (Hand derivation, Tier B — hints provided)** For centered data
\\(X\\), the variance of the data projected onto a unit vector \\(v\\)
(\\(\|v\|=1\\)) is \\(v^T \Sigma v\\) (where \\(\Sigma =
\frac{1}{m}X^TX\\)). We want to find the \\(v\\) that maximizes this
variance.

**Hint**: turn the problem of maximizing \\(v^T\Sigma v\\) subject to
\\(\|v\|=1\\) into \\(\mathcal{L}(v, \lambda) = v^T\Sigma v -
\lambda(v^Tv - 1)\\) using a Lagrange multiplier \\(\lambda\\), then
differentiate with respect to \\(v\\) and set it to zero (using
\\(\frac{\partial}{\partial v}(v^T\Sigma v) = 2\Sigma v\\) and
\\(\frac{\partial}{\partial v}(v^Tv) = 2v\\)) to get the eigenvalue
equation \\(\Sigma v = \lambda v\\). Substituting this back into the
original objective (\\(v^T\Sigma v = v^T(\lambda v) = \lambda\\)) shows
that the variance is exactly \\(\lambda\\) (the eigenvalue).

**Confirm correctness**: explain in one sentence why this result justifies
the algorithm's choice of "the eigenvector with the largest eigenvalue as
the first principal component."
