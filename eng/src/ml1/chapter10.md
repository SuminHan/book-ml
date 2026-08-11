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

## 10.2 k-means Clustering

An algorithm that partitions data into \\(k\\) groups, each represented by
a **centroid**.

1. Initialize \\(k\\) centroids randomly.
2. **Assignment step**: assign each data point to the group of its nearest
   centroid.
3. **Update step**: recompute each group's centroid as the average of the
   points assigned to it.
4. Repeat 2-3 until the assignments stop changing.

```python
def kmeans(X, k, max_iters=100):
    import random
    centroids = random.sample(X, k)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [sum((x[j]-c[j])**2 for j in range(len(x))) for c in centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(x)
        new_centroids = [
            [sum(pt[j] for pt in cluster) / len(cluster) for j in range(len(X[0]))]
            if cluster else centroids[i]
            for i, cluster in enumerate(clusters)
        ]
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters
```

**Choosing \\(k\\)**: a common approach is to plot within-cluster variance
against several values of \\(k\\) and pick the point where the decrease
sharply levels off (the "elbow").

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
