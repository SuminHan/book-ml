# Chapter 4. Distance-Based Models: kNN

"You are the average of your five closest friends," the saying goes.
Translate that sentence directly into an algorithm and you get
**k-Nearest Neighbors (kNN)**: to classify a new data point, there's no
training and no model needed — just look at what the \\(k\\) closest existing
points say, and go with the majority.

## 4.1 A Learning Algorithm With No Learning

Linear/logistic regression, which we covered earlier, had a **training
phase** where parameters \\(w\\) are optimized by looking at the data. kNN
has none of that — the only thing that could be called "training" is simply
storing the data. All the computation instead happens **at prediction
time**: when a new point arrives, only then do we measure its distance to
every stored point and find the closest ones. This approach is called lazy
learning, or instance-based learning.

## 4.2 The Algorithm

To classify a new point \\(x\\):

1. Compute the distance from \\(x\\) to every point \\(x^{(i)}\\) in the
   training data.
2. Pick the \\(k\\) closest ones.
3. Predict the majority vote (classification) or average (regression) of
   those \\(k\\) labels.

```python
def knn_predict(X_train, y_train, x_new, k):
    distances = []
    for i in range(len(X_train)):
        d = sum((X_train[i][j] - x_new[j]) ** 2 for j in range(len(x_new))) ** 0.5
        distances.append((d, y_train[i]))
    distances.sort(key=lambda p: p[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return max(set(k_nearest_labels), key=k_nearest_labels.count)  # majority vote
```

## 4.3 Distance Functions

The most common choice is **Euclidean distance**:

\\[d(x, x') = \sqrt{\sum_{j=1}^n (x_j - x'_j)^2}\\]

Another option is Manhattan distance (\\(\sum_j |x_j - x'_j|\\), moving only
along coordinate axes). **Feature normalization** is almost mandatory before
measuring distance — mixing "number of rooms" (range 0-10) with "house
price" (in the hundreds of thousands) means the large-scale feature
effectively dominates the distance entirely.

## 4.4 The Tradeoff in Choosing \\(k\\): Bias and Variance

- If \\(k\\) is too small (e.g. \\(k=1\\)): sensitive to a single noisy
  point — even a small change to the training data (a different sample)
  swings the prediction wildly. That's called having high **variance** —
  overfitting.
- If \\(k\\) is too large (e.g. \\(k=m\\), the whole dataset): always
  predicts the overall majority, ignoring local patterns entirely. It gives
  roughly the same simplistic prediction no matter what training data you
  feed it, so variance is low — but it's structurally incapable of
  capturing the pattern in the first place. That's called having high
  **bias** — underfitting.

This bias-variance relationship isn't specific to kNN — it's a principle
that applies across all of supervised learning. For a regression problem,
decomposing a model's expected error mathematically splits it into three
terms:

\\[\mathbb{E}\left[(y - \hat f(x))^2\right] =
\underbrace{\left(\text{Bias}[\hat f(x)]\right)^2}\_{\text{how far off the
model is structurally}} + \underbrace{\text{Var}[\hat f(x)]}\_{\text{how much
the prediction swings as training data changes}} +
\underbrace{\sigma^2}\_{\text{irreducible noise in the data itself}}\\]

A model that's too simple (large \\(k\\) in kNN, a shallow tree, a heavily
regularized linear model) has high bias and low variance — underfitting. A
model that's too flexible (\\(k=1\\) in kNN, a very deep tree, a neural
network with many parameters) has high variance and low bias — overfitting.
The two are always in this tradeoff, and the optimal model complexity is
wherever their sum (the total error) is minimized — this framework will
resurface whenever we ask "how do we prevent overfitting" in Chapter 5
(limiting tree depth / pruning) and Chapter 8 (neural network
regularization).

The right \\(k\\) is usually chosen by trying several values against a
validation set.

## 4.5 The Curse of Dimensionality

Intuition: as the number of features grows (higher dimensions), even the
"nearest" neighbors get farther and farther away. kNN's premise is that
nearby data has similar answers — but once there are hundreds of features,
that intuition starts to break down.

Let's get a feel for it with a formula. Suppose points are spread uniformly
inside the unit cube \\([0,1]^n\\) in \\(n\\) dimensions. To build a small
cube (sharing the same center) that contains a fraction \\(p\\) of the total
volume, its side length \\(\ell\\) must satisfy:

\\[\ell^n = p \quad\Longrightarrow\quad \ell = p^{1/n}\\]

With \\(p=0.01\\) (we want to contain just 1% of the total):

| \\(n\\) | \\(\ell = p^{1/n}\\) |
|---|---|
| 1 | 0.01 |
| 10 | 0.63 |
| 100 | 0.955 |

At \\(n=100\\), **95.5% of each side's length** is needed to contain just 1%
of the volume — the tight little region that could be called "nearby"
basically disappears. That's why, as the number of features grows, every
point starts to look about equally far from every other point.

## 4.6 k-means Clustering

If kNN is a lazy method that only measures distance at prediction time,
k-means is a way of finding structure in the data itself, using nothing
but distance, with no labels at all. Having no labels puts it in the
unsupervised learning territory we'll cover in Chapter 10, but its core
move — find the closest thing — is exactly this chapter's theme. It
partitions the data into \\(k\\) groups, each represented by a
**centroid**.

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

**Contrast with kNN**: kNN measures distance fresh at every prediction —
that's what makes it lazy. k-means instead sweeps over the data
repeatedly, updating centroids until they converge — the distance
computation is front-loaded into a "training" phase rather than
prediction time, which actually puts it closer to parameter learning like
linear regression. The only thing that separates it from supervised
learning is the absence of labels.

**Simplicity isn't automatically a weakness — neither kNN nor k-means
comes with an elaborate theory of what the model "learned," but that very
simplicity (pure distance computation) is exactly what lets us pin down,
with a formula, when and why they fail.**

---

## Exercises

**1. (Coding)** Extend `knn_predict` above (key lines left blank) into
`knn_predict_regression`, which returns the **average** instead of the
majority vote when labels are numeric (showing that kNN works for
regression too, not just classification):

```python
def knn_predict_regression(X_train, y_train, x_new, k):
    # ADD ADDITIONAL CODE HERE!!

X_train = [[1],[2],[3],[10],[11],[12]]
y_train = [10, 12, 11, 100, 102, 98]
print(knn_predict_regression(X_train, y_train, [2.5], k=3))  # approximately 11.0
```

**2. (Hand derivation, Tier B — hints provided)** Suppose points are
uniformly distributed in the unit cube \\([0,1]^n\\). Consider a small cube
(sharing the same center) with side length \\(\ell\\) that contains a
fraction \\(p\\) of the total volume.

**Step 1**: Derive \\(\ell = p^{1/n}\\). (Hint: the volume of an
\\(n\\)-dimensional cube is its side length raised to the \\(n\\)-th power.
Solve \\(\ell^n = p\\) for \\(\ell\\), so that the small cube's volume is
\\(p\\) times the total volume of 1.)

**Step 2**: Fix \\(p=0.01\\) and compute \\(\ell\\) directly for
\\(n=1, 10, 50, 100\\).

**Confirm correctness**: based on your calculations, explain in one
paragraph why the notion of "nearest \\(k\\)" progressively loses meaning
as the number of features (dimensions) grows.
