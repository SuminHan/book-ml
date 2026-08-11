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

## 4.4 The Tradeoff in Choosing \\(k\\)

- If \\(k\\) is too small (e.g. \\(k=1\\)): sensitive to a single noisy
  point — overfitting.
- If \\(k\\) is too large (e.g. \\(k=m\\), the whole dataset): always
  predicts the overall majority — ignores local patterns entirely
  (underfitting).

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

**Simplicity isn't automatically a weakness — kNN never explains what the
model "learned," but that very simplicity is exactly what lets us pin down,
with a formula, when and why it fails.**

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
