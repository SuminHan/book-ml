# Chapter 5. Tree-Based Models

"Is it an animal? Does it walk on four legs? Does it meow?" — the game of 20
Questions works because, if you pick your yes/no questions well, you can
narrow down almost any answer within twenty tries. A **Decision Tree**
translates this game directly into an algorithm: pick, one at a time, the
question that splits the data best ("is age greater than 30?"), and repeat
until each question has narrowed the answer down enough.

## 5.1 The Structure of a Decision Tree

Each internal node is a question (e.g., "\\(x_2 > 5\\)?"), and each leaf
node is a prediction. To predict, start at the root and descend to a leaf by
answering each question along the way.

## 5.2 What Makes a "Good" Question: Gini Impurity

A good 20-Questions player doesn't ask just any question — asking something
like "is it alive?" that splits the possible answers roughly in half yields
the most information. **Gini Impurity** measures how "mixed" the data in a
node is. With \\(K\\) classes and class \\(k\\) making up a fraction
\\(p_k\\):

\\[G = 1 - \sum_{k=1}^K p_k^2\\]

If a node is pure (all one class, so \\(p_k=1\\) for one \\(k\\) and 0 for
the rest), \\(G=0\\) — the best case. If the classes are evenly split
(\\(K=2\\), \\(p_1=p_2=0.5\\)), \\(G = 1 - 0.25 - 0.25 = 0.5\\) — the worst
case (maximum) for binary classification.

## 5.3 Information Gain

First, define **entropy**:

\\[H = -\sum_{k=1}^K p_k \log_2 p_k\\]

Like Gini impurity, this is small (0) when pure and large when mixed. If a
question splits a node into left (\\(L\\)) and right (\\(R\\)) children,
**information gain** is "entropy before the split, minus the (weighted
average) entropy after":

\\[\text{IG} = H(\text{parent}) - \left(\frac{|L|}{|L|+|R|}H(L) +
\frac{|R|}{|L|+|R|}H(R)\right)\\]

The larger the information gain, the better that question split the data.
When building a decision tree, at every step we choose **whichever question
gives the largest information gain (or Gini impurity reduction)** — this is
the algorithmic version of "ask the question that narrows things down the
most" from 20 Questions.

```python
import math

def gini(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return 1 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def information_gain(parent_labels, left_labels, right_labels):
    n = len(parent_labels)
    weighted_child = (len(left_labels) / n) * entropy(left_labels) + \
                      (len(right_labels) / n) * entropy(right_labels)
    return entropy(parent_labels) - weighted_child
```

## 5.4 When to Stop Splitting

If you let a tree grow fully, it keeps splitting until every leaf is
perfectly pure — the training data is matched 100%, but the resulting tree
is fragile (overfit) on new data. In practice, tree size is limited with
**stopping criteria** like maximum depth or a minimum number of samples per
leaf, or the fully-grown tree is trimmed back afterward by removing
unnecessary branches (**pruning**).

## 5.5 From One Tree to a Forest: Random Forests

A single tree can easily memorize the training data perfectly (overfitting).
The idea behind a **Random Forest** is simple: build many trees that each
differ slightly from one another, and predict by majority vote across them.
Two sources of randomness are added:

1. **Bagging (Bootstrap Aggregating)**: each tree is trained on a
   same-sized dataset drawn with replacement from the original data — so
   every tree sees slightly different data.
2. **Feature randomization**: at each split, instead of considering all
   features, only a randomly chosen subset is considered for the best
   question.

These two sources of randomness make the trees less like one another, so
that when you take the majority vote (or average), each individual tree's
overfitting tendencies tend to cancel out — the same principle behind why
the average of "100 experts who all make the exact same mistakes" is far
less reliable than the average of "100 experts who each make different
mistakes." Each individual tree makes its own biased errors, but if those
errors point in different directions, averaging cancels them out — this is
the intuition behind ensembling: ask several experts separately, then go
with the majority.

**A decision tree is simultaneously a tree of human-readable rules (if-then)
and an algorithm whose "how good is this split" can be measured exactly with
a formula — these two properties are why it's still widely used in
practice.**

---

## Exercises

**1. (Coding)** Complete `gini` and `best_split` below (key lines left
blank):

```python
def gini(labels):
    # ADD ADDITIONAL CODE HERE!!

def weighted_gini(left_labels, right_labels):
    # ADD ADDITIONAL CODE HERE!!
    # Gini impurity weighted by left/right node sizes

def best_split(X, y, feature_idx):
    best_threshold, best_gain = None, -1
    parent_gini = gini(y)
    for threshold in sorted(set(row[feature_idx] for row in X)):
        left_y = [y[i] for i in range(len(X)) if X[i][feature_idx] <= threshold]
        right_y = [y[i] for i in range(len(X)) if X[i][feature_idx] > threshold]
        if not left_y or not right_y:
            continue
        gain = parent_gini - weighted_gini(left_y, right_y)
        if gain > best_gain:
            best_threshold, best_gain = threshold, gain
    return best_threshold, best_gain

X = [[2.0],[3.0],[4.0],[7.0],[8.0],[9.0]]
y = ["A","A","A","B","B","B"]
print(best_split(X, y, 0))  # (4.0, 1.0) -- splits perfectly
```

**2. (Hand derivation, Tier A — free derivation)** Consider 8 samples:
classes `[A,A,A,A,B,B,B,B]`, feature values \\(x\\) = `[1,2,3,4,5,6,7,8]`.

Compute the Gini impurity of all 8 by hand, then compute the information
gain (Gini impurity reduction) of the split at \\(x \le 4\\) vs. \\(x > 4\\)
(threshold=4). Compare it against the information gain of splitting at
threshold=2 (left: `[A,A]`, right: `[A,A,B,B,B,B]`) — which split does a
better job of separating the data? Confirm your result matches the output
of `best_split(X, y, 0)` from Exercise 1.
