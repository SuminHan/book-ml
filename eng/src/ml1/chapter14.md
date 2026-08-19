# Chapter 14. Representation Learning: PCA, word2vec, Node2Vec, PageRank

In 1901, statistician Karl Pearson posed an interesting question: given
data with several measured variables, what's the "best line (or plane)"
that reduces the number of variables while preserving as much information
as possible? His answer — find the direction along which the data is most
spread out — is the origin of what's now called **Principal Component
Analysis (PCA)**. An idea from more than 120 years ago is still used today
to compress hundred-dimensional image embeddings down to a 2D space that
humans can actually look at.

## 14.1 Learning Without Correct Answers

Every model we've covered so far had a "correct answer" (\\(y\\)) — house
price, spam or not, a class label. **Unsupervised learning** has none of
that. Instead, it discovers the structure of the data \\(x\\) itself: "how
many natural groups do these customers fall into?" (clustering), "how many
dimensions of these 1000 features actually carry real information?"
(dimensionality reduction).

## 14.2 k-means Clustering (Recap)

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

## 14.3 Why We Want to Reduce Dimensionality

Recall the curse of dimensionality from Chapter 4 — with too many
features, the very notion of "closeness" breaks down. PCA compresses
features that are highly correlated (and thus effectively redundant) into
a handful of new axes (principal components), while preserving as much of
the original data's variance (information) as possible.

## 14.4 PCA: Finding the Axes That Preserve the Most Variance

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

## 14.5 Embeddings: word2vec and Node2Vec

While PCA compresses data that's already a vector, **embedding** is a way
of learning a low-dimensional vector representation for things that
weren't originally vectors at all (words, nodes in a graph). The goal is
the same as PCA's: **compress a high-dimensional (or non-vector) original
into a low-dimensional vector while preserving useful structure.**

### word2vec: Turning Words Into Vectors

Proposed in 2013 by Tomas Mikolov and colleagues, **word2vec** implements,
through learning, the intuition that "a word's meaning is determined by
what words tend to surround it" (the distributional hypothesis — "you
shall know a word by the company it keeps"). The **skip-gram** approach
trains a small neural network to predict the surrounding context words
from a single center word, and once training finishes, **the network's
weights themselves become each word's embedding vector** — "predict the
next word" is just the means; the vectors that fall out as a byproduct are
the actual goal (the same pattern as Chapter 13's observation that
next-token prediction leaves grammar and knowledge as a byproduct).

Vectors trained this way have a remarkable property — semantic
relationships between words show up as vector subtraction/addition:

\\[\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})\\]

Nobody explicitly taught the model "subtract maleness from king, add
femaleness, and you get queen" during training — the geometric structure
of the vector space organized itself that way on its own.

```python
import math, random

def train_skipgram(corpus, window=2, dim=8, epochs=50, lr=0.05, neg_k=3):
    # corpus: list of tokens (a single long sequence)
    vocab = sorted(set(corpus))
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    # center-word / context-word weight matrices
    W_in  = [[random.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(V)]
    W_out = [[random.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(V)]

    def sigmoid(z):
        return 1 / (1 + math.exp(-max(-20, min(20, z))))

    pairs = []
    for i, center in enumerate(corpus):
        for j in range(max(0, i - window), min(len(corpus), i + window + 1)):
            if j != i:
                pairs.append((idx[center], idx[corpus[j]]))

    for _ in range(epochs):
        random.shuffle(pairs)
        for c, o in pairs:
            # positive pair (c, o) + a few random negative words
            targets = [(o, 1)] + [(random.randrange(V), 0) for _ in range(neg_k)]
            for t, label in targets:
                z = sum(W_in[c][k] * W_out[t][k] for k in range(dim))
                pred = sigmoid(z)
                grad = (pred - label) * lr
                for k in range(dim):
                    g_in, g_out = W_in[c][k], W_out[t][k]
                    W_in[c][k]  -= grad * g_out
                    W_out[t][k] -= grad * g_in
    return {w: W_in[idx[w]] for w in vocab}
```

Remarkably, this training's gradient also starts with the exact same
`(pred - label)` factor as Chapter 2's logistic regression gradient
`(h_w(x) - y)`. That's not a coincidence: word2vec is trained as a
**binary classification** problem — "did word t actually appear next to
center word c (label=1), or was it a random fake word we mixed in
(label=0)?" (Mixing in `neg_k` random words as "fake answers" is
**negative sampling** — the key trick that avoids computing a softmax
over the entire vocabulary at every step. The code above is shrunk down
just to show the idea; real word2vec trains on millions of words and
billions of (center, context) pairs.)

### Node2Vec: The Same Idea, Applied to Graphs

**Node2Vec** reuses word2vec's skip-gram machinery as-is, but feeds it
node sequences generated by **random walks** over a graph instead of
sentences — starting from a node and randomly following its neighbors
produces a path that plays the role of a "sentence," with the nodes along
that path as its "words." The claim that "nodes with similar neighborhood
structure end up with similar vectors" is simply word2vec's "words that
appear in similar contexts end up with similar vectors" carried over to
graphs.

**Zachary's Karate Club** is the standard toy dataset for testing this
idea — in 1977, anthropologist Wayne Zachary recorded the friendships
among 34 members of a karate club, which happened to later split into two
real-world factions around the instructor (node 0) and the club president
(node 33). With just 34 nodes and 78 edges, it's tiny, but it's still used
today as the standard benchmark for checking whether a graph embedding
algorithm "actually works" — if the embedding, learned purely from graph
structure with no faction labels at all, still separates cleanly into the
two factions in vector space, that counts as success.

```python
KARATE_EDGES = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
    (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),
    (1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),
    (2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),
    (6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),
    (15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),
    (25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),
    (30,32),(30,33),(31,32),(31,33),(32,33),
]

def random_walk(neighbors, start, length):
    walk = [start]
    for _ in range(length - 1):
        cur = walk[-1]
        if not neighbors[cur]:
            break
        walk.append(random.choice(neighbors[cur]))
    return walk

def build_walks(edges, n_nodes, walks_per_node=10, walk_length=8):
    neighbors = {i: [] for i in range(n_nodes)}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    walks = []
    for node in range(n_nodes):
        for _ in range(walks_per_node):
            walks.append([str(n) for n in random_walk(neighbors, node, walk_length)])
    return walks
```

Concatenate the random walks (`corpus = sum(walks, [])`, where each walk
is a "sentence" and each node number is a "word") and feed them straight
into `train_skipgram` above, and out comes an embedding vector for each of
the 34 nodes — compress those down to 2D with the PCA from section 14.4
and plot them, and you'll actually see the two factions separate
spatially. **word2vec and Node2Vec are applied to different kinds of data
(text vs. graphs), but they're really two faces of the same idea: learn
vectors so that things that co-occur frequently end up close together.**

## 14.6 Same Random Walk, Different Question: PageRank

Node2Vec used the "sequence of nodes" a random walk produces. The same
random walk can be put to a completely different question: **if you
wander this graph at random forever, what fraction of the time do you
spend at each node?** The answer to that question is exactly what "node
importance" means in **PageRank**, the algorithm Larry Page and Sergey
Brin built Google on in 1998 — a page gets visited more often by a random
surfer following links if more pages link to it, and especially if
*important* pages link to it.

Node \\(i\\)'s PageRank score \\(PR(i)\\) must satisfy: the sum, over every
node \\(j\\) that links to it, of that node's own score split evenly across
its outgoing links (\\(\text{outdeg}(j)\\)):

\\[PR(i) = \frac{1-d}{N} + d\sum_{j \to i} \frac{PR(j)}{\text{outdeg}(j)}\\]

\\(N\\) is the total number of nodes, and \\(d\\) (usually 0.85) is the
**damping factor** — with probability \\(d\\) you follow a link, and with
probability \\(1-d\\) you teleport to a random node instead (a safeguard
against getting stuck in a dead end or trapped in a cycle).

This equation is **recursive** — \\(PR\\) appears again on the right-hand
side — so it can't be solved in one shot. Instead, start from any guess
(\\(1/N\\) for every node) and repeatedly plug the current values back into
the right-hand side until they stop changing — this is **power
iteration**:

```python
def pagerank(edges, n_nodes, d=0.85, iters=100):
    outgoing = {i: [] for i in range(n_nodes)}  # directed: a -> b
    for a, b in edges:
        outgoing[a].append(b)
    outdeg = {i: max(1, len(outgoing[i])) for i in range(n_nodes)}
    incoming = {i: [] for i in range(n_nodes)}  # who points to me?
    for a, b in edges:
        incoming[b].append(a)

    pr = {i: 1 / n_nodes for i in range(n_nodes)}
    for _ in range(iters):
        pr = {i: (1 - d) / n_nodes + d * sum(pr[j] / outdeg[j] for j in incoming[i])
              for i in range(n_nodes)}
    return pr

# Karate Club is undirected, so treat each friendship as a link both ways
directed_edges = KARATE_EDGES + [(b, a) for a, b in KARATE_EDGES]
scores = pagerank(directed_edges, 34)
print(sorted(scores.items(), key=lambda kv: -kv[1])[:3])
# node 33 and node 0 rank highest -- exactly the two real-world "hub" nodes
# (the club president and the instructor)
```

Why this power iteration converges is exactly the same argument as the
**Banach fixed point theorem** that **ML2 Chapter 4** uses to show the
Bellman optimality equation has a unique solution — "repeatedly applying
some transformation eventually settles at a fixed point that no longer
moves" is the same mathematics, just applied here to "a probability
distribution over a graph," and in reinforcement learning to "a state's
value function."

**PCA, word2vec, Node2Vec, and PageRank look like different problems at
first glance, but they all share one mathematical pattern: repeatedly
pushing something through some transformation makes it converge to a
special point (an eigenvector, an embedding, a stationary distribution).**

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

**2. (Conceptual)** word2vec uses skip-gram (predicting surrounding words
from a center word). If we instead used the reverse direction — predicting
the center word from its surrounding words (CBOW, Continuous
Bag-of-Words) — how would the training objective change, and why would you
still expect both approaches to end up producing similar embeddings?
Explain in two or three sentences.

**3. (Hand derivation, Tier B — hints provided)** For centered data
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
