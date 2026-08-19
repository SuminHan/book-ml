# Chapter 12. Attention & Transformer

In 2017, researchers at Google Brain gave their paper a rather provocative
title: **"Attention Is All You Need."** Until then, the best-performing
sequence models were all built on RNNs (or their variant, LSTM). This paper
stripped out the recurrent structure entirely and achieved better
performance than RNNs using only an **attention** mechanism. This structure
is the **Transformer**, and it's the foundation of nearly every large
language model (LLM) we use today.

## 12.1 The Fundamental Problem With RNN Sequentiality

The RNN we covered in Chapter 11 has to process words one at a time, in
order — to see the 100th word, you must go through words 1 through 99 in
sequence first. This creates two problems: (1) GPUs are built for parallel
computation, but a sequential structure can't exploit that parallelism, so
it's slow. (2) Information from far in the past fades as it passes through
many steps (exactly the vanishing gradient problem from Chapter 11).

## 12.2 The Idea Behind "Attention"

Attention's intuition is simple: "to understand this word, look at **every
other word in the sentence at once**, and pay more attention to whichever
ones are relevant." In the sentence "the animal didn't cross the road
because **it** was too tired," figuring out whether "it" refers to "the
animal" or "the road" requires looking at the whole sentence at once.

## 12.3 Query, Key, Value

Each word (more precisely, each word's embedding vector) is transformed
into three vectors: **Query** (Q) — "what am I looking for right now,"
**Key** (K) — "what information do I hold," **Value** (V) — "what I
actually pass along." All three are obtained by multiplying the same input
embedding \\(x\\) by three different learnable weight matrices:

\\[Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V\\]

## 12.4 Scaled Dot-Product Attention

How "relevant" one word's Query is to every word's Key is measured via a
dot product:

\\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\\]

Breaking this down step by step:

1. \\(QK^T\\): a single matrix computing the Query-Key dot product for
   every pair of words at once (each row = the raw relevance score of one
   word to every other word).
2. Divide by \\(\sqrt{d_k}\\) (the square root of the Key vector's
   dimension): dot products tend to grow larger with dimension, so this
   rescaling prevents softmax from producing only extreme values (near 0
   or 1).
3. **Softmax** turns each row into a probability distribution (summing to
   1) — "the proportion of attention this word distributes to the
   others."
4. That probability is used to take a weighted sum of \\(V\\) (what each
   word actually contributes) — the result is a new vector that mixes in
   the content of relevant words, weighted by how relevant they are.

```python
import math

def softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]

def attention(Q, K, V, d_k):
    scores = [[sum(Q[i][t] * K[j][t] for t in range(d_k)) / math.sqrt(d_k)
               for j in range(len(K))] for i in range(len(Q))]
    weights = [softmax(row) for row in scores]
    output = [[sum(weights[i][j] * V[j][t] for j in range(len(V)))
               for t in range(len(V[0]))] for i in range(len(Q))]
    return output, weights
```

## 12.5 Self-Attention: A Sentence Looking at Itself

When Q, K, and V all come from **the same sentence**, this is called
"self-attention" — each word computes its relevance to every other word in
the same sentence (including itself). This is the mechanism behind
figuring out whether "it" refers to "the animal" from this chapter's
opening example: "it"'s Query is trained to produce its highest dot-product
score against "the animal"'s Key, when computed against every word's Key
in the sentence.

## 12.6 Multi-Head Attention

Instead of using just one set of Q, K, V, several sets ("heads") are run
in parallel, each learning relevance from a different perspective — it's
commonly observed that one head focuses on grammatical relationships
(subject-verb), while another focuses on semantic relationships (synonyms).
The results from all heads are concatenated and passed through one more
linear transformation to produce the final output.

## 12.7 Positional Encoding: How Order Gets Injected

The attention operation itself doesn't distinguish order at all —
\\(QK^T\\) computes the same relevance score for each pair of words
regardless of their order (it treats them like a set). But in "the dog
chases the cat," order changes the meaning. So the Transformer **adds** a
vector encoding each word's position (positional encoding — a fixed
pattern built from sine/cosine functions) to each word's embedding — this
way, the same word at a different position produces a different input
vector, letting attention exploit order indirectly.

## 12.8 Advantages Over RNN

Self-attention computes every pair of words **all at once**, in
parallel — there's no need for sequential processing, so it fully exploits
GPU parallelism, and even "the relevance between word 1 and word 100" is
computed directly, without passing through 99 intermediate steps
(structurally far less prone to vanishing gradients). The tradeoff is a
new cost that scales with the square of the sentence length (\\(QK^T\\) is
an \\(n \times n\\) matrix) — this becomes a practical limitation when
handling very long documents.

**A single idea — look at every word at once, and compute relevance
directly between all of them — is the biggest change in deep learning over
the past several years.**

---

## Exercises

**1. (Coding)** Complete `scaled_dot_product_attention` below (key lines
left blank):

```python
import math

def softmax(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    total = sum(exps)
    return [e / total for e in exps]

def scaled_dot_product_attention(Q, K, V, d_k):
    # ADD ADDITIONAL CODE HERE!!
    # 1. scores[i][j] = (Q[i] . K[j]) / sqrt(d_k)
    # 2. weights = apply softmax to each row
    # 3. output[i] = weighted sum of V using weights[i]

    return output, weights

Q = [[1,0],[0,1]]
K = [[1,0],[0,1]]
V = [[10,0],[0,10]]
output, weights = scaled_dot_product_attention(Q, K, V, d_k=2)
print(weights)  # each word gives more weight to "itself"
```

**2. (Conceptual)** Explain why self-attention is less prone to vanishing
gradients than an RNN, from the perspective of "the path length between
word 1 and word 100" (compare with Chapter 11's BPTT).

**3. (Hand derivation, Tier C — fallback prepared)** For a 2-word sentence,
you're given \\(Q = \begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}\\),
\\(K = \begin{pmatrix}1 & 1\\ 1 & 0\end{pmatrix}\\), \\(V =
\begin{pmatrix}5 & 0\\ 0 & 5\end{pmatrix}\\) (\\(d_k=2\\)).

Compute \\(QK^T\\) by hand, divide by \\(\sqrt{d_k}\\), apply softmax to
each row to get the attention weight matrix, and finally compute the
output as a weighted sum with \\(V\\).

**Fill-in-the-blank fallback version** (if free calculation is too
difficult):

```
Step 1: compute QK^T (a 2x2 matrix)
  (QK^T)[0][0] = Q[0].K[0] = 1*1 + 0*1 = ______________
  (QK^T)[0][1] = Q[0].K[1] = 1*1 + 0*0 = ______________
  (QK^T)[1][0] = Q[1].K[0] = 0*1 + 1*1 = ______________
  (QK^T)[1][1] = Q[1].K[1] = 0*1 + 1*0 = ______________

Step 2: divide every element by sqrt(d_k) = sqrt(2) ≈ 1.41
  scaled[0] = [______________, ______________]
  scaled[1] = [______________, ______________]

Step 3: apply softmax to the first row
  exp(scaled[0][0]) = ______________  (use a calculator)
  exp(scaled[0][1]) = ______________
  weights[0] = [______________, ______________]  (normalized to sum to 1)

Step 4: output[0] = weights[0][0] * V[0] + weights[0][1] * V[1]
       = [______________, ______________]
```

**Confirm correctness**: check your completed calculation against the
output of `scaled_dot_product_attention(Q, K, V, d_k=2)` from Exercise 1.
