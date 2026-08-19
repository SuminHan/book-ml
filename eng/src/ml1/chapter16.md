# Chapter 16. Block B Capstone: Team Project & ML1 Review

Chapters 9-15 covered the core lineage of modern deep learning — starting
from neural network basics, through CNNs, sequence models, the
Transformer, LLMs, and finally latent-variable generative models. This
chapter is Chapter 8's counterpart — instead of new concepts, it's a
chance to apply this second half's tools to real data and wrap up the
semester.

## 16.1 Project Structure

- **Data**: alongside Kaggle and UCI, add Hugging Face Datasets
  (huggingface.co/datasets) as a candidate — if your project touches the
  sequence/LLM material from Chapters 11-13, you can load text data in
  just a few lines.
- **Application**: apply at least one deep learning technique from
  Chapters 9-15 (CNN, RNN/Transformer, a prompting application, or an
  EM/VAE-family generative model) to real data.
- **Required — at least one mathematical justification**: for example (1)
  if you chose a CNN architecture, justify the design using Section 10.4's
  parameter-count/FLOPs formulas, (2) if you used a regularization
  technique (dropout, etc.), explain its effect via Section 9.6, or (3) for
  a generative-model project, connect your implementation to Chapter 15's
  ELBO or min-max objective.
- As in Chapter 8, follow the train/validation/test split (Section 6.3).

## 16.2 Presentation and Peer Review (Blocks 1-2)

The format is the same as Chapter 8 (problem definition → methodology →
results and justification → limitations). The peer-review checklist is
also the same, with one addition this time: "why deep learning instead of
classical ML?" — if the data is unstructured (images/text), explain why a
neural network was necessary; if it's tabular data and you still chose a
neural network, explain why, compared to Chapter 7's GBDT.

## 16.3 ML1 Review

This semester started with gradient descent for linear/logistic
regression, moved through generative classifiers (Naive Bayes/GDA),
distance- and margin-based models (kNN, SVM), tree ensembles (GBDT),
neural networks (backprop, CNN), sequences, attention, and LLMs, and
closed with unsupervised learning (PCA, embeddings) and latent-variable
generative models (EM/GMM, VAE, GAN, Diffusion).

| Chapter | What we learned | The key question |
|---|---|---|
| Ch02 | Linear/logistic regression | How do we predict a continuous value and a probability? |
| Ch03 | Naive Bayes/GDA | Can we classify by working backward from how the data was generated? |
| Ch04 | kNN/k-means | Can "closeness" alone drive prediction and clustering, and what are its limits? |
| Ch05 | SVM | Can we set a decision boundary by margin instead of probability? |
| Ch06 | Regularization/model selection | How do we control overfitting through the loss function? |
| Ch07 | Trees/random forest/GBDT/SHAP | How do we choose good splits, and how do we explain the resulting predictions? |
| Ch09 | Neural networks/backprop/training techniques | How do we train layers, and why is training a deep network hard? |
| Ch10 | CNN basics & applications | How do we efficiently handle and reuse an image's local structure? |
| Ch11 | Sequence models | How do we remember and process ordered data? |
| Ch12 | Attention & Transformer | How do we compute every word's relationships at once, without sequential processing? |
| Ch13 | LLM: pretraining/prompting/alignment | How does next-token prediction become broad capability, and how do we refine it toward what people want? |
| Ch14 | PCA/embeddings/PageRank | How do we find structure in data and graphs with no correct answers? |
| Ch15 | EM/GMM/VAE/GAN/Diffusion | How do we generate new data from an unobserved latent variable? |

Looking back at the common thread: **define a model → quantify how wrong
it is with a loss function → adjust parameters to reduce that loss** — this
structure repeated in nearly every chapter.

## 16.4 Before Moving On to ML2

ML1 was the world of supervised and unsupervised learning — "learning from
data that has a correct answer." ML2 starts from a completely different
question — instead of a correct answer, only a **reward** is given, and
instead of data existing in advance, an agent generates it by acting on
its own: the world of **reinforcement learning**. And its stage isn't text
or images, but **robot simulation environments** governed by the laws of
physics.

You don't need to have taken ML1 to start ML2 — ML2's Week 1 compresses
and reviews the neural-network fundamentals needed for this semester. That
said, if the feel of Chapter 9 (backpropagation) and Chapter 6
(regularization, especially cross-validation) has stuck with you, you'll
find ML2 considerably easier to follow.

**We'll close this semester with one principle we've confirmed again and
again: the question to ask before choosing any model is always the
same — what data does this problem actually have, and what am I trying to
predict?**
