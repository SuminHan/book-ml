# Chapter 13. LLM: Pretraining, Prompting & Alignment

A medical student doesn't start out learning about one specific disease —
they first spend years absorbing broad fundamentals like anatomy,
physiology, and pharmacology, and only then get intensively trained in a
specific specialty (say, cardiology) over a short period. Large language
models (LLMs) are trained in a remarkably similar way: they first absorb
broad knowledge about language and the world from a vast amount of
internet text, through the very simple task of "predicting the next word"
(**pretraining**), and are then refined with comparatively little data
toward a specific purpose — conversation, writing code, summarization —
(**fine-tuning**). The second half of this chapter goes further, covering
**post-training**, which refines that process even more precisely toward
"what people actually want."

## 13.1 Language Modeling: Predicting the Next Token

LLMs treat text as a sequence of **tokens** (whole words, or more commonly,
pieces of a word). Pretraining's goal is to predict the probability
distribution of the next token, given the tokens so far:

\\[P(x_t \mid x_1, x_2, \ldots, x_{t-1})\\]

To correctly complete the sentence "the capital of France is ___," the
model needs to have the fact "Paris" stored somewhere. Stack up billions of
sentences like this, and in order to do well at the simple goal of
"guessing the next word," the model naturally picks up grammar, factual
knowledge, and even a degree of reasoning ability.

This probability is computed using the Transformer from Chapter 12 (more
precisely, a decoder-only Transformer masked so it cannot see future
tokens). Training proceeds by minimizing the cross-entropy loss between
the actual next token and the predicted probability distribution (exactly
the same loss Chapter 2.6 introduced from Shannon's information theory).
Take this loss, swap base \\(e\\) for base 2, and exponentiate, and you get
**perplexity** — a measure of "how many choices the model is, on average,
effectively torn between" when predicting the next token; lower means the
model is more confident.

```python
def next_token_probs(logits):
    # logits: raw scores for each token in the vocabulary
    # applying softmax gives "the probability of each possible next token"
    return softmax(logits)
```

## 13.2 Pretraining vs. Fine-tuning

| | Pretraining | Fine-tuning |
|---|---|---|
| Data | Internet-scale amounts of text | Comparatively little data, targeted to a purpose |
| Goal | Next-token prediction (unsupervised) | Supervised learning for a specific task (conversation, instruction-following, etc.) |
| Result | Broad but unrefined ability | Behavior refined toward a purpose |

Pretraining costs an enormous amount of compute (weeks to months on
thousands of GPUs), while fine-tuning can be done with comparatively few
resources — the key to fine-tuning's efficiency is that it "refines what
the model already knows in a specific direction, without relearning the
fundamentals."

## 13.3 Prompting: Changing Behavior Without Retraining

Fine-tuning changes the model's weights themselves; **prompting** changes
the model's output using only its input (instructions, examples), without
touching the weights at all.

- **Zero-shot**: request something directly with just an instruction and
  no examples ("translate this sentence into Korean: ...").
- **Few-shot**: include a few input-output examples in the prompt, guiding
  the model to recognize and follow the pattern.
- **Chain-of-Thought**: prompts like "think step by step" that induce the
  model to produce intermediate reasoning before the final answer —
  often substantially improves accuracy on complex reasoning problems.

The fact that including a few examples in a prompt (few-shot prompting) can
make a model perform a brand-new kind of task is itself evidence of how
much broad capability pretraining compresses into the model. Pretraining
data already contains countless examples of "question-answer" and
"example-pattern" text, so once the model has learned these patterns
through next-token prediction, showing a similar pattern inside the prompt
makes it predict the next token in a way that continues that pattern —
this isn't learning new knowledge so much as the prompt pointing at which
existing capability to use.

LLMs sometimes confidently generate plausible-sounding sentences that
aren't true (**hallucination**) — because they were only trained to
predict "what's plausible as the next token," with no built-in process for
verifying facts.

## 13.4 Why Pretraining Alone Isn't Enough

In 2022, OpenAI reported a surprising observation in the InstructGPT
paper — a model with 100x fewer parameters, if refined the way people
actually wanted, was preferred by human raters far more often than a much
larger model that had only been pretrained. Section 13.1's pretraining
(next-token prediction) produces "plausible-sounding text," but that's no
guarantee it's "the answer a person actually wants" — pretraining data
mixes in brilliant writing alongside nonsensical or rude writing. From
here, we cover **post-training**, the stage that refines a pretrained model
in the direction people actually want.

**SFT (Supervised Fine-Tuning)**: the simplest form of post-training —
collect ideal (question, answer) pairs written by people, and train the
pretrained model on that data with Section 13.1's next-token-prediction
loss one more time, just with the data swapped from internet text to
hand-written, high-quality examples. The problem is scale: having a person
**write** "what a good answer looks like" from scratch, every time, is
expensive.

## 13.5 RLHF: Training a Reward Model From Comparisons, Not Writing

**RLHF** (Reinforcement Learning from Human Feedback) takes a different
strategy — instead of a person writing a fresh "correct answer" every
time, they only need to **pick which of several model-generated answers is
better** (comparing is much easier and faster than writing). This
preference data is used to train a **Reward Model**, and then the language
model itself is adjusted to maximize that reward model's score.

**Step 1 — Train the reward model**: given two answers \\(y_w\\) (the one
a person preferred, the "winner") and \\(y_l\\) (the "loser") to the same
question \\(x\\), train a reward model \\(r_\phi(x,y)\\) to assign a
higher score to \\(y_w\\). The Bradley-Terry model represents this
preference probability with a sigmoid:

\\[P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\\]

The loss that maximizes this probability is **exactly the same
cross-entropy form as Chapter 2's logistic regression** — because this is
a binary classification problem whose only "correct label" is the fact
that "\\(y_w\\) won":

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def reward_model_loss(r_win, r_lose):
    # Bradley-Terry: P(y_w > y_l) = sigmoid(r_win - r_lose)
    return -math.log(sigmoid(r_win - r_lose))
```

**Step 2 — Train the policy with reinforcement learning**: once the reward
model is ready, treat the language model as "a policy that sequentially
picks an answer \\(y\\) (one token at a time) as its action, given
question \\(x\\) as its state," and adjust its parameters using **reinforcement
learning** (a way of learning from problems framed around states, actions,
and rewards — not covered this semester; ML2 covers it as its dedicated
subject) with the reward model's score \\(r_\phi(x,y)\\) as the reward. The
most widely used algorithm here is **PPO** (Proximal Policy Optimization);
the detailed derivation is covered in ML2, but here's the shape of its
objective:

\\[J(\theta) = \mathbb{E}\left[r_\phi(x,y)\right] - \beta \, D_{KL}\big(\pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x)\big)\\]

Why the penalty term matters: the reward model is itself only an
approximation, so the policy risks exploiting its blind spots to rack up
score while actually producing strange text (**reward hacking**). Tying
the policy to the original pretrained (SFT) model \\(\pi_{\text{ref}}\\)
with a KL-divergence penalty (the same concept introduced in Chapter 2.6)
constrains it to raise reward only within the space of "text that still
makes sense."

## 13.6 DPO, PEFT/LoRA, and an Overview of Agents & RAG

**DPO** (Direct Preference Optimization, 2023): RLHF is a two-stage
process — (1) train a separate reward model, then (2) run reinforcement
learning against it — which can be complex and unstable. DPO shows
mathematically that the optimal policy for the RLHF objective can express
the reward function directly as a ratio between \\(\pi_\theta\\) and
\\(\pi_{\text{ref}}\\), and substituting that relationship into the
reward-model loss yields a loss that **learns the policy directly from
preference data, with no separate reward model and no reinforcement
learning loop at all** — replacing Section 13.5's entire two-stage process
with a single supervised loss:

\\[\mathcal{L}\_{\text{DPO}} = -\log \sigma\left(\beta \log
\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log
\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\\]

Look at the shape — it's still the exact same logistic loss as Section
13.5's reward-model loss, just with the role of "reward" played by
\\(\beta \log(\pi_\theta/\pi_{\text{ref}})\\) (how much more the policy now
prefers that answer compared to the reference model). DPO's appeal is
reaching the same theoretical destination as RLHF with a single round of
supervised learning, no separate reward model and no RL loop required.

**PEFT/LoRA** (Parameter-Efficient Fine-Tuning / Low-Rank Adaptation):
fine-tuning all tens or hundreds of billions of a model's parameters is
computationally and storage-wise expensive. LoRA freezes the original
weights \\(W\\) and instead trains only the product of two much smaller
low-rank matrices, \\(\Delta W = BA\\) (where \\(B, A\\) have far smaller
dimensions than \\(W\\)), using \\(W + \Delta W\\) as the new weights —
cutting the number of trainable parameters by hundreds of times while
achieving performance close to full fine-tuning in practice.

**Agents and RAG** (Retrieval-Augmented Generation): prompting alone can't
answer questions about information the model doesn't know or external
documents. RAG first retrieves documents relevant to the question and
includes them in the prompt; an **agent** goes a step further, letting the
model **call tools** — search, a calculator, code execution — on its own.
Both solve the same underlying problem in the same direction: instead of
cramming all knowledge into the model's parameters, let the model reach
for outside information/tools when it needs to.

**An LLM isn't a new algorithm — it's Chapter 12's Transformer, trained at
an extreme scale. If pretraining teaches "how to plausibly continue
anything," post-training is the stage that picks out just "what people
actually want" from that — and that refinement uses both supervised
learning (SFT, DPO) and reinforcement learning (RLHF) tools you're
learning about in this semester in different chapters.**

---

## Exercises

**1. (Coding)** Complete `next_token_distribution` below (a very simplified
language model simulation), and write `reward_model_loss` above and the
DPO loss `dpo_loss` (key lines left blank):

```python
def next_token_distribution(corpus_tokens, context_word):
    # ADD ADDITIONAL CODE HERE!!
    # at every occurrence of context_word, count the token that follows it,
    # and return a {next_word: probability} dictionary

corpus = "I go to school I go home I go to school again".split()
print(next_token_distribution(corpus, "I"))
# {"go": 1.0}

import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def reward_model_loss(r_win, r_lose):
    # ADD ADDITIONAL CODE HERE!!
    # -log(sigmoid(r_win - r_lose))

def dpo_loss(logp_win_theta, logp_win_ref, logp_lose_theta, logp_lose_ref, beta):
    # ADD ADDITIONAL CODE HERE!!
    # log_ratio_win = beta * (logp_win_theta - logp_win_ref)
    # log_ratio_lose = beta * (logp_lose_theta - logp_lose_ref)
    # -log(sigmoid(log_ratio_win - log_ratio_lose))

print(round(reward_model_loss(r_win=2.0, r_lose=-1.0), 3))  # 0.049 -- loss is small when well-separated
print(round(reward_model_loss(r_win=-1.0, r_lose=2.0), 3))  # 3.049 -- loss is large when reversed
```

**2. (Prompt design + conceptual)** (a) Indicate which of the following two
prompts uses the chain-of-thought technique, and explain why chain-of-thought
prompts tend to produce more accurate answers on complex arithmetic/logic
problems.

> (A) "What is 23×17? Just give me the answer."
>
> (B) "Compute 23×17. First show me how you calculate it step by step,
> then give the final answer."

(b) If you removed the KL-divergence penalty from RLHF's objective
entirely, what problem could arise? Explain using the term "reward
hacking."

**3. (Hand derivation, Tier C — fallback prepared)** Show that
differentiating the Bradley-Terry loss \\(\mathcal{L} =
-\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\\) with respect to
\\(r_\phi(x,y_w)\\) reduces to the same
\\((\text{prediction} - \text{label})\\)-shaped gradient you get from
differentiating Chapter 2's logistic regression loss.

**Hint**: substituting \\(z = r_\phi(x,y_w) - r_\phi(x,y_l)\\) gives
\\(\mathcal{L} = -\log\sigma(z)\\), which is exactly the logistic
regression loss \\(-\log h_w(x)\\) for "a true label that's always 1."
Reuse the result \\(\frac{d}{dz}(-\log\sigma(z)) = \sigma(z) - 1\\) you
derived in section 2.6 to find \\(\frac{\partial \mathcal{L}}{\partial
r_\phi(x,y_w)}\\).

**Confirm correctness**: looking at the sign of the gradient you found,
explain in one sentence why it approaches zero (i.e., stops updating much
further) once the model already confidently prefers \\(y_w\\)
(\\(\sigma(z) \to 1\\)).
