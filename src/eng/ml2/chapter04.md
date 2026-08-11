# Chapter 4. LLM Preview

A medical student doesn't start out learning about one specific disease —
they first spend years absorbing broad fundamentals like anatomy,
physiology, and pharmacology, and only then get intensively trained in a
specific specialty (say, cardiology) over a short period. Large language
models (LLMs) are trained in a remarkably similar way: they first absorb
broad knowledge about language and the world from a vast amount of
internet text, through the very simple task of "predicting the next word"
(**pretraining**), and are then refined with comparatively little data
toward a specific purpose — conversation, writing code, summarization —
(**fine-tuning**).

## 4.1 Language Modeling: Predicting the Next Token

LLMs treat text as a sequence of **tokens** (whole words, or more commonly,
pieces of a word). Pretraining's goal is to predict the probability
distribution of the next token, given the tokens so far:

\\[P(x_t \mid x_1, x_2, \ldots, x_{t-1})\\]

To correctly complete the sentence "the capital of France is ___," the
model needs to have the fact "Paris" stored somewhere. Stack up billions of
sentences like this, and in order to do well at the simple goal of
"guessing the next word," the model naturally picks up grammar, factual
knowledge, and even a degree of reasoning ability.

This probability is computed using the Transformer from Chapter 3 (more
precisely, a decoder-only Transformer masked so it cannot see future
tokens). Training proceeds by minimizing the cross-entropy loss between
the actual next token and the predicted probability distribution (the same
form as the classification loss from before Chapter 3).

```python
def next_token_probs(logits):
    # logits: raw scores for each token in the vocabulary
    # applying softmax gives "the probability of each possible next token"
    return softmax(logits)
```

## 4.2 Pretraining vs. Fine-tuning

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

## 4.3 Prompting: Changing Behavior Without Retraining

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
much broad capability pretraining compresses into the model.

## 4.4 Why Prompting Works

Pretraining data already contains countless examples of "question-answer"
and "example-pattern" text. Since the model already learned these patterns
themselves through next-token prediction, showing a similar pattern inside
the prompt makes it predict the next token in a way that continues that
pattern — this isn't learning new knowledge so much as **the prompt
pointing at which existing capability to use.**

## 4.5 Limitations

LLMs sometimes confidently generate plausible-sounding sentences that
aren't true (**hallucination**) — because they were only trained to
predict "what's plausible as the next token," with no built-in process for
verifying facts. This limitation is still an active area of research and
goes beyond this semester's scope — but it's worth remembering that "why
this limitation exists" can be traced directly back to the pretraining
objective itself (next-token prediction).

**An LLM isn't a new algorithm — it's the Transformer we've already
learned, trained at an extreme scale (parameter count, data volume). "Why
does scale produce new capabilities" is still an actively researched
question today.**

---

## Exercises

**1. (Coding)** Complete `simple_tokenize` and `next_token_distribution`
below (a very simplified language model simulation, key lines left blank):

```python
def simple_tokenize(text):
    # ADD ADDITIONAL CODE HERE!!

def next_token_distribution(corpus_tokens, context_word):
    # ADD ADDITIONAL CODE HERE!!
    # at every occurrence of context_word, count the token that follows it,
    # and return a {next_word: probability} dictionary

corpus = simple_tokenize("I go to school I go home I go to school again")
print(next_token_distribution(corpus, "I"))
# {"go": 1.0}
```

**2. (Prompt design, not coding)** Indicate which of the following two
prompts uses the chain-of-thought technique, and explain in one paragraph
why chain-of-thought prompts tend to produce more accurate answers on
complex arithmetic/logic problems.

> (A) "What is 23×17? Just give me the answer."
>
> (B) "Compute 23×17. First show me how you calculate it step by step,
> then give the final answer."
