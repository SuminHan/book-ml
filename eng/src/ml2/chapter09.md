# Chapter 9. LLM Post-training: RLHF & Alignment

In 2022, OpenAI reported a surprising observation in the InstructGPT
paper — a model with 100x fewer parameters, if refined the way people
actually wanted, was preferred by human raters far more often than a
much larger model that had only been pretrained. Chapter 4's pretraining
(next-token prediction) produces "plausible-sounding text," but that's no
guarantee it's "the answer a person actually wants" — pretraining data
mixes in brilliant writing alongside nonsensical or rude writing. This
chapter covers **post-training**, the stage that refines a pretrained
model in the direction people actually want — and its central tool turns
out to be Chapter 8's PPO.

## 9.1 SFT: Refining With Ideal Answers a Person Wrote

The simplest form of post-training is **Supervised Fine-Tuning** (SFT):
collect ideal (question, answer) pairs written by people, and train the
pretrained model on that data with ordinary supervised learning one more
time — exactly Chapter 4's next-token-prediction loss, just with the data
swapped from internet text to hand-written, high-quality examples. The
problem is scale: having a person **write** "what a good answer looks
like" from scratch, every time, is expensive, and it's hard to cover the
huge variety of questions a model might encounter.

## 9.2 RLHF: Applying PPO Through a Reward Model

**RLHF** (Reinforcement Learning from Human Feedback) takes a different
strategy — instead of a person writing a fresh "correct answer" every
time, they only need to **pick which of several model-generated answers
is better** (comparing is much easier and faster than writing). This
preference data is used to train a **Reward Model**, and then Chapter 8's
PPO is applied to the language model itself, treated as a policy, to
maximize that reward model's score.

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

**Step 2 — Train the policy with PPO**: once the reward model is ready,
treat the language model \\(\pi_\theta\\) as "a policy that sequentially
picks an answer \\(y\\) (one token at a time) as its action, given
question \\(x\\) as its state," use the reward model's score
\\(r_\phi(x,y)\\) as the reward, and apply Chapter 8's PPO directly. One
term gets added, though — a KL-divergence penalty subtracted from the
objective, keeping the policy from straying too far from the original
pretrained (SFT) model \\(\pi_{\text{ref}}\\):

\\[J(\theta) = \mathbb{E}\left[r_\phi(x,y)\right] - \beta \, D_{KL}\big(\pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x)\big)\\]

Why this penalty matters: the reward model is itself only an
approximation, so the policy risks exploiting its blind spots to rack up
score while actually producing strange text (**reward hacking**). Tying
the policy to \\(\pi_{\text{ref}}\\) constrains it to raise reward only
within the space of "text that still makes sense" — this is exactly what
Chapter 8.10's mention of "tuning ChatGPT-style LLMs with RLHF" refers to.

## 9.3 DPO, PEFT/LoRA, and an Overview of Agents & RAG

**DPO** (Direct Preference Optimization, 2023): RLHF is a two-stage
process — (1) train a separate reward model, then (2) run PPO against it
— which can be complex and unstable. DPO shows mathematically that the
optimal policy for the RLHF objective can express the reward function
directly as a ratio between \\(\pi_\theta\\) and \\(\pi_{\text{ref}}\\),
and substituting that relationship into the reward-model loss yields a
loss that **learns the policy directly from preference data, with no
separate reward model and no PPO loop at all**:

\\[\mathcal{L}\_{\text{DPO}} = -\log \sigma\left(\beta \log
\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log
\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\\]

Look at the shape — it's still the exact same logistic loss as 9.2's
reward-model loss, just with the role of "reward" played by
\\(\beta \log(\pi_\theta/\pi_{\text{ref}})\\) (how much more the policy
now prefers that answer compared to the reference model). DPO's appeal is
reaching the same theoretical destination as RLHF with a single round of
supervised learning, no separate reward model and no RL loop required.

**PEFT/LoRA** (Parameter-Efficient Fine-Tuning / Low-Rank Adaptation):
fine-tuning all tens or hundreds of billions of a model's parameters is
computationally and storage-wise expensive. LoRA freezes the original
weights \\(W\\) and instead trains only the product of two much
smaller low-rank matrices, \\(\Delta W = BA\\) (where \\(B, A\\) have far
smaller dimensions than \\(W\\)), using \\(W + \Delta W\\) as the new
weights — cutting the number of trainable parameters by hundreds of times
while achieving performance close to full fine-tuning in practice.

**Agents and RAG** (Retrieval-Augmented Generation): prompting alone can't
answer questions about information the model doesn't know or external
documents. RAG first retrieves documents relevant to the question and
includes them in the prompt; an **agent** goes a step further, letting
the model **call tools** — search, a calculator, code execution — on its
own. Both solve the same underlying problem in the same direction:
instead of cramming all knowledge into the model's parameters, let the
model reach for outside information/tools when it needs to.

**If pretraining teaches "how to plausibly continue anything," post-training
is the stage that picks out just "what people actually want" from that —
and its central tool (PPO) is the very same algorithm we already learned
this semester to solve reinforcement learning problems.**

---

## Exercises

**1. (Coding)** Complete `reward_model_loss` above, and write the DPO
loss `dpo_loss` (key lines left blank):

```python
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

**2. (Conceptual)** If you removed the KL-divergence penalty
(\\(-\beta D_{KL}(\pi_\theta\|\pi_{\text{ref}})\\)) from RLHF's objective
entirely, what problem could arise? Explain in two or three sentences,
using the term "reward hacking."

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
