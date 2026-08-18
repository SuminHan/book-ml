# Chapter 2. Sequence Models

In 1990, cognitive scientist Jeffrey Elman posed a question in his paper
"Finding Structure in Time": could a neural network be made to remember not
just "the input it's looking at right now," but also "what it saw a moment
ago"? The structure he proposed — a recurrent connection feeding the hidden
layer's output back in as the next timestep's input — is the prototype of
what's now called the RNN (Recurrent Neural Network).

## 2.1 Why Order Matters

"The dog chases the cat" and "the cat chases the dog" use the exact same
three words but mean completely different things. Feeding a sentence like
this into the MLPs or CNNs we covered through ML1 requires bundling the
words into a single vector, which erases the order information entirely —
there's no longer any way to tell who is chasing whom.

## 2.2 Hidden State: A Summary of Everything Read So Far

RNN's core idea is the **hidden state**: as the network reads a sentence
one word at a time, it continuously updates a single vector that summarizes
"everything read so far." When processing the second word, the network
receives not just that word itself but also "the summary left over from
reading the first word" — this lets the current output be influenced by the
entire past context.

## 2.3 RNN Forward Propagation

At each timestep \\(t\\), given input \\(x_t\\) and the previous hidden
state \\(h_{t-1}\\), the network produces a new hidden state \\(h_t\\) and
(if needed) an output \\(y_t\\):

\\[h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h), \qquad y_t = W_{hy} h_t + b_y\\]

**The key point is that \\(W_{xh}, W_{hh}, W_{hy}\\) are the same weights
reused at every timestep** (parameter sharing — the same idea as reusing a
CNN filter, applied along the time axis instead).

```python
def rnn_step(x_t, h_prev, Wxh, Whh, b_h):
    z = [sum(Wxh[i][j]*x_t[j] for j in range(len(x_t))) +
         sum(Whh[i][j]*h_prev[j] for j in range(len(h_prev))) + b_h[i]
         for i in range(len(h_prev))]
    return [tanh(v) for v in z]

def rnn_forward(inputs, h0, Wxh, Whh, b_h):
    h = h0
    hidden_states = []
    for x_t in inputs:
        h = rnn_step(x_t, h, Wxh, Whh, b_h)
        hidden_states.append(h)
    return hidden_states
```

## 2.4 BPTT: Unrolling Time to Backpropagate

To train this "keep updating a summary" structure, backpropagation must be
applied by unrolling the network along the time axis. This method — treat
each of the \\(T\\) timesteps as an independent layer by "unrolling" the
network, then apply ordinary backpropagation — is called **BPTT**
(Backpropagation Through Time). Since \\(h_t\\) depends on \\(h_{t-1}\\),
which depends on \\(h_{t-2}\\), and so on, we must climb back up this chain,
so the gradient with respect to \\(h_1\\) ends up containing a product of
this form:

\\[\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^T \frac{\partial h_t}{\partial
h_{t-1}} = \prod_{t=2}^T \text{diag}(\tanh'(z_t)) \, W_{hh}\\]

## 2.5 Why Vanishing Gradients Reappear Along the Time Axis

This is exactly the same pattern we saw in ML1 Chapter 9: \\(\tanh'\\)'s
maximum value is 1, but it's below 1 across most of its range, and this
gets multiplied by \\(W_{hh}\\) as well, \\(T\\) times (once per timestep).
The longer the sequence (the larger \\(T\\)), the more this product shrinks
exponentially toward 0 (vanishing gradient — when \\(W_{hh}\\)'s eigenvalue
is less than 1) or diverges (exploding gradient — when its eigenvalue is
greater than 1). As a result, a basic RNN **barely remembers information
from far in the past** — it's especially weak on sentences that require
using "the subject that appeared 10 words ago" at the current timestep.

## 2.6 LSTM/GRU: Using Gates to Mitigate Vanishing

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) add a device
called a "gate," which **selectively keeps or updates** the hidden state
instead of recomputing it entirely at every timestep. The key trick is
designing the path information travels along to mix in **addition**
instead of multiplication — addition passes the gradient through unchanged
(its derivative is 1), so it vanishes far less than repeated multiplication
alone would. We won't cover the detailed gate equations this semester, but
the answer to "why are LSTM/GRU more robust to long sequences than a basic
RNN" always comes back to this principle.

## 2.7 The Fundamental Limitation of RNNs

Even with gates, an RNN still has to process one timestep at a time,
**sequentially** — to process the 100th word, you have to go through words
1 through 99 in order first. This sequential nature makes parallelization
hard, and information from far in the past still fades on very long
sequences. Attention/Transformer, which we cover in Chapter 3, is a
completely different approach that eliminates this sequential requirement
entirely.

**An RNN is the simplest implementation of the insight that "handling
ordered data requires a state that remembers the past" — the next two
chapters are the story of overcoming this structure's limitations.**

---

## Exercises

**1. (Coding)** Complete `rnn_forward_scalar` below (the simplest possible
RNN, with a scalar hidden state, \\(h_t = \tanh(w_{xh} x_t + w_{hh}
h_{t-1} + b_h)\\)) and `gradient_through_time` (key lines left blank):

```python
import math

def rnn_forward_scalar(inputs, h0, w_xh, w_hh, b_h):
    # ADD ADDITIONAL CODE HERE!!

print(rnn_forward_scalar([1.0, 1.0, 1.0], h0=0.0, w_xh=0.5, w_hh=0.8, b_h=0.0))

def gradient_through_time(tanh_derivatives, w_hh):
    # input: tanh_derivatives = [tanh'(z_1), ..., tanh'(z_T)]
    # return: product of (tanh'(z_t) * w_hh) for all t
    # ADD ADDITIONAL CODE HERE!!

print(gradient_through_time([0.5]*20, w_hh=0.9))  # (0.5*0.9)^20 -- effectively 0
print(gradient_through_time([0.9]*20, w_hh=1.1))  # (0.9*1.1)^20 -- close to 1
```

**2. (Hand derivation, Tier B — hints provided)** Show that
\\(\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^T \tanh'(z_t) \cdot
w_{hh}\\) (in the simplified case of a scalar hidden state), by applying
the chain rule repeatedly, \\(T-1\\) times (hint: multiply
\\(\frac{\partial h_t}{\partial h_{t-1}} = \tanh'(z_t) \cdot w_{hh}\\) as a
chain from \\(t=2\\) to \\(t=T\\)).

Assume \\(\tanh'(z_t) \approx 0.5\\) (a typical value) and \\(w_{hh}=0.9\\),
and compute the magnitude of \\(\frac{\partial h_T}{\partial h_1}\\) for
sequence lengths \\(T=5, 10, 20\\). Then repeat with \\(w_{hh}=1.5\\) and
check what happens at \\(T=20\\).

**Confirm correctness**: verify your calculations against Exercise 1's
`gradient_through_time` function, and explain in one paragraph why a
single value, \\(w_{hh}\\), determines whether the outcome is a vanishing
or exploding gradient.
