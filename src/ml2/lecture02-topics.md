# Topics Covered

## RNN의 순전파

시점 \\(t\\)마다 입력 \\(x_t\\)와 이전 은닉 상태 \\(h_{t-1}\\)을 받아, 새 은닉
상태 \\(h_t\\)와 (필요하면) 출력 \\(y_t\\)를 낸다:

\\[h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h), \qquad y_t = W_{hy} h_t + b_y\\]

**핵심은 \\(W_{xh}, W_{hh}, W_{hy}\\)가 모든 시점에서 동일한 가중치를 재사용한다는
것**이다(파라미터 공유 — CNN의 필터 재사용과 같은 아이디어를 시간 축에 적용한 것).

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

## BPTT: 시간을 펼쳐서 역전파하기

RNN을 학습시키려면, 시퀀스 길이 \\(T\\)만큼 신경망을 "펼쳐서(unroll)" 각 시점을
독립된 층처럼 취급한 뒤 일반적인 역전파를 적용한다 — 이를 **BPTT**(Backpropagation
Through Time)라 부른다. \\(h_t\\)가 \\(h_{t-1}\\)에, \\(h_{t-1}\\)이
\\(h_{t-2}\\)에 의존하는 사슬을 거슬러 올라가야 하므로, \\(h_1\\)에 대한
그래디언트는 다음과 같은 형태의 곱을 포함하게 된다:

\\[\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^T \frac{\partial h_t}{\partial
h_{t-1}} = \prod_{t=2}^T \text{diag}(\tanh'(z_t)) \, W_{hh}\\]

## 그래디언트 소실이 시간 축에서 재발하는 이유

ML1 W08에서 본 것과 정확히 같은 패턴이다: \\(\tanh'\\)의 최댓값은 1이지만 대부분의
구간에서 1보다 작고, 여기에 \\(W_{hh}\\)까지 \\(T\\)번(시퀀스 길이만큼) 곱해진다.
시퀀스가 길수록(\\(T\\)가 클수록) 이 곱은 지수적으로 0에 가까워지거나(그래디언트
소실 — \\(W_{hh}\\)의 고유값이 1보다 작을 때) 발산한다(그래디언트 폭발 —
고유값이 1보다 클 때). 결과적으로 기본 RNN은 **먼 과거의 정보를 거의 기억하지
못한다** — "10단어 전에 나온 주어"를 지금 시점에서 활용해야 하는 문장에서 특히
취약하다.

## LSTM/GRU: 게이트로 소실을 완화

LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Unit)는 "게이트(gate)"라는
장치를 추가해, 은닉 상태를 매 시점 완전히 새로 계산하는 대신 **선택적으로 유지하거나
갱신**한다. 핵심 트릭은 정보가 지나가는 경로에 곱셈 대신 **덧셈**이 섞이도록
설계하는 것이다 — 덧셈은 그래디언트를 그대로 통과시키므로(미분이 1), 곱셈만 반복될
때보다 소실이 훨씬 덜하다. 자세한 게이트 수식은 이번 학기에서는 다루지 않지만, "왜
LSTM/GRU가 기본 RNN보다 긴 시퀀스에 강한가"의 답은 항상 이 원리로 귀결된다.

## RNN의 근본적 한계

게이트를 추가해도 RNN은 여전히 **순차적으로** 한 시점씩 처리해야 한다 — 100번째
단어를 처리하려면 1번째부터 99번째까지 순서대로 다 거쳐야 한다. 이 순차성 때문에
병렬화가 어렵고, 아주 긴 시퀀스에서는 여전히 먼 과거 정보가 흐려진다. W03에서 배울
Attention/Transformer는 이 순차성 자체를 없애는 완전히 다른 접근이다.
