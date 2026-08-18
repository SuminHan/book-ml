# Chapter 2. 시퀀스 모델 (Sequence Models)

1990년, 인지과학자 제프리 엘먼(Jeffrey Elman)은 논문 "Finding Structure in
Time"에서 질문 하나를 던졌다: 신경망이 "지금 보고 있는 입력"뿐 아니라 "조금
전에 무엇을 봤는지"도 기억하게 만들 수 있을까? 그가 제안한 구조 — 은닉층의
출력을 다음 시점의 입력으로 다시 연결하는 순환(recurrent) 구조 — 는 지금
RNN(Recurrent Neural Network)이라 불리는 모델의 원형이다.

## 2.1 왜 순서가 중요한가

"강아지가 고양이를 쫓는다"와 "고양이가 강아지를 쫓는다"는 똑같은 단어 세
개로 이루어져 있지만 완전히 다른 문장이다. ML1까지 배운 MLP나 CNN에 이
문장을 넣으려면 단어들을 그냥 벡터 하나로 뭉쳐야 했는데, 그러면 순서
정보가 사라진다 — "누가 누구를 쫓는지"를 구분할 방법이 없어진다.

## 2.2 은닉 상태: 지금까지 읽은 것의 요약

RNN의 핵심 아이디어는 **은닉 상태**(hidden state)다: 문장을 한 단어씩
읽어가면서, "지금까지 읽은 내용의 요약"을 하나의 벡터에 계속 갱신해나간다.
두 번째 단어를 처리할 때는 두 번째 단어 자체뿐 아니라 "첫 번째 단어를
읽고 남은 요약"도 함께 입력받는다 — 이렇게 하면 지금 시점의 출력이 과거
전체의 맥락에 영향을 받을 수 있다.

## 2.3 RNN의 순전파

시점 \\(t\\)마다 입력 \\(x_t\\)와 이전 은닉 상태 \\(h_{t-1}\\)을 받아, 새
은닉 상태 \\(h_t\\)와 (필요하면) 출력 \\(y_t\\)를 낸다:

\\[h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h), \qquad y_t = W_{hy} h_t + b_y\\]

**핵심은 \\(W_{xh}, W_{hh}, W_{hy}\\)가 모든 시점에서 동일한 가중치를
재사용한다는 것**이다(파라미터 공유 — CNN의 필터 재사용과 같은 아이디어를
시간 축에 적용한 것).

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

## 2.4 BPTT: 시간을 펼쳐서 역전파하기

이 "요약을 계속 갱신한다"는 구조를 학습시키려면, 역전파를 시간 축을 따라
펼쳐서 적용해야 한다. 시퀀스 길이 \\(T\\)만큼 신경망을 "펼쳐서(unroll)"
각 시점을 독립된 층처럼 취급한 뒤 일반적인 역전파를 적용하는 이 방법을
**BPTT**(Backpropagation Through Time)라 부른다. \\(h_t\\)가
\\(h_{t-1}\\)에, \\(h_{t-1}\\)이 \\(h_{t-2}\\)에 의존하는 사슬을 거슬러
올라가야 하므로, \\(h_1\\)에 대한 그래디언트는 다음과 같은 형태의 곱을
포함하게 된다:

\\[\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^T \frac{\partial h_t}{\partial
h_{t-1}} = \prod_{t=2}^T \text{diag}(\tanh'(z_t)) \, W_{hh}\\]

## 2.5 그래디언트 소실이 시간 축에서 재발하는 이유

ML1 Chapter 9에서 본 것과 정확히 같은 패턴이다: \\(\tanh'\\)의 최댓값은
1이지만 대부분의 구간에서 1보다 작고, 여기에 \\(W_{hh}\\)까지
\\(T\\)번(시퀀스 길이만큼) 곱해진다. 시퀀스가 길수록(\\(T\\)가 클수록)
이 곱은 지수적으로 0에 가까워지거나(그래디언트 소실 — \\(W_{hh}\\)의
고유값이 1보다 작을 때) 발산한다(그래디언트 폭발 — 고유값이 1보다 클
때). 결과적으로 기본 RNN은 **먼 과거의 정보를 거의 기억하지 못한다** —
"10단어 전에 나온 주어"를 지금 시점에서 활용해야 하는 문장에서 특히
취약하다.

## 2.6 LSTM/GRU: 게이트로 소실을 완화

LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Unit)는
"게이트(gate)"라는 장치를 추가해, 은닉 상태를 매 시점 완전히 새로
계산하는 대신 **선택적으로 유지하거나 갱신**한다. 핵심 트릭은 정보가
지나가는 경로에 곱셈 대신 **덧셈**이 섞이도록 설계하는 것이다 — 덧셈은
그래디언트를 그대로 통과시키므로(미분이 1), 곱셈만 반복될 때보다 소실이
훨씬 덜하다. 자세한 게이트 수식은 이번 학기에서는 다루지 않지만, "왜
LSTM/GRU가 기본 RNN보다 긴 시퀀스에 강한가"의 답은 항상 이 원리로
귀결된다.

## 2.7 RNN의 근본적 한계

게이트를 추가해도 RNN은 여전히 **순차적으로** 한 시점씩 처리해야 한다 —
100번째 단어를 처리하려면 1번째부터 99번째까지 순서대로 다 거쳐야 한다.
이 순차성 때문에 병렬화가 어렵고, 아주 긴 시퀀스에서는 여전히 먼 과거
정보가 흐려진다. Chapter 3에서 배울 Attention/Transformer는 이 순차성
자체를 없애는 완전히 다른 접근이다.

**RNN은 "순서가 있는 데이터를 다루려면 과거를 기억하는 상태가 필요하다"는
통찰의 가장 단순한 구현이다 — 다음 두 장은 이 구조의 한계를 극복하는
이야기다.**

---

## 연습문제

**1. (코딩)** 다음과 같은 함수 `rnn_forward_scalar`(은닉 상태가 스칼라인
가장 단순한 RNN, \\(h_t = \tanh(w_{xh} x_t + w_{hh} h_{t-1} + b_h)\\))와
`gradient_through_time`을 완성하라(핵심 줄은 빈칸으로 남겨져 있다고
가정):

```python
import math

def rnn_forward_scalar(inputs, h0, w_xh, w_hh, b_h):
    # ADD ADDITIONAL CODE HERE!!

print(rnn_forward_scalar([1.0, 1.0, 1.0], h0=0.0, w_xh=0.5, w_hh=0.8, b_h=0.0))

def gradient_through_time(tanh_derivatives, w_hh):
    # input: tanh_derivatives = [tanh'(z_1), ..., tanh'(z_T)]
    # return: product of (tanh'(z_t) * w_hh) for all t
    # ADD ADDITIONAL CODE HERE!!

print(gradient_through_time([0.5]*20, w_hh=0.9))  # (0.5*0.9)^20 -- 사실상 0
print(gradient_through_time([0.9]*20, w_hh=1.1))  # (0.9*1.1)^20 -- 1에 가까움
```

**2. (손유도, Tier B — 힌트 제공)** \\(\frac{\partial h_T}{\partial h_1} =
\prod_{t=2}^T \tanh'(z_t) \cdot w_{hh}\\)(은닉 상태가 스칼라인 단순화된
경우)임을, 연쇄법칙을 \\(T-1\\)번 반복 적용하는 방식으로 보여라(힌트:
\\(\frac{\partial h_t}{\partial h_{t-1}} = \tanh'(z_t) \cdot w_{hh}\\)를
\\(t=2\\)부터 \\(t=T\\)까지 사슬처럼 곱하면 된다).

\\(\tanh'(z_t) \approx 0.5\\)(평균적인 경우), \\(w_{hh}=0.9\\)라고 하고,
시퀀스 길이 \\(T=5, 10, 20\\)일 때 각각 \\(\frac{\partial h_T}{\partial
h_1}\\)의 크기를 계산하라. 이번엔 \\(w_{hh}=1.5\\)로 바꿔서 같은 계산을
반복하고, \\(T=20\\)일 때 어떤 일이 일어나는지 확인하라.

**정확성 확인**: 계산 결과를 문제 1의 `gradient_through_time` 함수로
검증하고, "\\(w_{hh}\\)의 값 하나가 그래디언트 소실이 될지 폭발이 될지를
가른다"는 문장이 왜 참인지 한 문단으로 설명하라.
