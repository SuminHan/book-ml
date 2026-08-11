# Problem Set

난이도 등급: **Tier B (적정하나 힌트 제공)**

**1.** (코딩) 다음과 같은 함수 `rnn_forward_scalar`를 작성하라 (은닉 상태가 스칼라인
가장 단순한 RNN):

- input parameter: 입력 시퀀스 `inputs`(숫자 리스트), 초기 은닉 상태 `h0`, 가중치
  `w_xh`, `w_hh`, 편향 `b_h`
- return value: 모든 시점의 은닉 상태 리스트 `[h_1, h_2, ..., h_T]`
- \\(h_t = \tanh(w_{xh} x_t + w_{hh} h_{t-1} + b_h)\\)

```python
import math

def rnn_forward_scalar(inputs, h0, w_xh, w_hh, b_h):
    # ADD ADDITIONAL CODE HERE!!

print(rnn_forward_scalar([1.0, 1.0, 1.0], h0=0.0, w_xh=0.5, w_hh=0.8, b_h=0.0))
```

**2.** (코딩) 다음과 같은 함수 `gradient_through_time`을 작성하라 — ML1 W08의
`gradient_norm_through_layers`와 유사하지만, 매 시점 \\(\tanh'(z_t)\\)와
\\(w_{hh}\\)를 함께 곱하는 BPTT 시뮬레이션이다.

```python
def gradient_through_time(tanh_derivatives, w_hh):
    # input: tanh_derivatives = [tanh'(z_1), ..., tanh'(z_T)]
    # return: product of (tanh'(z_t) * w_hh) for all t
    # ADD ADDITIONAL CODE HERE!!

print(gradient_through_time([0.5]*20, w_hh=0.9))  # (0.5*0.9)^20 -- 사실상 0
print(gradient_through_time([0.9]*20, w_hh=1.1))  # (0.9*1.1)^20 -- 1에 가까움
```

---

## 손유도 과제 (실습시간, Tier B — 힌트 제공)

### BPTT(시간에 따른 역전파) 그래디언트 소실 수식 확인

**단계 1**: \\(\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^T \tanh'(z_t) \cdot
w_{hh}\\) (은닉 상태가 스칼라인 단순화된 경우)임을, 연쇄법칙을 \\(T-1\\)번 반복
적용하는 방식으로 보여라. (힌트: \\(h_t = \tanh(w_{xh}x_t + w_{hh}h_{t-1} + b_h)\\)이므로
\\(\frac{\partial h_t}{\partial h_{t-1}} = \tanh'(z_t) \cdot w_{hh}\\)이다. 이걸
\\(t=2\\)부터 \\(t=T\\)까지 사슬처럼 곱하면 된다.)

**단계 2**: \\(\tanh'(z_t) \approx 0.5\\)(평균적인 경우), \\(w_{hh}=0.9\\)라고 하자.
시퀀스 길이 \\(T=5, 10, 20\\)일 때 각각 \\(\frac{\partial h_T}{\partial h_1}\\)의
크기를 계산하라.

**단계 3**: 이번엔 \\(w_{hh}=1.5\\)로 바꿔서 같은 계산을 반복하라. \\(T=20\\)일 때
어떤 일이 일어나는가?

**정확성 확인**: 단계 2, 3의 결과를 문제 2의 `gradient_through_time` 함수로 검증하고,
"\\(w_{hh}\\)의 값 하나가 그래디언트 소실이 될지 폭발이 될지를 가른다"는 문장이 왜
참인지 한 문단으로 설명하라.
