# Problem Set

난이도 등급: **Tier B (적정하나 힌트 제공)**

**1.** (코딩) 다음과 같은 함수 `relu`와 `relu_prime`을 작성하라:

```python
def relu(z):
    # ADD ADDITIONAL CODE HERE!!

def relu_prime(z):
    # ADD ADDITIONAL CODE HERE!!
    # z > 0이면 1, 아니면 0 (z=0에서는 관례적으로 0으로 처리)

print([relu(z) for z in [-2, -0.5, 0, 1, 3]])        # [0, 0, 0, 1, 3]
print([relu_prime(z) for z in [-2, -0.5, 0, 1, 3]])  # [0, 0, 0, 1, 1]
```

**2.** (코딩) 다음과 같은 함수 `gradient_norm_through_layers`를 작성하라: 시그모이드를
쓰는 \\(L\\)개 층을 거치며 그래디언트가 얼마나 작아지는지 시뮬레이션한다.

- input parameter: 층 개수 `n_layers`, 각 층에서의 시그모이드 미분값 리스트
  `sigmoid_derivatives`(길이 `n_layers`, 모두 0~0.25 사이)
- return value: 모든 미분값을 곱한 값(그래디언트가 첫 층까지 도달했을 때 남은 비율)

```python
def gradient_norm_through_layers(n_layers, sigmoid_derivatives):
    # ADD ADDITIONAL CODE HERE!!

print(gradient_norm_through_layers(5, [0.2]*5))   # 0.2^5 = 0.00032
print(gradient_norm_through_layers(20, [0.2]*20)) # 0.2^20 -- 사실상 0
```

---

## 손유도 과제 (실습시간, Tier B — 힌트 제공)

### 그래디언트 소실/폭발을 활성함수 미분값으로 설명

**단계 1**: 시그모이드 \\(\sigma(z) = \frac{1}{1+e^{-z}}\\)의 미분
\\(\sigma'(z) = \sigma(z)(1-\sigma(z))\\)이 \\(z=0\\)에서 최댓값을 가짐을 보이고,
그 최댓값이 0.25임을 계산하라. (힌트: \\(f(p) = p(1-p)\\)를 \\(p\\)에 대해 미분해서
0으로 놓으면 \\(p=0.5\\)에서 최댓값을 가짐을 알 수 있다. \\(\sigma(0)=0.5\\)이다.)

**단계 2**: 10개 층을 가진 신경망에서, 각 층의 \\(\sigma'(z_l)\\)이 모두 최댓값인
0.25라고 가정하고, 문제 2의 함수를 이용해 그래디언트가 10개 층을 거치며 원래 크기의
몇 %로 줄어드는지 계산하라.

**단계 3**: ReLU를 썼다면 활성화된(양수) 뉴런에서 \\(\sigma'(z_l)=1\\)이다. 같은 10개
층 시나리오에서 그래디언트가 몇 %로 줄어드는지 계산하고, 시그모이드와 비교하라.

**정확성 확인**: 단계 2, 3의 계산 결과를 바탕으로, "층이 깊어질수록 시그모이드 기반
신경망은 학습이 왜 어려워지는가"를 한 문단으로 설명하고, ReLU가 이 문제를 완전히
해결하지는 못한다는 점(음수 구간에서는 미분이 0)도 함께 언급하라.
