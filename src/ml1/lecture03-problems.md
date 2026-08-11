# Problem Set

난이도 등급: **Tier B**(적정하나 힌트 제공)

**1.** (코딩) 다음과 같은 함수 `logistic_gradient_descent`를 작성하라:

- input parameter: 특징행렬 `X`, 타겟 `y`(0 또는 1의 리스트), 학습률 `alpha`, 반복횟수
  `epochs`
- return value: 학습된 가중치 벡터 `w` (bias 포함)
- \\(h_w(x) = \sigma(w^Tx)\\)이고, 그래디언트는 \\(\frac{\partial J}{\partial w_j} =
  \frac{1}{m}\sum_i (h_w(x^{(i)}) - y^{(i)}) x_j^{(i)}\\)이다(W02와 동일한 형태임에
  주의).

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def logistic_gradient_descent(X, y, alpha, epochs):
    m, n = len(X), len(X[0])
    w = [0.0] * (n + 1)
    for _ in range(epochs):
        # ADD ADDITIONAL CODE HERE!!
        # h_w(x) = sigmoid(w^T x)를 각 샘플에 대해 계산하고,
        # W02와 같은 형태의 그래디언트로 w를 갱신

    return w
```

**2.** (코딩) 다음과 같은 함수 `precision_recall_f1`을 작성하라:

- input parameter: `y_true`(정답 0/1 리스트), `y_pred`(예측 0/1 리스트)
- return value: `(precision, recall, f1)` 튜플

```python
def precision_recall_f1(y_true, y_pred):
    # ADD ADDITIONAL CODE HERE!!
    # TP, FP, FN을 센 뒤 precision, recall, f1 계산
    # (0으로 나누는 경우는 각 값을 0.0으로 처리)

y_true = [1, 1, 1, 0, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
print(precision_recall_f1(y_true, y_pred))  # (0.75, 0.75, 0.75)
```

---

## 손유도 과제 (실습시간, Tier B — 힌트 제공)

### 로지스틱 손실함수의 그래디언트 손유도

교차 엔트로피 손실 하나의 샘플에 대해:

\\[J^{(i)}(w) = -y^{(i)}\log h_w(x^{(i)}) - (1-y^{(i)})\log(1-h_w(x^{(i)}))\\]

를 \\(w_j\\)로 미분해서 \\(\frac{\partial J^{(i)}}{\partial w_j} = (h_w(x^{(i)}) -
y^{(i)})x_j^{(i)}\\)가 됨을 유도하라.

**힌트**(연쇄법칙을 세 단계로 나눠서 적용):

1. 먼저 \\(\frac{\partial J^{(i)}}{\partial h}\\)를 구하라 (h는 \\(h_w(x^{(i)})\\)의
   줄임 표기). \\(\frac{d}{dh}\log h = \frac{1}{h}\\)임을 이용.
2. 그 다음 \\(\sigma'(z) = \sigma(z)(1-\sigma(z))\\)임을 이용해 \\(\frac{\partial
   h}{\partial z}\\)를 구하라 (여기서 \\(z = w^Tx^{(i)}\\)).
3. 마지막으로 \\(\frac{\partial z}{\partial w_j} = x_j^{(i)}\\)임을 이용해 연쇄법칙
   \\(\frac{\partial J^{(i)}}{\partial w_j} = \frac{\partial J^{(i)}}{\partial h}
   \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial w_j}\\)로
   세 조각을 곱하면, 놀랍게도 \\(h(1-h)\\) 항이 통째로 약분되어 사라진다 — 왜 그런
   일이 일어나는지 확인하라.

**정확성 확인**: 유도한 결과가 W02의 선형회귀 그래디언트와 형태가 똑같다는 것을
확인하고, 왜 손실함수의 형태(교차 엔트로피 vs MSE)가 다른데도 그래디언트는 같은
형태로 나오는지 한 문장으로 설명하라.
