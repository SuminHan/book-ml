# Chapter 3. 로지스틱회귀와 분류 평가 (Logistic Regression & Classification Metrics)

1838년, 벨기에 수학자 피에르프랑수아 페르휠스트(Pierre François Verhulst)는
인구가 지수적으로 무한히 증가할 수 없다는 문제를 풀고 있었다 — 자원은 한정돼
있으니, 인구는 어느 시점부터 증가 속도가 느려지고 특정 상한선에 수렴해야 한다.
그가 이 현상을 표현하기 위해 만든 S자 곡선에 **로지스틱**(logistic)이라는 이름을
붙였다. 180년 뒤, 이 곡선은 인구 증가와는 전혀 상관없어 보이는 문제 — "이
이메일이 스팸일 확률은 몇 퍼센트인가" — 를 푸는 데 그대로 쓰이고 있다.

## 3.1 회귀에서 분류로

로지스틱회귀라는 이름 때문에 헷갈리기 쉽지만, 이건 회귀가 아니라 **분류**
알고리즘이다. 선형회귀의 출력 \\(w^Tx\\)는 \\(-\infty\\)부터 \\(+\infty\\)까지
아무 값이나 될 수 있는데, "스팸일 확률"은 반드시 0과 1 사이여야 한다.
로지스틱함수(시그모이드)는 정확히 이 변환을 해준다:

\\[\sigma(z) = \frac{1}{1+e^{-z}}\\]

\\(z \to +\infty\\)이면 \\(\sigma(z) \to 1\\), \\(z \to -\infty\\)이면
\\(\sigma(z) \to 0\\), \\(z=0\\)이면 \\(\sigma(0)=0.5\\)다. 어떤 실수든 이
함수를 통과하면 0~1 사이의 값이 되므로, "확률"로 해석할 수 있다.

## 3.2 모델

\\[h_w(x) = \sigma(w^Tx) = \frac{1}{1+e^{-w^Tx}}\\]

\\(h_w(x)\\)는 "\\(x\\)가 양성 클래스(class 1)일 확률"로 해석한다. 예측은
\\(h_w(x) \ge 0.5\\)면 클래스 1, 아니면 클래스 0으로 정한다 — \\(h_w(x)=0.5\\)는
정확히 \\(w^Tx=0\\)인 지점이므로, 결정 경계(decision boundary)는 여전히
**직선**(또는 초평면)이다.

## 3.3 손실함수: 왜 MSE를 안 쓰는가

시그모이드에 평균제곱오차를 그대로 적용하면 \\(J(w)\\)가 \\(w\\)에 대해
**볼록하지 않아(non-convex)**, 경사하강법이 지역 최솟값(local minimum)에 갇힐
위험이 있다. 대신 **교차 엔트로피**(cross-entropy) 손실을 쓴다:

\\[J(w) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log h_w(x^{(i)}) +
(1-y^{(i)}) \log(1-h_w(x^{(i)}))\right]\\]

직관: 정답이 \\(y=1\\)인데 모델이 \\(h_w(x) \to 0\\)으로 확신하면
\\(-\log h_w(x) \to \infty\\) — 손실이 무한히 커진다. **확신을 갖고 틀리면
그만큼 크게 벌한다.** 신기하게도 이 손실함수를 미분하면 선형회귀와 **똑같은
형태**의 그래디언트가 나온다:

\\[\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^m \left(h_w(x^{(i)}) - y^{(i)}\right) x_j^{(i)}\\]

그래서 경사하강법 구현 코드 자체는 Chapter 2와 거의 동일하다 — \\(h_w\\)를
계산하는 부분에 시그모이드만 추가하면 된다.

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def logistic_gradient_descent(X, y, alpha, epochs):
    m, n = len(X), len(X[0])
    w = [0.0] * (n + 1)
    for _ in range(epochs):
        grad = [0.0] * (n + 1)
        for i in range(m):
            pred = sigmoid(w[0] + sum(w[j+1] * X[i][j] for j in range(n)))
            error = pred - y[i]
            grad[0] += error
            for j in range(n):
                grad[j+1] += error * X[i][j]
        for j in range(n + 1):
            w[j] -= alpha * grad[j] / m
    return w
```

## 3.4 "정확도"만으로는 안 되는 이유

암 검진 모델을 상상해보자. 전체 환자의 99%가 정상이라면, "무조건 정상"이라고만
찍는 모델도 정확도(accuracy) 99%를 자랑한다 — 하지만 이 모델은 암 환자를 단 한
명도 잡아내지 못하는, 쓸모없는 모델이다. Precision/Recall/F1은 바로 이런
상황(클래스 불균형)에서 정확도가 숨기는 진실을 드러내기 위한 지표다.

| 실제\\예측 | 양성 예측 | 음성 예측 |
|---|---|---|
| 실제 양성 | True Positive (TP) | False Negative (FN) |
| 실제 음성 | False Positive (FP) | True Negative (TN) |

- **Precision** \\(= \frac{TP}{TP+FP}\\): "양성이라고 예측한 것 중 진짜 양성
  비율" — 거짓 경보를 얼마나 피했는가.
- **Recall** \\(= \frac{TP}{TP+FN}\\): "실제 양성 중 잡아낸 비율" — 놓친 양성이
  얼마나 적은가.
- **F1** \\(= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}\\):
  Precision과 Recall의 조화평균 — 둘 다 낮으면 F1도 낮아지도록, 한쪽만
  높다고 점수를 후하게 주지 않는다.

**Precision-Recall 트레이드오프**: 임계값(threshold)을 0.5가 아니라 0.9로
올리면 Precision은 오르지만(확신 있을 때만 양성 판정) Recall은
떨어진다(애매한 양성을 놓침). 반대로 임계값을 낮추면 Recall은 오르고
Precision은 떨어진다. **PR-AUC**(Precision-Recall 곡선 아래 면적)는 이
트레이드오프 전 구간에서의 성능을 임계값 하나에 의존하지 않고 요약한 지표다 —
특히 클래스 불균형이 심할 때 accuracy보다 훨씬 신뢰할 수 있는 지표다.

---

## 연습문제

**1. (코딩)** 위 `logistic_gradient_descent`(핵심 줄은 빈칸으로 남겨져 있다고
가정)와, 다음 `precision_recall_f1`을 완성하라:

```python
def precision_recall_f1(y_true, y_pred):
    # ADD ADDITIONAL CODE HERE!!
    # TP, FP, FN을 센 뒤 precision, recall, f1 계산
    # (0으로 나누는 경우는 각 값을 0.0으로 처리)

y_true = [1, 1, 1, 0, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
print(precision_recall_f1(y_true, y_pred))  # (0.75, 0.75, 0.75)
```

**2. (손유도, Tier B — 힌트 제공)** 교차 엔트로피 손실 하나의 샘플에 대해:

\\[J^{(i)}(w) = -y^{(i)}\log h_w(x^{(i)}) - (1-y^{(i)})\log(1-h_w(x^{(i)}))\\]

를 \\(w_j\\)로 미분해서 \\(\frac{\partial J^{(i)}}{\partial w_j} =
(h_w(x^{(i)}) - y^{(i)})x_j^{(i)}\\)가 됨을 유도하라.

**힌트**(연쇄법칙을 세 단계로 나눠서 적용): (1) 먼저 \\(\frac{\partial
J^{(i)}}{\partial h}\\)를 구하라(h는 \\(h_w(x^{(i)})\\)의 줄임 표기,
\\(\frac{d}{dh}\log h = \frac{1}{h}\\)임을 이용). (2) \\(\sigma'(z) =
\sigma(z)(1-\sigma(z))\\)임을 이용해 \\(\frac{\partial h}{\partial z}\\)를
구하라(\\(z=w^Tx^{(i)}\\)). (3) \\(\frac{\partial z}{\partial w_j} =
x_j^{(i)}\\)임을 이용해 세 조각을 연쇄법칙으로 곱하면, 놀랍게도
\\(h(1-h)\\) 항이 통째로 약분되어 사라진다 — 왜 그런지 확인하라. 유도한
결과가 Chapter 2의 선형회귀 그래디언트와 형태가 똑같다는 것도 확인하라.
