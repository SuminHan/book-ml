# Topics Covered

## 모델

선형회귀는 입력 \\(x = (x_1, \ldots, x_n)\\)에서 출력을 다음과 같이 예측한다:

\\[h_w(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n\\]

\\(w_0\\)은 절편(bias), \\(w_1, \ldots, w_n\\)은 각 특징의 가중치(weight)다. 표기를
간단히 하기 위해 \\(x_0 = 1\\)을 항상 추가해서 \\(h_w(x) = w^Tx = \sum_{j=0}^n w_j
x_j\\)로 쓴다 — bias까지 하나의 내적으로 통일하는 트릭이다.

## 비용함수(Cost Function): 평균제곱오차

\\(m\\)개의 학습 데이터 \\((x^{(i)}, y^{(i)})\\)에 대해, 모델이 얼마나 틀렸는지를
다음으로 정의한다:

\\[J(w) = \frac{1}{2m}\sum_{i=1}^m \left(h_w(x^{(i)}) - y^{(i)}\right)^2\\]

앞의 \\(\frac{1}{2}\\)는 나중에 미분할 때 제곱의 지수 2와 상쇄돼서 식이 깔끔해지도록
넣은 관례적 상수다 — 최솟값의 위치 자체는 바뀌지 않는다.

## 경사하강법(Gradient Descent)

\\(J(w)\\)를 최소화하는 \\(w\\)를 찾기 위해, 그래디언트(gradient, 가장 가파르게
증가하는 방향)의 **반대** 방향으로 조금씩 이동한다:

\\[w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j}, \qquad
\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^m \left(h_w(x^{(i)}) -
y^{(i)}\right) x_j^{(i)}\\]

\\(\alpha\\)는 **학습률(learning rate)** — 한 걸음의 크기다. 모든 \\(w_j\\)를 동시에
업데이트하는 이 과정을 손실이 충분히 작아질 때까지(또는 정해진 반복 횟수만큼) 반복한다.

```python
def gradient_descent(X, y, alpha, epochs):
    m, n = len(X), len(X[0])
    w = [0.0] * (n + 1)  # w[0]=bias
    for _ in range(epochs):
        grad = [0.0] * (n + 1)
        for i in range(m):
            pred = w[0] + sum(w[j+1] * X[i][j] for j in range(n))
            error = pred - y[i]
            grad[0] += error
            for j in range(n):
                grad[j+1] += error * X[i][j]
        for j in range(n + 1):
            w[j] -= alpha * grad[j] / m
    return w
```

## 학습률의 함정

\\(\alpha\\)가 너무 작으면 수렴이 느리다. \\(\alpha\\)가 너무 크면 최솟값 근처에서
오히려 튕겨 나가 발산할 수 있다 — 골짜기 바닥으로 가려다가 걸음이 너무 커서 반대편
벽으로 넘어가 버리는 것과 같다. 적절한 \\(\alpha\\)를 찾는 것 자체가 실무에서 중요한
튜닝 과정이다.

## 정규방정식(Normal Equation): 미분으로 한 번에 풀기

\\(J(w)\\)는 \\(w\\)에 대한 이차함수이므로, 경사하강법 없이 미분을 0으로 놓아 **닫힌
형태(closed-form)**로 최적해를 구할 수도 있다:

\\[w^* = (X^TX)^{-1}X^Ty\\]

여기서 \\(X\\)는 각 행이 \\(x^{(i)}\\)(bias 포함)인 \\(m \times (n+1)\\) 행렬이다.
이 유도는 이번 주 손유도 과제의 핵심이다.

**경사하강법 대 정규방정식**: 정규방정식은 반복이 필요 없어 정확하지만, \\((X^TX)^{-1}\\)
계산은 특징 개수 \\(n\\)의 세제곱에 비례하는 비용이 든다 — \\(n\\)이 수천~수만이 되는
실전 문제에서는 감당이 안 된다. 반면 경사하강법은 \\(n\\)이 커져도 매 반복의 비용이
선형으로만 늘어난다. 특징이 적을 땐 정규방정식, 많을 땐 경사하강법을 쓰는 이유다.
