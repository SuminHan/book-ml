# Problem Set

난이도 등급: **Tier A (자유 유도)**

**1.** (코딩) 다음과 같은 함수 `gradient_descent`를 작성하라:

- input parameter: 특징행렬 `X`(리스트의 리스트, 각 행이 샘플), 타겟 `y`(리스트),
  학습률 `alpha`, 반복횟수 `epochs`
- return value: 학습된 가중치 벡터 `w` (bias 포함, 길이 = `len(X[0]) + 1`)
- 비용함수는 평균제곱오차(MSE): \\(J(w) = \frac{1}{2m}\sum_{i=1}^m (h_w(x^{(i)}) -
  y^{(i)})^2\\)

```python
def gradient_descent(X, y, alpha, epochs):
    # ADD ADDITIONAL CODE HERE!!
    # 초기화: w를 0벡터로, bias 포함하여 길이 len(X[0])+1

    for epoch in range(epochs):
        # ADD ADDITIONAL CODE HERE!!
        # 예측값 계산 h_w(x) = w^T x, 그래디언트 계산, w 갱신

X = [[1.0], [2.0], [3.0], [4.0]]
y = [3.0, 5.0, 7.0, 9.0]
print(gradient_descent(X, y, alpha=0.01, epochs=1000))  # 대략 [1.0, 2.0] (bias=1, weight=2)
```

**2.** 학습률 `alpha`를 너무 크게 주면 어떤 현상이 발생하는가? 그 이유를 경사하강법의
업데이트 공식 \\(w \leftarrow w - \alpha \nabla J(w)\\)을 이용해 한 문단으로 설명하라.

---

## 손유도 과제 (실습시간, Tier A — 자유 유도)

### 정규방정식(Normal Equation) 유도

선형회귀의 비용함수 \\(J(w) = \frac{1}{2m}\|Xw - y\|^2\\)을 \\(w\\)에 대해 미분하여
0으로 놓고,

\\[w^* = (X^TX)^{-1}X^Ty\\]

가 되는 과정을 **처음부터 끝까지 손으로 유도**하라. (힌트: \\(\nabla_w \|Xw-y\|^2 =
2X^T(Xw-y)\\)임을 먼저 보인 뒤, 이를 0으로 놓고 \\(w\\)에 대해 정리한다.)

**정확성 증명 관점**: 유도한 \\(w^*\\)가 실제로 \\(J(w)\\)의 **전역 최솟값**임을
(Hessian이 양의 준정부호 행렬임을 보이는 방식으로) 논증하라.
