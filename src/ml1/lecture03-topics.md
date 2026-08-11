# Topics Covered

## 시그모이드(Sigmoid) 함수

\\[\sigma(z) = \frac{1}{1+e^{-z}}\\]

\\(z \to +\infty\\)이면 \\(\sigma(z) \to 1\\), \\(z \to -\infty\\)이면 \\(\sigma(z)
\to 0\\), \\(z=0\\)이면 \\(\sigma(0)=0.5\\)다. 어떤 실수든 이 함수를 통과하면 0~1
사이의 값이 되므로, "확률"로 해석할 수 있다.

## 모델

\\[h_w(x) = \sigma(w^Tx) = \frac{1}{1+e^{-w^Tx}}\\]

\\(h_w(x)\\)는 "\\(x\\)가 양성 클래스(class 1)일 확률"로 해석한다. 예측은
\\(h_w(x) \ge 0.5\\)면 클래스 1, 아니면 클래스 0으로 정한다 — \\(h_w(x)=0.5\\)는
정확히 \\(w^Tx=0\\)인 지점이므로, 결정 경계(decision boundary)는 여전히
**직선**(또는 초평면)이다.

## 손실함수: 왜 MSE를 안 쓰는가

시그모이드에 평균제곱오차를 그대로 적용하면 \\(J(w)\\)가 \\(w\\)에 대해 **볼록하지
않아(non-convex)**, 경사하강법이 지역 최솟값(local minimum)에 갇힐 위험이 있다. 대신
**교차 엔트로피**(cross-entropy) 손실을 쓴다:

\\[J(w) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log h_w(x^{(i)}) + (1-y^{(i)})
\log(1-h_w(x^{(i)}))\right]\\]

직관: 정답이 \\(y=1\\)인데 모델이 \\(h_w(x) \to 0\\)으로 확신하면 \\(-\log h_w(x)
\to \infty\\) — 손실이 무한히 커진다. **확신을 갖고 틀리면 그만큼 크게 벌한다.**
신기하게도 이 손실함수를 미분하면 선형회귀와 **똑같은 형태**의 그래디언트가 나온다:

\\[\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^m \left(h_w(x^{(i)}) -
y^{(i)}\right) x_j^{(i)}\\]

그래서 경사하강법 구현 코드 자체는 W02와 거의 동일하다 — `h_w`를 계산하는 부분에
시그모이드만 추가하면 된다.

## 분류 평가 지표

| 실제\\예측 | 양성 예측 | 음성 예측 |
|---|---|---|
| 실제 양성 | True Positive (TP) | False Negative (FN) |
| 실제 음성 | False Positive (FP) | True Negative (TN) |

- **Precision** \\(= \frac{TP}{TP+FP}\\): "양성이라고 예측한 것 중 진짜 양성 비율" —
  거짓 경보를 얼마나 피했는가.
- **Recall** \\(= \frac{TP}{TP+FN}\\): "실제 양성 중 잡아낸 비율" — 놓친 양성이
  얼마나 적은가.
- **F1** \\(= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} +
  \text{Recall}}\\): Precision과 Recall의 조화평균 — 둘 다 낮으면 F1도 낮아지도록,
  한쪽만 높다고 점수를 후하게 주지 않는다.

**Precision-Recall 트레이드오프**: 임계값(threshold)을 0.5가 아니라 0.9로 올리면
Precision은 오르지만(확신 있을 때만 양성 판정) Recall은 떨어진다(애매한 양성을 놓침).
반대로 임계값을 낮추면 Recall은 오르고 Precision은 떨어진다. **PR-AUC**(Precision-Recall
곡선 아래 면적)는 이 트레이드오프 전 구간에서의 성능을 임계값 하나에 의존하지 않고
요약한 지표다 — 특히 클래스 불균형이 심할 때 accuracy보다 훨씬 신뢰할 수 있는 지표다.
