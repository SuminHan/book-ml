# Topics Covered

## 알고리즘

새로운 점 \\(x\\)를 분류하려면:

1. 학습 데이터의 모든 점 \\(x^{(i)}\\)에 대해 \\(x\\)와의 거리를 계산한다.
2. 거리가 가장 가까운 \\(k\\)개를 고른다.
3. 그 \\(k\\)개의 라벨 중 다수결(분류) 또는 평균(회귀)을 예측값으로 낸다.

```python
def knn_predict(X_train, y_train, x_new, k):
    distances = []
    for i in range(len(X_train)):
        d = sum((X_train[i][j] - x_new[j]) ** 2 for j in range(len(x_new))) ** 0.5
        distances.append((d, y_train[i]))
    distances.sort(key=lambda p: p[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return max(set(k_nearest_labels), key=k_nearest_labels.count)  # 다수결
```

## 거리 함수

가장 흔한 선택은 **유클리드 거리(Euclidean distance)**:

\\[d(x, x') = \sqrt{\sum_{j=1}^n (x_j - x'_j)^2}\\]

다른 선택지로 맨해튼 거리(\\(\sum_j |x_j - x'_j|\\), 좌표축을 따라서만 이동)도 있다.
거리를 재기 전에 **특징 정규화(normalization)**가 거의 필수다 — "방 개수"(0~10 범위)와
"집값"(수억 원 단위)을 그대로 섞어 거리를 재면, 스케일이 큰 특징이 거리를 사실상
독차지해버린다.

## \\(k\\)를 고르는 트레이드오프

- \\(k\\)가 너무 작으면(예: \\(k=1\\)): 노이즈 하나에도 민감하게 반응 — 과적합
  (overfitting).
- \\(k\\)가 너무 크면(예: \\(k=m\\), 전체 데이터): 항상 전체 다수결만 예측 — 지역적
  패턴을 완전히 무시(underfitting).

적절한 \\(k\\)는 보통 검증 데이터(validation set)로 여러 값을 시도해보고 정한다.

## 차원의 저주(Curse of Dimensionality)

직관: 특징이 늘어날수록(고차원일수록), "가장 가까운 이웃"조차 점점 멀어진다.

수식으로 감을 잡아보자. \\(n\\)차원 단위 정육면체 \\([0,1]^n\\) 안에 점들이 균등하게
뿌려져 있다고 하자. 전체 부피의 비율 \\(p\\)만큼을 "가까운 이웃"으로 포함하는 작은
정육면체를 만들려면, 그 한 변의 길이 \\(\ell\\)은:

\\[\ell^n = p \quad\Longrightarrow\quad \ell = p^{1/n}\\]

\\(p=0.01\\)(전체의 1%만 포함하고 싶다)일 때:

| \\(n\\) | \\(\ell = p^{1/n}\\) |
|---|---|
| 1 | 0.01 |
| 10 | 0.63 |
| 100 | 0.955 |

\\(n=100\\)에서는 각 변의 **95.5%**를 차지해야 겨우 전체의 1%만 포함된다 — "가까운
이웃"이라 부를 만한 좁은 영역이 사실상 사라진다. 특징이 많아질수록 모든 점이 서로
비슷하게 멀어 보이는 이유다. 이번 주 손유도 과제에서 이 식을 직접 유도하고, 왜 kNN이
고차원 데이터에서 잘 작동하지 않는지를 수식으로 설명한다.
