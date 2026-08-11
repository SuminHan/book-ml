# Problem Set

난이도 등급: **Tier B (적정하나 힌트 제공)**

**1.** (코딩) 다음과 같은 함수 `knn_predict`를 작성하라:

- input parameter: 학습 특징 `X_train`(리스트의 리스트), 학습 라벨 `y_train`(리스트),
  예측할 점 `x_new`(리스트), 이웃 개수 `k`
- return value: 다수결로 예측한 라벨

```python
def knn_predict(X_train, y_train, x_new, k):
    # ADD ADDITIONAL CODE HERE!!
    # 1. x_new와 X_train의 각 점 사이 유클리드 거리 계산
    # 2. 거리 기준 오름차순 정렬
    # 3. 가장 가까운 k개의 라벨 중 다수결 반환

X_train = [[1,1],[1,2],[2,1],[6,6],[6,7],[7,6]]
y_train = ["A","A","A","B","B","B"]
print(knn_predict(X_train, y_train, [2,2], k=3))  # "A"
print(knn_predict(X_train, y_train, [6,5], k=3))  # "B"
```

**2.** (코딩) `knn_predict`를 확장해서, 라벨이 숫자일 때 다수결 대신 **평균**을
반환하는 `knn_predict_regression`을 작성하라 (kNN이 분류뿐 아니라 회귀에도 쓰일 수
있음을 보이는 문제).

```python
def knn_predict_regression(X_train, y_train, x_new, k):
    # ADD ADDITIONAL CODE HERE!!

X_train = [[1],[2],[3],[10],[11],[12]]
y_train = [10, 12, 11, 100, 102, 98]
print(knn_predict_regression(X_train, y_train, [2.5], k=3))  # 대략 11.0
```

---

## 손유도 과제 (실습시간, Tier B — 힌트 제공)

### 차원의 저주(Curse of Dimensionality) 수식적 설명

\\(n\\)차원 단위 정육면체 \\([0,1]^n\\)에 점들이 균등하게 분포한다고 하자. 전체 부피의
\\(p\\)만큼을 포함하는, 한 변의 길이가 \\(\ell\\)인 작은 정육면체(중심을 공유)를
생각한다.

**단계 1**: \\(\ell\\)을 \\(n\\)과 \\(p\\)로 표현하는 식 \\(\ell = p^{1/n}\\)을
유도하라. (힌트: \\(n\\)차원 정육면체의 부피는 한 변 길이의 \\(n\\)제곱이다. 작은
정육면체의 부피가 전체(부피 1)의 \\(p\\)배가 되도록 \\(\ell^n = p\\)를 풀어라.)

**단계 2**: \\(p=0.01\\)로 고정하고, \\(n=1, 10, 50, 100\\)일 때 각각 \\(\ell\\)을
직접 계산하라 (계산기 사용 가능).

**단계 3**: 단계 2의 결과를 바탕으로, "고차원에서는 데이터의 1%만 포함하려 해도 각 축의
길이 대부분을 써야 한다"는 문장이 왜 참인지 한 문단으로 설명하라.

**정확성 확인**: 이 결과가 kNN에 시사하는 바를 한 문장으로 요약하라 — 특징(차원)이
많아질수록 "가까운 \\(k\\)개"라는 개념 자체가 왜 점점 의미를 잃는가?
