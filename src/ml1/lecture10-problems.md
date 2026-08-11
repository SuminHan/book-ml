# Problem Set

난이도 등급: **Tier B (적정하나 힌트 제공)**

**1.** (코딩) 다음과 같은 함수 `kmeans_assign`을 작성하라 (k-means의 할당 단계만):

- input parameter: 데이터 `X`(리스트의 리스트), 중심점 `centroids`(리스트의 리스트)
- return value: 각 데이터가 배정된 중심점의 인덱스 리스트

```python
def kmeans_assign(X, centroids):
    # ADD ADDITIONAL CODE HERE!!

X = [[1,1],[1,2],[8,8],[9,9]]
centroids = [[1,1],[9,9]]
print(kmeans_assign(X, centroids))  # [0, 0, 1, 1]
```

**2.** (코딩) 다음과 같은 함수 `center_data`(PCA의 전처리 단계)를 작성하라:

- input parameter: 데이터 `X`(리스트의 리스트, 각 행이 샘플)
- return value: 각 열의 평균이 0이 되도록 이동시킨 `X`

```python
def center_data(X):
    # ADD ADDITIONAL CODE HERE!!

X = [[1,2],[3,4],[5,6]]
print(center_data(X))  # [[-2,-2],[0,0],[2,2]]
```

---

## 손유도 과제 (실습시간, Tier B — 힌트 제공)

### PCA의 고유값분해 유도(공분산행렬 → 고유벡터)

중심화된 데이터 \\(X\\)(각 행이 샘플)를 단위벡터 \\(v\\)(\\(\|v\|=1\\)) 방향으로
투영한 결과의 분산은 \\(v^T \Sigma v\\)이다(단, \\(\Sigma = \frac{1}{m}X^TX\\)).
이 분산을 최대화하는 \\(v\\)를 구하고 싶다.

**단계 1**: \\(\|v\|=1\\) 제약 하에 \\(v^T\Sigma v\\)를 최대화하는 문제를, 라그랑주
승수 \\(\lambda\\)를 이용해 다음 목적함수로 바꿔라:

\\[\mathcal{L}(v, \lambda) = v^T\Sigma v - \lambda(v^Tv - 1)\\]

**단계 2**: \\(\mathcal{L}\\)을 \\(v\\)로 미분해서 0으로 놓아라. (힌트:
\\(\frac{\partial}{\partial v}(v^T\Sigma v) = 2\Sigma v\\), \\(\frac{\partial}{\partial
v}(v^Tv) = 2v\\)임을 이용하면, \\(\Sigma v = \lambda v\\)라는 익숙한 형태가 나온다 —
이게 바로 고유값 방정식이다.)

**단계 3**: 단계 2의 결과 \\(\Sigma v = \lambda v\\)를 원래 목적함수 \\(v^T\Sigma v\\)에
대입해서, 분산이 정확히 \\(\lambda\\)(고유값)가 됨을 보여라. (힌트:
\\(v^T\Sigma v = v^T(\lambda v) = \lambda(v^Tv) = \lambda \cdot 1\\).)

**정확성 확인**: 단계 3의 결과가 왜 "고유값이 가장 큰 고유벡터를 첫 번째 주성분으로
고른다"는 알고리즘의 근거가 되는지 한 문장으로 설명하라.
