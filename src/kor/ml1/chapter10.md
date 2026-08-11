# Chapter 10. 비지도학습과 표현학습 (Unsupervised Learning & Representation Learning)

1901년, 통계학자 칼 피어슨(Karl Pearson)은 흥미로운 질문을 던졌다: 여러 변수를
측정한 데이터가 있을 때, 정보를 최대한 보존하면서 변수의 개수를 줄일 수 있는
"가장 좋은 직선(또는 평면)"은 무엇인가? 그의 답 — 데이터가 가장 많이 퍼져 있는
방향을 찾아라 — 이 지금 **주성분분석**(Principal Component Analysis, PCA)이라
불리는 방법의 시작이다. 120년도 더 전에 나온 이 아이디어가, 지금은 수백
차원짜리 이미지 임베딩을 사람이 눈으로 볼 수 있는 2차원으로 줄이는 데도 그대로
쓰인다.

## 10.1 정답 없이 배운다는 것

지금까지 배운 모든 모델은 "정답(\\(y\\))"이 있었다 — 집값, 스팸 여부, 클래스
라벨. **비지도학습**은 그 정답이 없다. 대신 데이터 \\(x\\) 자체의 구조를
찾는다: "이 고객들은 몇 개의 그룹으로 자연스럽게 나뉘는가"(군집화), "이 1000개
특징 중 실제로 중요한 정보는 몇 차원짜리인가"(차원축소).

## 10.2 k-means 클러스터링

\\(k\\)개의 그룹으로 데이터를 나누는 알고리즘. 각 그룹은 **중심점**(centroid)으로
대표된다.

1. \\(k\\)개의 중심점을 무작위로 초기화한다.
2. **할당 단계**: 각 데이터 점을 가장 가까운 중심점의 그룹으로 배정한다.
3. **갱신 단계**: 각 그룹의 새 중심점을, 그 그룹에 속한 점들의 평균으로 다시
   계산한다.
4. 할당이 더 이상 바뀌지 않을 때까지 2~3을 반복한다.

```python
def kmeans(X, k, max_iters=100):
    import random
    centroids = random.sample(X, k)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [sum((x[j]-c[j])**2 for j in range(len(x))) for c in centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(x)
        new_centroids = [
            [sum(pt[j] for pt in cluster) / len(cluster) for j in range(len(X[0]))]
            if cluster else centroids[i]
            for i, cluster in enumerate(clusters)
        ]
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters
```

**\\(k\\)를 고르는 법**: 여러 \\(k\\)에 대해 클러스터 내 분산(within-cluster
variance)을 그려보고, 감소폭이 급격히 줄어드는 지점("팔꿈치", elbow)을
고르는 방법이 흔히 쓰인다.

## 10.3 왜 차원을 줄이고 싶은가

Chapter 4에서 배운 차원의 저주를 떠올려보자 — 특징이 너무 많으면 "가까움"이라는
개념 자체가 무너진다. PCA는 원본 데이터의 분산(정보)을 최대한 보존하면서,
서로 상관관계가 높아 사실상 중복인 특징들을 몇 개의 새로운 축(주성분)으로
압축한다.

## 10.4 PCA: 분산을 최대로 보존하는 축 찾기

데이터를 저차원으로 투영(project)하되, **투영된 데이터의 분산이 최대**가
되는 방향을 찾는 것이 목표다 — 정보를 가장 적게 잃는 방향이라는 뜻이다.

**절차**:

1. 데이터를 평균이 0이 되도록 중심화(centering)한다.
2. 공분산 행렬(covariance matrix) \\(\Sigma = \frac{1}{m}X^TX\\)를 계산한다.
3. \\(\Sigma\\)의 고유벡터(eigenvector)와 고유값(eigenvalue)을 구한다.
4. 고유값이 큰 순서대로 고유벡터를 정렬한다 — 각 고유벡터가 "주성분(principal
   component)"이고, 대응하는 고유값이 그 방향의 분산 크기다.
5. 상위 \\(d\\)개의 고유벡터에 데이터를 투영하면, \\(d\\)차원으로 압축된
   표현을 얻는다.

**왜 고유벡터인가**(직관): 공분산 행렬 \\(\Sigma\\)에 대해 \\(\Sigma v =
\lambda v\\)를 만족하는 \\(v\\)(고유벡터)는 "그 방향으로 데이터를
투영했을 때, 그 결과의 분산이 정확히 \\(\lambda\\)(고유값)가 되는" 특별한
방향이다. 분산을 최대화하는 문제를 라그랑주 승수법으로 풀면, 정확히 이
고유값 문제로 귀결된다는 것이 이번 장 연습문제의 핵심이다.

## 10.5 임베딩 맛보기: Node2Vec

PCA가 "이미 벡터인 데이터"를 압축한다면, **임베딩**은 원래 벡터가 아니었던
것(단어, 그래프의 노드)을 학습을 통해 저차원 벡터로 표현하는 방법이다.
Node2Vec은 그래프에서 "비슷한 이웃 구조를 가진 노드는 비슷한 벡터를 갖도록"
학습한다 — 소셜 네트워크에서 비슷한 친구 그룹에 속한 두 사람의 벡터가
가까워지는 식이다. 원리는 다르지만 목표는 PCA와 같다: **고차원(또는
벡터가 아닌) 원본을, 유용한 구조를 보존한 채 저차원 벡터로 압축한다.**

**지도학습이 "정답을 맞히는 법"을 배운다면, 비지도학습은 "데이터가 스스로
어떤 모양을 하고 있는지"를 배운다 — 둘은 서로 다른 질문에 답한다.**

---

## 연습문제

**1. (코딩)** 다음과 같은 함수 `kmeans_assign`(k-means의 할당 단계)과
`center_data`(PCA의 전처리 단계)를 완성하라(핵심 줄은 빈칸으로 남겨져 있다고
가정):

```python
def kmeans_assign(X, centroids):
    # ADD ADDITIONAL CODE HERE!!

X = [[1,1],[1,2],[8,8],[9,9]]
centroids = [[1,1],[9,9]]
print(kmeans_assign(X, centroids))  # [0, 0, 1, 1]

def center_data(X):
    # ADD ADDITIONAL CODE HERE!!

X2 = [[1,2],[3,4],[5,6]]
print(center_data(X2))  # [[-2,-2],[0,0],[2,2]]
```

**2. (손유도, Tier B — 힌트 제공)** 중심화된 데이터 \\(X\\)를 단위벡터
\\(v\\)(\\(\|v\|=1\\)) 방향으로 투영한 결과의 분산은 \\(v^T \Sigma v\\)이다
(단, \\(\Sigma = \frac{1}{m}X^TX\\)). 이 분산을 최대화하는 \\(v\\)를 구하고
싶다.

**힌트**: \\(\|v\|=1\\) 제약 하에 \\(v^T\Sigma v\\)를 최대화하는 문제를,
라그랑주 승수 \\(\lambda\\)를 이용해 \\(\mathcal{L}(v, \lambda) =
v^T\Sigma v - \lambda(v^Tv - 1)\\)로 바꾼 뒤, \\(v\\)로 미분해서 0으로
놓으면(\\(\frac{\partial}{\partial v}(v^T\Sigma v) = 2\Sigma v\\),
\\(\frac{\partial}{\partial v}(v^Tv) = 2v\\)임을 이용) \\(\Sigma v =
\lambda v\\)라는 고유값 방정식이 나온다. 이 결과를 원래 목적함수에
대입하면(\\(v^T\Sigma v = v^T(\lambda v) = \lambda\\)) 분산이 정확히
\\(\lambda\\)(고유값)가 됨을 보일 수 있다.

**정확성 확인**: 위 결과가 왜 "고유값이 가장 큰 고유벡터를 첫 번째
주성분으로 고른다"는 알고리즘의 근거가 되는지 한 문장으로 설명하라.
