# Topics Covered

## k-means 클러스터링

\\(k\\)개의 그룹으로 데이터를 나누는 알고리즘. 각 그룹은 **중심점(centroid)**으로
대표된다.

1. \\(k\\)개의 중심점을 무작위로 초기화한다.
2. **할당 단계**: 각 데이터 점을 가장 가까운 중심점의 그룹으로 배정한다.
3. **갱신 단계**: 각 그룹의 새 중심점을, 그 그룹에 속한 점들의 평균으로 다시 계산한다.
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
variance)을 그려보고, 감소폭이 급격히 줄어드는 지점("팔꿈치", elbow)을 고르는 방법이
흔히 쓰인다.

## PCA: 분산을 최대로 보존하는 축 찾기

데이터를 저차원으로 투영(project)하되, **투영된 데이터의 분산이 최대**가 되는 방향을
찾는 것이 목표다 — 정보를 가장 적게 잃는 방향이라는 뜻이다.

**절차**:

1. 데이터를 평균이 0이 되도록 중심화(centering)한다.
2. 공분산 행렬(covariance matrix) \\(\Sigma = \frac{1}{m}X^TX\\)를 계산한다.
3. \\(\Sigma\\)의 고유벡터(eigenvector)와 고유값(eigenvalue)을 구한다.
4. 고유값이 큰 순서대로 고유벡터를 정렬한다 — 각 고유벡터가 "주성분(principal
   component)"이고, 대응하는 고유값이 그 방향의 분산 크기다.
5. 상위 \\(d\\)개의 고유벡터에 데이터를 투영하면, \\(d\\)차원으로 압축된 표현을
   얻는다.

**왜 고유벡터인가(직관)**: 공분산 행렬 \\(\Sigma\\)에 대해 \\(\Sigma v = \lambda v\\)를
만족하는 \\(v\\)(고유벡터)는 "그 방향으로 데이터를 투영했을 때, 그 결과의 분산이
정확히 \\(\lambda\\)(고유값)가 되는" 특별한 방향이다. 분산을 최대화하는 문제를
라그랑주 승수법으로 풀면, 정확히 이 고유값 문제로 귀결된다는 것이 이번 주 손유도
과제의 핵심이다.

## 임베딩(Embedding) 맛보기: Node2Vec

PCA가 "이미 벡터인 데이터"를 압축한다면, **임베딩**은 원래 벡터가 아니었던 것(단어,
그래프의 노드)을 학습을 통해 저차원 벡터로 표현하는 방법이다. Node2Vec은 그래프에서
"비슷한 이웃 구조를 가진 노드는 비슷한 벡터를 갖도록" 학습한다 — 소셜 네트워크에서
비슷한 친구 그룹에 속한 두 사람의 벡터가 가까워지는 식이다. 원리는 다르지만 목표는
PCA와 같다: **고차원(또는 벡터가 아닌) 원본을, 유용한 구조를 보존한 채 저차원 벡터로
압축한다.**
