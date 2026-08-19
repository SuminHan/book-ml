# Chapter 14. 표현학습: PCA, word2vec, Node2Vec, PageRank (Representation Learning)

1901년, 통계학자 칼 피어슨(Karl Pearson)은 흥미로운 질문을 던졌다: 여러 변수를
측정한 데이터가 있을 때, 정보를 최대한 보존하면서 변수의 개수를 줄일 수 있는
"가장 좋은 직선(또는 평면)"은 무엇인가? 그의 답 — 데이터가 가장 많이 퍼져 있는
방향을 찾아라 — 이 지금 **주성분분석**(Principal Component Analysis, PCA)이라
불리는 방법의 시작이다. 120년도 더 전에 나온 이 아이디어가, 지금은 수백
차원짜리 이미지 임베딩을 사람이 눈으로 볼 수 있는 2차원으로 줄이는 데도 그대로
쓰인다.

## 14.1 정답 없이 배운다는 것

지금까지 배운 모든 모델은 "정답(\\(y\\))"이 있었다 — 집값, 스팸 여부, 클래스
라벨. **비지도학습**은 그 정답이 없다. 대신 데이터 \\(x\\) 자체의 구조를
찾는다: "이 고객들은 몇 개의 그룹으로 자연스럽게 나뉘는가"(군집화), "이 1000개
특징 중 실제로 중요한 정보는 몇 차원짜리인가"(차원축소).

## 14.2 k-means 클러스터링 (복습)

\\(k\\)개의 중심점(centroid)까지의 거리로 데이터를 그룹 짓는 알고리즘 —
알고리즘 절차와 코드는 4.6절에서 이미 다뤘다(사실 k-means도 "가장 가까운
것을 찾는다"는 점에서 거리 기반 모델의 일종이다). 여기서 짚을 것은
알고리즘 자체가 아니라 **왜 이게 비지도학습인가**다: kNN이나 지금까지의
회귀·분류는 정답 라벨 \\(y\\)를 향해 맞춰가지만, k-means는 라벨 없이 데이터
\\(x\\)의 위치만 보고 "가까운 점끼리 묶는다"는 구조 하나로 그룹을
만들어낸다. \\(k\\)를 고르는 법(팔꿈치 방법)도 4.6절과 동일하다.

## 14.3 왜 차원을 줄이고 싶은가

Chapter 4에서 배운 차원의 저주를 떠올려보자 — 특징이 너무 많으면 "가까움"이라는
개념 자체가 무너진다. PCA는 원본 데이터의 분산(정보)을 최대한 보존하면서,
서로 상관관계가 높아 사실상 중복인 특징들을 몇 개의 새로운 축(주성분)으로
압축한다.

## 14.4 PCA: 분산을 최대로 보존하는 축 찾기

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

## 14.5 임베딩: word2vec과 Node2Vec

PCA가 "이미 벡터인 데이터"를 압축한다면, **임베딩**(embedding)은 원래
벡터가 아니었던 것(단어, 그래프의 노드)을 학습을 통해 저차원 벡터로
표현하는 방법이다. 목표는 PCA와 같다: **고차원(또는 벡터가 아닌) 원본을,
유용한 구조를 보존한 채 저차원 벡터로 압축한다.**

### word2vec: 단어를 벡터로

2013년 미콜로프(Tomas Mikolov) 등이 제안한 **word2vec**은 "단어의 의미는
주변에 어떤 단어가 오는지로 결정된다"(분포 가설, distributional
hypothesis — "그 단어의 친구를 보면 그 단어를 안다")는 직관을 학습으로
구현한다. **Skip-gram** 방식은 중심 단어 하나로 주변 단어들을 예측하도록
작은 신경망을 학습시키는데, 학습이 끝난 뒤 **이 신경망의 가중치 자체가
각 단어의 임베딩 벡터**가 된다 — "다음 단어 맞히기"라는 목표는 수단일
뿐, 진짜 원하는 결과물은 그 과정에서 부산물로 나오는 벡터다(Chapter
13의 "다음 토큰 예측이 문법·지식을 부산물로 남긴다"는 이야기와 같은
패턴이다).

이렇게 학습된 벡터는 놀라운 성질을 갖는다 — 단어 사이의 의미 관계가
벡터의 뺄셈/덧셈으로 나타난다:

\\[\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})\\]

"왕에서 남자다움을 빼고 여자다움을 더하면 여왕"이라는 관계를 학습 당시엔
전혀 명시적으로 가르치지 않았는데도, 벡터 공간의 기하학적 구조가 스스로
그렇게 정리된 것이다.

```python
import math, random

def train_skipgram(corpus, window=2, dim=8, epochs=50, lr=0.05, neg_k=3):
    # corpus: list of tokens (a single long sequence)
    vocab = sorted(set(corpus))
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    # center-word / context-word weight matrices
    W_in  = [[random.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(V)]
    W_out = [[random.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(V)]

    def sigmoid(z):
        return 1 / (1 + math.exp(-max(-20, min(20, z))))

    pairs = []
    for i, center in enumerate(corpus):
        for j in range(max(0, i - window), min(len(corpus), i + window + 1)):
            if j != i:
                pairs.append((idx[center], idx[corpus[j]]))

    for _ in range(epochs):
        random.shuffle(pairs)
        for c, o in pairs:
            # positive pair (c, o) + a few random negative words
            targets = [(o, 1)] + [(random.randrange(V), 0) for _ in range(neg_k)]
            for t, label in targets:
                z = sum(W_in[c][k] * W_out[t][k] for k in range(dim))
                pred = sigmoid(z)
                grad = (pred - label) * lr
                for k in range(dim):
                    g_in, g_out = W_in[c][k], W_out[t][k]
                    W_in[c][k]  -= grad * g_out
                    W_out[t][k] -= grad * g_in
    return {w: W_in[idx[w]] for w in vocab}
```

신기하게도 이 학습의 그래디언트도 `(pred - label)`이라는 똑같은 인자로
시작한다 — Chapter 2 로지스틱회귀의 `(h_w(x) - y)`와 정확히 같은 형태다.
우연이 아니다: "중심 단어 c 옆에 실제로 단어 t가 나왔는가(label=1)
아니면 무작위로 끼워넣은 가짜 단어인가(label=0)"를 맞히는 **이진
분류** 문제로 word2vec을 학습시키기 때문이다. (`neg_k`개의 무작위
단어를 "가짜 정답"으로 같이 학습시키는 것이 **네거티브
샘플링**(negative sampling)으로, 매 스텝마다 어휘 전체에 대해 softmax를
계산하는 비용을 피하는 핵심 트릭이다. 위 코드는 아이디어만 보이려고
아주 작게 줄인 구현이고, 실제 word2vec은 수백만 단어·수십억 개의
(중심, 주변) 쌍으로 학습된다.)

### Node2Vec: 같은 아이디어를 그래프로

**Node2Vec**은 word2vec의 skip-gram을 그대로 재사용하되, "문장" 대신
그래프 위의 **무작위 걷기**(random walk)로 만든 노드 시퀀스를 입력으로
쓴다 — 한 노드에서 출발해 무작위로 이웃을 따라가며 만든 경로가 곧
"문장"이고, 그 경로 위의 노드들이 "단어"다. "비슷한 이웃 구조를 가진
노드는 비슷한 벡터를 갖게 된다"는 것도, word2vec의 "비슷한 문맥에
나오는 단어는 비슷한 벡터를 갖는다"는 원리를 그래프에 그대로 물려준
결과다.

**Zachary's Karate Club**은 이 아이디어를 테스트하는 표준 예제
데이터셋이다 — 1977년 인류학자 웨인 재커리(Wayne Zachary)가 한 가라테
동호회 34명의 친분 관계를 기록했는데, 마침 이 동호회가 감독(node 0)과
관장(node 33) 두 파벌로 실제 분열됐다. 34개 노드, 78개의 친분
관계(edge)뿐인 작은 그래프지만, 그래프 임베딩 알고리즘이 "제대로
작동하는지" 확인하는 표준 벤치마크로 지금까지도 쓰인다 — 라벨(파벌
소속) 없이 순수하게 그래프 구조만 보고 임베딩했는데도 두 파벌이 벡터
공간에서 자연스럽게 갈라지면 성공이다.

```python
KARATE_EDGES = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
    (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),
    (1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),
    (2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),
    (6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),
    (15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),
    (25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),
    (30,32),(30,33),(31,32),(31,33),(32,33),
]

def random_walk(neighbors, start, length):
    walk = [start]
    for _ in range(length - 1):
        cur = walk[-1]
        if not neighbors[cur]:
            break
        walk.append(random.choice(neighbors[cur]))
    return walk

def build_walks(edges, n_nodes, walks_per_node=10, walk_length=8):
    neighbors = {i: [] for i in range(n_nodes)}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    walks = []
    for node in range(n_nodes):
        for _ in range(walks_per_node):
            walks.append([str(n) for n in random_walk(neighbors, node, walk_length)])
    return walks
```

무작위 걷기로 만든 `walks`(각 걷기가 "문장", 각 노드 번호가 "단어")를
이어붙여(`corpus = sum(walks, [])`) 위 `train_skipgram`에 그대로 넣으면
34개 노드 각각의 임베딩 벡터가 나온다 — 이 벡터를 14.4의 PCA로 2차원까지
압축해서 그려보면, 실제로 두 파벌이 공간적으로 갈라지는 것을 볼 수
있다. **word2vec과 Node2Vec은 서로 다른 데이터(텍스트 vs. 그래프)에
적용됐을 뿐, "함께 자주 나타나는 것들은 비슷한 벡터를 갖도록 학습한다"는
같은 아이디어의 두 얼굴이다.**

## 14.6 같은 무작위 걷기, 다른 질문: PageRank

Node2Vec은 무작위 걷기가 만들어내는 "노드의 순서(시퀀스)"를 이용했다.
같은 무작위 걷기 자체를 완전히 다른 질문에 쓸 수도 있다: **이 그래프
위를 무작위로 영원히 돌아다닌다면, 각 노드에 머물러 있을 확률은 각각
얼마나 될까?** 이 질문에 대한 답이 곧 그 노드의 "중요도"라는 것이,
1998년 래리 페이지(Larry Page)와 세르게이 브린(Sergey Brin)이 구글을
세울 때 쓴 **PageRank** 알고리즘의 핵심 아이디어다 — 많은 페이지가
링크하는 페이지일수록, 또 중요한 페이지가 링크하는 페이지일수록,
무작위로 링크를 따라다니는 서퍼(random surfer)가 그 페이지에 더 자주
머문다.

노드 \\(i\\)의 페이지랭크 \\(PR(i)\\)는 다음을 만족해야 한다 — 자신을
가리키는 모든 노드 \\(j\\)로부터, 그 노드가 가진 점수를 그 노드가 가진
바깥 링크 수(\\(\text{outdeg}(j)\\))만큼 나눠 받은 합:

\\[PR(i) = \frac{1-d}{N} + d\sum_{j \to i} \frac{PR(j)}{\text{outdeg}(j)}\\]

\\(N\\)은 전체 노드 수, \\(d\\)(보통 0.85)는 **댐핑 팩터**(damping
factor) — 확률 \\(d\\)로 링크를 따라가고, 확률 \\(1-d\\)로 아무 노드로나
순간이동(teleport)한다는 뜻이다(막다른 골목에 갇히거나 순환에 갇히는
것을 막기 위한 장치).

이 식은 우변에 \\(PR\\) 자신이 다시 등장하는 **재귀식**이라 한 번에 풀 수
없다 — 그래서 아무 값(모든 노드에 \\(1/N\\))에서 시작해, 우변에 반복해서
대입하며 값이 더 이상 바뀌지 않을 때까지 되풀이하는
**거듭제곱법**(power iteration)으로 푼다:

```python
def pagerank(edges, n_nodes, d=0.85, iters=100):
    outgoing = {i: [] for i in range(n_nodes)}  # directed: a -> b
    for a, b in edges:
        outgoing[a].append(b)
    outdeg = {i: max(1, len(outgoing[i])) for i in range(n_nodes)}
    incoming = {i: [] for i in range(n_nodes)}  # who points to me?
    for a, b in edges:
        incoming[b].append(a)

    pr = {i: 1 / n_nodes for i in range(n_nodes)}
    for _ in range(iters):
        pr = {i: (1 - d) / n_nodes + d * sum(pr[j] / outdeg[j] for j in incoming[i])
              for i in range(n_nodes)}
    return pr

# Karate Club은 무방향 그래프이므로, 친분 관계 하나를 양방향 링크로 취급한다
directed_edges = KARATE_EDGES + [(b, a) for a, b in KARATE_EDGES]
scores = pagerank(directed_edges, 34)
print(sorted(scores.items(), key=lambda kv: -kv[1])[:3])
# node 33, node 0이 가장 높다 -- 실제 두 "허브" 노드(관장, 감독)와 정확히 일치
```

이 거듭제곱법이 왜 수렴하는지는, **ML2 Chapter 4**에서 벨만 최적방정식의
해가 유일하게 존재함을 보일 때 쓸 **바나흐 고정점 정리**(Banach fixed
point theorem)와 정확히 같은 논리다 — "어떤 변환을 값이 안 바뀔 때까지
반복해서 적용하면 결국 고정점(fixed point)에 도달한다"는 같은 수학이,
여기서는 "그래프 위 확률분포"에, 강화학습에서는 "상태의 가치함수"에
적용된다.

**PCA·word2vec·Node2Vec·PageRank는 얼핏 서로 다른 문제처럼 보이지만,
전부 "무언가를 어떤 변환에 반복해서 통과시키면 특별한 지점(고유벡터,
임베딩, 정상분포)에 수렴한다"는 하나의 수학적 패턴을 공유한다.**

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

**2. (개념 서술)** word2vec은 skip-gram(중심 단어로 주변 단어 예측)을
쓴다. 만약 반대로 "주변 단어들로 중심 단어를 예측"하는 방식(CBOW,
Continuous Bag-of-Words)을 쓴다면 학습 목표가 어떻게 달라질지, 그리고
왜 두 방식 모두 결국 비슷한 임베딩 벡터를 만들어낼 것으로 예상되는지
두세 문장으로 설명하라.

**3. (손유도, Tier B — 힌트 제공)** 중심화된 데이터 \\(X\\)를 단위벡터
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
