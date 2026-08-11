# Topics Covered

## 결정 트리의 구조

각 내부 노드는 하나의 질문(예: "\\(x_2 > 5\\)?")이고, 각 리프(leaf) 노드는 예측값이다.
예측할 때는 루트에서 시작해 질문에 답하며 리프까지 내려간다.

## 지니불순도(Gini Impurity)

노드 안의 데이터가 얼마나 "섞여" 있는지를 재는 지표. 클래스가 \\(K\\)개이고, 클래스
\\(k\\)의 비율이 \\(p_k\\)일 때:

\\[G = 1 - \sum_{k=1}^K p_k^2\\]

- 노드 안이 한 클래스로만 순수하면(\\(p_k=1\\), 나머지 0) \\(G=0\\) — 가장 좋은 상태.
- 클래스가 반반이면(\\(K=2\\), \\(p_1=p_2=0.5\\)) \\(G = 1 - 0.25 - 0.25 = 0.5\\) —
  가장 나쁜 상태(이진 분류에서 최댓값).

## 정보이득(Information Gain)

먼저 **엔트로피**(entropy)를 정의한다:

\\[H = -\sum_{k=1}^K p_k \log_2 p_k\\]

지니불순도와 마찬가지로 순수할수록 작다(0), 섞여 있을수록 크다. 어떤 질문으로 노드를
왼쪽(\\(L\\))과 오른쪽(\\(R\\))으로 나눴을 때, **정보이득**은 "나누기 전 엔트로피 —
나눈 뒤 엔트로피(가중평균)"이다:

\\[\text{IG} = H(\text{parent}) - \left(\frac{|L|}{|L|+|R|}H(L) +
\frac{|R|}{|L|+|R|}H(R)\right)\\]

정보이득이 클수록 그 질문이 데이터를 잘 나눴다는 뜻이다. 결정 트리를 만들 때는 각
단계에서 **정보이득(또는 지니불순도 감소량)이 가장 큰 질문**을 고른다 — 이게
스무고개에서 "가장 잘 좁히는 질문을 먼저 물어라"에 해당하는 알고리즘이다.

```python
import math

def gini(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return 1 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def information_gain(parent_labels, left_labels, right_labels):
    n = len(parent_labels)
    weighted_child = (len(left_labels) / n) * entropy(left_labels) + \
                      (len(right_labels) / n) * entropy(right_labels)
    return entropy(parent_labels) - weighted_child
```

## 언제 나누기를 멈추는가

트리를 끝까지 키우면 모든 리프가 완벽히 순수해질 때까지 나눈다 — 학습 데이터는 100%
맞히지만, 새 데이터에는 취약한(과적합) 트리가 된다. 실무에서는 최대 깊이(max depth),
리프의 최소 샘플 수 같은 **정지 조건**(stopping criteria)으로 트리 크기를 제한하거나,
트리를 다 키운 뒤 불필요한 가지를 잘라내는 **가지치기**(pruning)를 한다.

## 랜덤 포레스트: 배깅(Bagging) + 특징 무작위화

랜덤 포레스트는 두 가지 무작위성을 더한 트리 여러 개를 학습한다:

1. **배깅**(Bootstrap Aggregating): 각 트리는 원본 데이터에서 복원추출(중복 허용)로
   같은 크기의 데이터셋을 뽑아 학습한다 — 트리마다 조금씩 다른 데이터를 본다.
2. **특징 무작위화**: 각 분기(split)에서 전체 특징이 아니라, 무작위로 고른 일부
   특징 중에서만 최선의 질문을 찾는다.

이 두 무작위성이 트리들을 서로 덜 닮게 만들고, 그래야 다수결(또는 평균)을 냈을 때
개별 트리의 과적합 성향이 서로 상쇄된다 — "완전히 같은 실수를 하는 전문가 100명"보다
"서로 다른 실수를 하는 전문가 100명"의 평균이 훨씬 안정적인 것과 같은 원리다.
