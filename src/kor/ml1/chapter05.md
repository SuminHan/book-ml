# Chapter 5. 트리 기반 모델 (Tree-Based Models)

"동물인가요? 네 다리로 걷나요? 야옹 소리를 내나요?" — 스무고개(20 Questions)
게임은 예/아니오 질문을 잘 골라서 물어보면, 스무 번 안에 어떤 답이든 좁혀낼 수
있다는 게임이다. **결정 트리**(Decision Tree)는 이 게임을 알고리즘으로 그대로
옮긴 것이다: 데이터를 가장 잘 나누는 질문("나이가 30보다 큰가?")을 하나씩 골라
가며, 각 질문이 답을 좁혀갈 때까지 반복한다.

## 5.1 결정 트리의 구조

각 내부 노드는 하나의 질문(예: "\\(x_2 > 5\\)?")이고, 각 리프(leaf) 노드는
예측값이다. 예측할 때는 루트에서 시작해 질문에 답하며 리프까지 내려간다.

## 5.2 "가장 잘 나누는 질문"이란: 지니불순도

스무고개를 잘하는 사람은 아무 질문이나 던지지 않는다 — "생물인가요?"처럼 답이
반반으로 갈릴 만한 질문을 먼저 던져야 정보를 가장 많이 얻는다. **지니불순도**
(Gini Impurity)는 노드 안의 데이터가 얼마나 "섞여" 있는지를 재는 지표다.
클래스가 \\(K\\)개이고, 클래스 \\(k\\)의 비율이 \\(p_k\\)일 때:

\\[G = 1 - \sum_{k=1}^K p_k^2\\]

노드 안이 한 클래스로만 순수하면(\\(p_k=1\\), 나머지 0) \\(G=0\\) — 가장 좋은
상태다. 클래스가 반반이면(\\(K=2\\), \\(p_1=p_2=0.5\\)) \\(G = 1 - 0.25 -
0.25 = 0.5\\) — 이진 분류에서 가장 나쁜 상태(최댓값)다.

## 5.3 정보이득 (Information Gain)

먼저 **엔트로피**(entropy)를 정의한다:

\\[H = -\sum_{k=1}^K p_k \log_2 p_k\\]

지니불순도와 마찬가지로 순수할수록 작다(0), 섞여 있을수록 크다. 어떤 질문으로
노드를 왼쪽(\\(L\\))과 오른쪽(\\(R\\))으로 나눴을 때, **정보이득**은 "나누기
전 엔트로피 — 나눈 뒤 엔트로피(가중평균)"이다:

\\[\text{IG} = H(\text{parent}) - \left(\frac{|L|}{|L|+|R|}H(L) +
\frac{|R|}{|L|+|R|}H(R)\right)\\]

정보이득이 클수록 그 질문이 데이터를 잘 나눴다는 뜻이다. 결정 트리를 만들 때는
각 단계에서 **정보이득(또는 지니불순도 감소량)이 가장 큰 질문**을 고른다 —
이게 스무고개에서 "가장 잘 좁히는 질문을 먼저 물어라"에 해당하는 알고리즘이다.

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

## 5.4 언제 나누기를 멈추는가

트리를 끝까지 키우면 모든 리프가 완벽히 순수해질 때까지 나눈다 — 학습 데이터는
100% 맞히지만, 새 데이터에는 취약한(과적합) 트리가 된다. 실무에서는 최대
깊이(max depth), 리프의 최소 샘플 수 같은 **정지 조건**(stopping criteria)으로
트리 크기를 제한하거나, 트리를 다 키운 뒤 불필요한 가지를 잘라내는
**가지치기**(pruning)를 한다.

## 5.5 트리 하나에서 숲으로: 랜덤 포레스트

트리 하나는 데이터를 완벽하게 외워버리기(과적합) 쉽다. **랜덤 포레스트**(Random
Forest)의 아이디어는 단순하다: 서로 조금씩 다른 트리를 여러 개 만들고, 그
트리들의 다수결로 예측한다. 두 가지 무작위성을 더한다:

1. **배깅**(Bootstrap Aggregating): 각 트리는 원본 데이터에서 복원추출(중복
   허용)로 같은 크기의 데이터셋을 뽑아 학습한다 — 트리마다 조금씩 다른 데이터를
   본다.
2. **특징 무작위화**: 각 분기(split)에서 전체 특징이 아니라, 무작위로 고른
   일부 특징 중에서만 최선의 질문을 찾는다.

이 두 무작위성이 트리들을 서로 덜 닮게 만들고, 그래야 다수결(또는 평균)을 냈을
때 개별 트리의 과적합 성향이 서로 상쇄된다 — "완전히 같은 실수를 하는 전문가
100명"보다 "서로 다른 실수를 하는 전문가 100명"의 평균이 훨씬 안정적인 것과
같은 원리다. 개별 트리는 각자 편향된 실수를 하지만, 그 실수들이 서로 다른
방향이면 평균을 내는 순간 상쇄된다 — "여러 전문가에게 따로 물어보고 다수결을
따른다"는 앙상블(ensemble)의 직관이다.

**결정 트리는 인간이 읽을 수 있는 규칙(if-then)의 나무이면서 동시에, "얼마나
잘 나누는가"를 정확히 수식으로 잴 수 있는 알고리즘이다 — 이 두 가지 성질이
실무에서 여전히 널리 쓰이는 이유다.**

---

## 연습문제

**1. (코딩)** 다음과 같은 함수 `gini`와 `best_split`을 완성하라(핵심 줄은
빈칸으로 남겨져 있다고 가정):

```python
def gini(labels):
    # ADD ADDITIONAL CODE HERE!!

def weighted_gini(left_labels, right_labels):
    # ADD ADDITIONAL CODE HERE!!
    # 왼쪽/오른쪽 노드 크기로 가중평균한 지니불순도

def best_split(X, y, feature_idx):
    best_threshold, best_gain = None, -1
    parent_gini = gini(y)
    for threshold in sorted(set(row[feature_idx] for row in X)):
        left_y = [y[i] for i in range(len(X)) if X[i][feature_idx] <= threshold]
        right_y = [y[i] for i in range(len(X)) if X[i][feature_idx] > threshold]
        if not left_y or not right_y:
            continue
        gain = parent_gini - weighted_gini(left_y, right_y)
        if gain > best_gain:
            best_threshold, best_gain = threshold, gain
    return best_threshold, best_gain

X = [[2.0],[3.0],[4.0],[7.0],[8.0],[9.0]]
y = ["A","A","A","B","B","B"]
print(best_split(X, y, 0))  # (4.0, 1.0) -- 완벽하게 나뉨
```

**2. (손유도, Tier A — 자유 유도)** 다음 8개 샘플이 있다: 클래스는
`[A,A,A,A,B,B,B,B]`이고, 특징 \\(x\\) 값은 `[1,2,3,4,5,6,7,8]`이다.

전체(8개)의 지니불순도를 손으로 계산한 뒤, \\(x \le 4\\)와 \\(x > 4\\)로
나누는 분기(threshold=4)의 정보이득(지니불순도 감소량)을 계산하라. threshold=2로
나눴을 때(왼쪽: `[A,A]`, 오른쪽: `[A,A,B,B,B,B]`)의 정보이득과 비교해 어느 쪽
분기가 데이터를 더 잘 나누는지 논하고, 문제 1의 `best_split(X, y, 0)` 결과와
일치하는지 확인하라.
