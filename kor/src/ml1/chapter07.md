# Chapter 7. 트리 기반 모델: 결정트리에서 GBDT까지 (Tree-Based Models: Decision Trees to GBDT)

"동물인가요? 네 다리로 걷나요? 야옹 소리를 내나요?" — 스무고개(20 Questions)
게임은 예/아니오 질문을 잘 골라서 물어보면, 스무 번 안에 어떤 답이든 좁혀낼 수
있다는 게임이다. **결정 트리**(Decision Tree)는 이 게임을 알고리즘으로 그대로
옮긴 것이다. 그런데 미국에서는 은행이 대출을 거절하면, 법(신용기회균등법)에
따라 **구체적인 거절 사유**를 신청자에게 알려줘야 한다. 트리 하나라면 "왜
거절됐는가"는 루트에서 리프까지의 질문들을 그대로 읽으면 되니 쉽다 — 그런데
트리 수백 개를 조합한 **GBDT**라면, 정확도는 훨씬 높아지지만 "왜"에 답하기는
훨씬 어려워진다. 이번 장은 트리 하나(결정 트리)에서 시작해 트리 여러 개를
합치는 두 가지 전략(랜덤 포레스트, GBDT)까지, 그리고 그 대가로 잃은
설명가능성을 되찾는 방법(SHAP)까지 다룬다.

## 7.1 결정 트리의 구조

각 내부 노드는 하나의 질문(예: "\\(x_2 > 5\\)?")이고, 각 리프(leaf) 노드는
예측값이다. 예측할 때는 루트에서 시작해 질문에 답하며 리프까지 내려간다.

## 7.2 "가장 잘 나누는 질문"이란: 지니불순도

스무고개를 잘하는 사람은 아무 질문이나 던지지 않는다 — "생물인가요?"처럼 답이
반반으로 갈릴 만한 질문을 먼저 던져야 정보를 가장 많이 얻는다. **지니불순도**
(Gini Impurity)는 노드 안의 데이터가 얼마나 "섞여" 있는지를 재는 지표다.
클래스가 \\(K\\)개이고, 클래스 \\(k\\)의 비율이 \\(p_k\\)일 때:

\\[G = 1 - \sum_{k=1}^K p_k^2\\]

노드 안이 한 클래스로만 순수하면(\\(p_k=1\\), 나머지 0) \\(G=0\\) — 가장 좋은
상태다. 클래스가 반반이면(\\(K=2\\), \\(p_1=p_2=0.5\\)) \\(G = 1 - 0.25 -
0.25 = 0.5\\) — 이진 분류에서 가장 나쁜 상태(최댓값)다.

## 7.3 정보이득 (Information Gain)

먼저 **엔트로피**(entropy, Chapter 2.6에서 섀넌의 정보이론으로 이미 도입했다)를
정의한다:

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

## 7.4 언제 나누기를 멈추는가

트리를 끝까지 키우면 모든 리프가 완벽히 순수해질 때까지 나눈다 — 학습 데이터는
100% 맞히지만, 새 데이터에는 취약한(과적합) 트리가 된다. 실무에서는 최대
깊이(max depth), 리프의 최소 샘플 수 같은 **정지 조건**(stopping criteria)으로
트리 크기를 제한하거나, 트리를 다 키운 뒤 불필요한 가지를 잘라내는
**가지치기**(pruning)를 한다 — Chapter 6의 정규화와 같은 목적(모델을 덜
유연하게 만들어 분산을 줄인다)을 트리 구조에 맞게 적용한 것이다.

## 7.5 트리 하나에서 숲으로: 랜덤 포레스트

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
같은 원리다.

## 7.6 트리를 더 강하게: GBDT

랜덤 포레스트는 트리 여러 개를 **독립적으로** 만들고 다수결로 합쳤다.
**GBDT**(Gradient Boosted Decision Trees)는 다른 전략을 쓴다: 트리를
하나씩 **순차적으로** 추가하는데, 매번 새로운 트리는 "지금까지의 예측이 아직
못 맞힌 부분(잔차, residual)"을 목표로 학습한다.

목표: \\(T\\)개의 트리 \\(f_1, \ldots, f_T\\)를 순차적으로 더해서 예측
\\(F_T(x) = \sum_{t=1}^T f_t(x)\\)를 만든다. \\(t\\)번째 트리를 학습할 때는,
지금까지 쌓은 예측 \\(F_{t-1}(x)\\)의 **잔차** \\(r^{(i)} = y^{(i)} -
F_{t-1}(x^{(i)})\\)를 새로운 목표값으로 삼아, 그 잔차를 예측하는 트리를
학습한다:

```python
def gbdt_fit(X, y, n_trees, learning_rate):
    trees = []
    predictions = [0.0] * len(y)  # F_0(x) = 0
    for t in range(n_trees):
        residuals = [y[i] - predictions[i] for i in range(len(y))]
        tree = fit_single_tree(X, residuals)  # 잔차를 목표로 트리 하나 학습
        trees.append(tree)
        for i in range(len(y)):
            predictions[i] += learning_rate * tree.predict(X[i])
    return trees
```

`learning_rate`(축소율, shrinkage)는 각 트리의 기여를 일부러 줄여서, 한 트리가
과적합하는 것을 막고 여러 트리가 조금씩 나눠서 학습하게 한다 — Chapter 2의
경사하강법 학습률과 비슷한 역할이다. 실제로 "잔차를 향해 조금씩 이동한다"는 이
과정은 함수 공간에서의 경사하강법으로 볼 수 있다(그래서 이름이 **gradient**
boosting). XGBoost, LightGBM은 이 아이디어를 실무에서 쓸 수 있도록 극도로
최적화한 라이브러리이며, 지금도 정형(tabular) 데이터 대회(Kaggle 등)에서 가장
자주 우승하는 축에 속한다.

\\(F_{t-1}\\)이 이미 어느 정도 맞히고 있다면, 남은 오차(잔차)는 원래
\\(y\\)보다 작다. 그 작은 오차를 새 트리가 다시 줄이면, 전체 예측 \\(F_t =
F_{t-1} + \eta f_t\\)의 오차는 한 단계 더 작아진다. 트리를 계속 추가할수록
학습 데이터에 대한 오차는 이론상 계속 줄어든다 — 물론 실전에서는 검증 데이터
성능이 나빠지기 시작하는 시점(과적합)에서 멈춰야 한다(early stopping, Chapter
6의 검증 데이터 원칙과 동일하다).

## 7.7 정확한데 설명이 안 된다는 문제

GBDT는 강력하지만, 수백 개의 트리가 얽혀 있어 "이 예측 하나"에 어떤 특징이
얼마나 기여했는지 알기 어렵다. **SHAP**(SHapley Additive exPlanations)은
게임이론의 섀플리값(Shapley value) — "여러 사람이 협업해서 낸 성과를, 각자의
기여도에 따라 어떻게 공정하게 나눌 것인가"라는 문제의 답 — 을 빌려와서,
모델의 예측 하나를 "각 특징이 기여한 양"으로 정확히 분해한다.

## 7.8 SHAP: 예측 하나를 특징별로 분해하기

원래 섀플리값은 "여러 플레이어가 협력 게임에 참여할 때, 각 플레이어의 공정한
몫은 얼마인가"를 답한다 — 모든 가능한 참여 순서에서, 그 플레이어가 합류함으로써
늘어난 가치를 평균낸다.

SHAP은 "특징들"을 "플레이어들"로 바꿔서 적용한다: 특징 \\(j\\)의 SHAP값
\\(\phi_j\\)는, 특징들이 하나씩 추가되는 모든 가능한 순서에 대해 "특징
\\(j\\)가 추가됨으로써 예측값이 변한 정도"를 평균낸 것이다. 핵심 성질
(**가산성, additivity**):

\\[f(x) = \phi_0 + \sum_{j=1}^n \phi_j\\]

여기서 \\(\phi_0\\)는 기준값(baseline, 전체 데이터의 평균 예측), \\(\phi_j\\)는
특징 \\(j\\)가 그 기준에서 예측값을 얼마나 밀어올렸는지(양수) 또는
밀어내렸는지(음수)다. 이 예측 하나에 대한 모든 \\(\phi_j\\)의 합이 정확히
실제 예측값과 기준값의 차이와 같다는 것이 SHAP의 핵심 보장이다 — "왜 이
대출이 거절됐는가"에 대해 "신용점수 기여 -0.3, 소득 기여 +0.1, ..."처럼
정확히 합이 맞는 분해를 준다.

**작은 예제로 감 잡기**: 특징이 2개(\\(A, B\\))뿐이면, 가능한 추가 순서는
\\(A \to B\\)와 \\(B \to A\\) 두 가지뿐이다:

\\[\phi_A = \frac{1}{2}\left[\big(f(\{A\})-f(\{\})\big) +
\big(f(\{A,B\})-f(\{B\})\big)\right]\\]

특징이 많아지면 가능한 순서의 수가 \\(n!\\)로 폭발하므로, 실제 SHAP
라이브러리는 이 평균을 근사(sampling)로 빠르게 계산한다.

**결정 트리는 인간이 읽을 수 있는 규칙(if-then)의 나무이면서 동시에, "얼마나
잘 나누는가"를 정확히 수식으로 잴 수 있는 알고리즘이다. 트리를 여러 개
합치면(랜덤 포레스트, GBDT) 정확도는 올라가지만 그 읽기 쉬움을 잃는다 —
SHAP은 그 대가로 잃은 설명가능성을, 게임이론이라는 전혀 다른 도구로
되찾아오는 시도다.**

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

**2. (코딩)** 결정 트리 스텁 함수 `fit_stump`(깊이 1짜리 트리)가 이미 주어졌을
때, 다음과 같은 함수 `gbdt_fit`을 완성하라(핵심 줄은 빈칸으로 남겨져 있다고
가정):

```python
def gbdt_fit(X, y, n_trees, learning_rate):
    trees = []
    predictions = [0.0] * len(y)
    for t in range(n_trees):
        # ADD ADDITIONAL CODE HERE!!
        # 1. 잔차 = y - predictions 계산
        # 2. fit_stump(X, residuals)로 트리 하나 학습
        # 3. predictions에 learning_rate * 트리 예측값을 더해 누적

    return trees
```

**3. (개념 서술)** 랜덤 포레스트(트리를 독립적으로 병렬 학습)와 GBDT(트리를
순차적으로 학습)는 둘 다 "트리 여러 개를 합친다"는 앙상블이지만 학습
방식이 정반대다. 데이터에 노이즈(라벨링 오류)가 많이 섞여 있을 때 어느
쪽이 그 노이즈에 더 취약할지, 이유와 함께 답하라(힌트: GBDT는 잔차를
반복해서 좇는다는 점을 생각하라).

**4. (손유도, Tier A — 자유 유도)** 다음 8개 샘플이 있다: 클래스는
`[A,A,A,A,B,B,B,B]`이고, 특징 \\(x\\) 값은 `[1,2,3,4,5,6,7,8]`이다.

전체(8개)의 지니불순도를 손으로 계산한 뒤, \\(x \le 4\\)와 \\(x > 4\\)로
나누는 분기(threshold=4)의 정보이득(지니불순도 감소량)을 계산하라. threshold=2로
나눴을 때(왼쪽: `[A,A]`, 오른쪽: `[A,A,B,B,B,B]`)의 정보이득과 비교해 어느 쪽
분기가 데이터를 더 잘 나누는지 논하고, 문제 1의 `best_split(X, y, 0)` 결과와
일치하는지 확인하라.

**5. (손유도, Tier C — 폴백 준비 대상)** 특징이 2개(\\(A, B\\))인 장난감
모델의 예측함수 \\(f(S)\\)가 다음과 같이 주어졌다:

\\[f(\{\}) = 10, \quad f(\{A\}) = 16, \quad f(\{B\}) = 13, \quad f(\{A,B\}) = 20\\]

가능한 두 추가 순서(\\(A \to B\\), \\(B \to A\\))에서 각 특징의 한계
기여(marginal contribution)를 모두 구하고, \\(\phi_A\\)와 \\(\phi_B\\)를
계산하라. \\(\phi_0 + \phi_A + \phi_B = f(\{A,B\})\\)가 정확히 성립하는지
확인하라.

**빈칸채움형 폴백 버전**(자유 계산이 어려운 경우):

```
순서 A -> B: A의 한계 기여 = f({A}) - f({}) = 16 - 10 = ______________
             B의 한계 기여 = f({A,B}) - f({A}) = 20 - 16 = ______________
순서 B -> A: B의 한계 기여 = f({B}) - f({}) = 13 - 10 = ______________
             A의 한계 기여 = f({A,B}) - f({B}) = 20 - 13 = ______________

phi_A = (A->B에서 A의 기여 + B->A에서 A의 기여) / 2 = ______________
phi_B = (A->B에서 B의 기여 + B->A에서 B의 기여) / 2 = ______________
검산: phi_0 + phi_A + phi_B = f({}) + phi_A + phi_B = ______________ (20과 같아야 함)
```
