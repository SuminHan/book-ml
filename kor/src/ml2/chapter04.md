# Chapter 4. 동적계획법 (Dynamic Programming)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter04_policy_evaluation.ipynb)

Chapter 3에서 MDP를 정식화했지만, "누적 보상을 최대화하는 행동을 고른다"는
목표는 여전히 계산 불가능해 보인다 — 무한히 먼 미래까지 내다봐야 하는
것처럼 보이기 때문이다. 벨만의 통찰은 이 무한합을 **재귀적 관계**로 다시
쓸 수 있다는 것이었다. 이번 장은 그 재귀식(벨만방정식)을 유도하고, MDP의
전이확률 \\(P\\)를 정확히 알고 있다는 전제 하에(모델 기반, model-based)
최적 정책을 계산하는 절차를 다룬다.

## 4.1 가치함수

"지금 이 상태에서 시작해서, 이 정책을 계속 따른다면, 기대할 수 있는 누적
보상은 얼마인가?" — 이 질문에 정확한 숫자로 답하는 것이 가치함수다. 정책
\\(\pi\\)를 따를 때, 상태 \\(s\\)의 가치는 그 상태에서 시작해 앞으로 받을
할인된 누적 보상의 기댓값이다:

\\[V^\pi(s) = \mathbb{E}\_\pi\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) \,\middle|\,
s_0 = s\right]\\]

## 4.2 벨만 기대방정식: 재귀적 정의

저 무한합을 재귀적 관계로 다시 쓸 수 있다:

\\[V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s,\pi(s)) V^\pi(s')\\]

직관: "지금 상태의 가치 = 지금 당장 받는 보상 + 할인된, 다음에 갈
상태들의 가치의 기댓값." 이 식이 성립하는 이유는 무한합 \\(\sum_{t=0}^\infty
\gamma^t r_t = r_0 + \gamma(r_1 + \gamma r_2 + \cdots)\\)를 "첫 항 + 할인된
나머지"로 다시 묶을 수 있기 때문이다 — 괄호 안이 바로 \\(V^\pi(s')\\)의
정의와 같다. 이 통찰 덕분에, "게임이 끝날 때까지"를 통째로 계산하는 대신,
한 스텝 앞만 내다보는 계산을 반복하는 것만으로 정확한 답에 도달할 수 있다.

## 4.3 반복적 정책평가

상태가 많으면, 다음을 값이 더 이상 바뀌지 않을 때까지 반복해서 근사한다:

```python
def policy_evaluation(P, R, policy, gamma, theta=1e-6):
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        delta = 0
        for s in range(n_states):
            a = policy[s]
            v_new = R[s][a] + gamma * sum(prob * V[next_s] for prob, next_s in P[s][a])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V
```

매 반복마다 모든 상태의 \\(V(s)\\)를 벨만방정식 우변으로 갱신하는데,
우변에 쓰이는 \\(V(s')\\)도 (아직 정확한 값이 아니라) 지금까지 추정한
값이다 — 그런데도 이 반복은 \\(\gamma < 1\\)이기만 하면 정확한 \\(V^\pi\\)로
수렴한다는 것이 증명돼 있다(**수축 사상**, contraction mapping 성질 —
4.5절에서 다시 다룬다).

## 4.4 벨만 최적방정식과 정책 반복

지금까지는 "주어진 정책을 평가하는 법"을 배웠다. 강화학습이 정말로 풀고
싶은 문제는 "가장 좋은 정책을 찾는 것"이다. 최적 가치 \\(V^*(s)\\)를
다음 **벨만 최적방정식**으로 정의한다 — 각 상태에서, 그 이후를 최적으로
이어갔을 때 가치가 가장 커지는 행동을 골랐다고 가정했을 때 나오는 가치다:

\\[V^*(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]\\]

**정책 반복**(Policy Iteration)은 이 최적해를 향해 두 단계를 번갈아
반복한다:

1. **정책평가**: 현재 정책 \\(\pi\\)의 \\(V^\pi\\)를 계산한다(4.3절).
2. **정책개선**: 각 상태에서, \\(V^\pi\\) 기준으로 더 나은 행동이 있으면
   정책을 그 행동으로 바꾼다: \\(\pi'(s) = \arg\max_a[R(s,a) + \gamma
   \sum_{s'} P(s'|s,a) V^\pi(s')]\\).

정책이 더 이상 안 바뀔 때까지 반복하면 최적 정책에 도달한다. **가치
반복**(Value Iteration)은 이 둘을 합쳐서, 정책평가를 끝까지 수렴시키지
않고 딱 한 번만 하고 바로 개선하는 것을 매 스텝 반복한다 — 벨만
최적방정식 자체를 반복 대입으로 직접 푸는 것과 같다:

```python
def value_iteration(P, R, n_actions, gamma, theta=1e-6):
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        delta = 0
        for s in range(n_states):
            v_new = max(R[s][a] + gamma * sum(prob * V[next_s] for prob, next_s in P[s][a])
                        for a in range(n_actions))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    policy = [max(range(n_actions),
                  key=lambda a: R[s][a] + gamma * sum(prob * V[next_s] for prob, next_s in P[s][a]))
              for s in range(n_states)]
    return V, policy
```

## 4.5 왜 이 반복이 수렴하는가: 바나흐 고정점 정리

"적어도 하나의 최적 정책이 항상 존재한다"는 사실은 자명하지 않다 — 어떤
상태에서는 정책 \\(\pi\\)가 낫고 다른 상태에서는 \\(\pi'\\)이 나을 수
있어서, "모든 정책 중 최댓값"이라는 개념 자체가 무너질 수도 있기
때문이다. 이 문제는 4.2·4.4절의 우변(벨만 기대방정식과 벨만
최적방정식 각각의 우변)을 하나의 **연산자** \\(T\\)로 보면 풀린다 —
\\(T\\)는 **\\(\gamma\\)-축소 사상**(contraction mapping)이라는 것이
증명돼 있다: 임의의 두 가치함수 \\(V_1, V_2\\)에 대해,

\\[\|TV_1 - TV_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty\\]

즉 \\(T\\)를 한 번 적용할 때마다 두 함수 사이의 (최대) 차이가 최소
\\(\gamma\\)배로 줄어든다. **바나흐 고정점 정리**(Banach fixed point
theorem)는 이런 축소 사상이 항상 **유일한 고정점**을 가지고, 어디서
출발해 반복해도 그 고정점에 수렴한다는 것을 보장한다 — 그래서 벨만
최적방정식을 만족하는 \\(V^*\\)가 존재하고 유일하며, 4.4절의 가치반복
(임의의 값에서 시작해 \\(T\\)를 반복 적용하는 것)이 실제로 그 값에
도달한다.

\\(V^*\\)를 알면, 각 상태에서 그 \\(\max\\)를 실제로 달성하는 행동을
그대로 고르는 결정론적 정책 \\(\pi^*\\)를 만들 수 있다. 이 정의가 항상
가능한 이유는 단순하다 — 각 상태에서 고를 수 있는 행동의 집합이
**유한**하기 때문에, 유한집합의 최댓값은 항상 어딘가에서 달성된다.

이 증명이 보장하는 건 "존재"이지 "계산 가능성"이나 "저장 가능성"이
아니라는 점도 중요하다 — 바둑처럼 상태 수가 초천문학적으로 많으면(약
\\(10^{170}\\)), \\(\pi^*\\)가 이론적으로 존재해도 모든 상태에 대해
표로 저장하는 것 자체가 불가능하다. Chapter 9(DQN)에서 배울, 표 대신
신경망으로 근사하는 방법이 필요해지는 이유가 바로 여기서 나온다. (이
거듭제곱법이 왜 수렴하는지는, 그래프 위에서 무작위 걷기의 정상분포를
구하는 PageRank 알고리즘과 정확히 같은 수학이다.)

## 4.6 모델 기반이라는 전제

이번 장의 모든 알고리즘은 전이확률 \\(P(s'|s,a)\\)를 **정확히 알고
있다는** 전제 위에 서 있다 — 체스라면 "이 수를 두면 상대는 어떻게
반응할 확률이 얼마인가"를 미리 다 알아야 한다는 뜻인데, 현실에서는
대부분 이걸 알 수 없다. Chapter 5·6은 이 모델을 몰라도 경험만으로
학습하는 방법(몬테카를로, 시간차 학습)을 다룬다.

**동적계획법의 핵심 아이디어("큰 문제 = 작은 부분 문제 + 재귀")는
컴퓨터공학 전반에서 반복해서 등장하는 패턴이고, 강화학습은 그 패턴을
"미래의 불확실한 보상을 어떻게 지금 계산할 것인가"라는 문제에 적용한
것이다 — 그리고 그 반복이 항상 정답에 수렴한다는 것 자체가, 축소
사상이라는 하나의 수학적 성질로 보장된다.**

---

## 연습문제

**1. (코딩)** 3-state MDP(state 0,1,2)가 주어질 때, `policy_evaluation`과
`value_iteration`(핵심 줄은 빈칸으로 남겨져 있다고 가정)을 완성하라:

```python
def policy_evaluation(P, R, policy, gamma, theta=1e-6):
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        # ADD ADDITIONAL CODE HERE!!
        # delta 초기화 -> 각 상태에 대해 v_new 계산, delta 갱신, V[s] 갱신
        # -> theta보다 delta가 작으면 종료

    return V

def value_iteration(P, R, n_actions, gamma, theta=1e-6):
    # ADD ADDITIONAL CODE HERE!!
    # policy_evaluation과 비슷하되, 정책을 따르는 대신 max_a를 취함
    # 수렴 후 각 상태에서 최선의 행동으로 policy 구성

    return V, policy
```

**2. (손유도, Tier B — 힌트 제공)** 다음 3-state MDP를 놓고 각 상태의
가치 \\(V(s)\\)를 구한다: State 0은 항상 State 1로 이동(보상 -1), State
1은 항상 State 2로 이동(보상 -1), State 2는 종료 상태(\\(V(2)=0\\)).
할인율 \\(\gamma = 0.9\\).

벨만방정식 \\(V(s) = R(s) + \gamma V(s')\\)을 State 2(터미널)부터
거꾸로 대입하여 \\(V(1), V(0)\\)을 **직접 계산**하라.

**힌트**: 터미널 상태부터 거꾸로 풀면(backward induction) 연립방정식을
한 번에 풀 필요 없이 대입만으로 답이 나온다. \\(V(2) = 0\\) →
\\(V(1) = -1 + 0.9 \times V(2) = ?\\) → \\(V(0) = -1 + 0.9 \times V(1)
= ?\\)

**정확성 확인**: 문제 1의 `policy_evaluation`으로 같은 MDP를 실행했을 때
나오는 값과 손계산 값이 일치하는지 비교하라.

**3. (손유도, Tier C — 폴백 준비 대상)** \\(T\\)가 \\(\gamma\\)-축소
사상이라는 성질 \\(\|TV_1 - TV_2\|_\infty \le \gamma \|V_1 -
V_2\|_\infty\\)을 받아들이고, 이로부터 "임의의 초기값 \\(V_0\\)에서
시작해 \\(T\\)를 반복 적용하면 \\(V^*\\)로 수렴한다"는 결론이 왜
따라 나오는지 논증하라.

**빈칸채움형 폴백 버전** (자유 논증이 어려운 경우):

```
가정: T는 gamma-축소 사상이고, V*는 T의 고정점(TV* = V*)이다.

Step 1: ||V_1 - V*|| = ||TV_0 - TV*|| <= gamma * ||______________||
Step 2: 같은 방식으로 ||V_2 - V*|| = ||TV_1 - TV*|| <= gamma * ||______________||
                                    <= gamma^2 * ||______________||
Step 3: n번 반복하면 ||V_n - V*|| <= gamma^n * ||______________||

결론: gamma < 1이므로 n이 커질수록 gamma^n은 ______________ (0에 가까워진다/커진다)
      따라서 V_n은 ______________ (V*로 수렴한다/발산한다)
```

**정확성 확인**: 이 결과가 "정책평가·가치반복이 초기값 \\(V_0\\)를
아무렇게나(예: 전부 0으로) 잡아도 항상 같은 답에 도달한다"는 사실을
왜 보장하는지 한 문장으로 설명하라.
