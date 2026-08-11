# Chapter 6. 강화학습 알고리즘 (Reinforcement Learning Algorithms)

1989년, 크리스 왓킨스(Chris Watkins)는 박사 학위 논문에서
**Q-learning**이라는 알고리즘을 제안했다. 지난 장에서 배운 정책평가는
환경의 전이확률 \\(P(s'|s,a)\\)를 정확히 알고 있어야 계산할 수 있었다 —
체스라면 "이 수를 두면 상대는 어떻게 반응할 확률이 얼마인가"를 미리 다
알아야 한다는 뜻인데, 현실에서는 대부분 이걸 알 수 없다. Q-learning의
혁신은 **환경의 모델을 전혀 몰라도**, 그저 행동해보고 결과를 관찰하는
것만으로 최적의 행동을 학습할 수 있다는 것을 보인 데 있다.

## 6.1 모델 기반 vs 모델-프리(Model-Free)

지난 장의 정책평가는 "모델 기반(model-based)" 방법이었다 — 환경이 어떻게
작동하는지(전이확률)를 알고 있다는 전제 하에 계산했다. 그런데 실제
게임이나 로봇 제어에서는 "이 버튼을 누르면 정확히 어떤 일이 일어날지"에
대한 완벽한 수학적 모델이 없는 경우가 대부분이다. **Q-learning**은
모델을 몰라도, 직접 행동해서 얻은 경험(상태, 행동, 보상, 다음 상태)만으로
학습하는 **모델-프리**(model-free) 방법이다.

## 6.2 Q-함수: 상태와 행동 둘 다에 대한 가치

Chapter 5의 \\(V(s)\\)는 "상태의 가치"였다. Q-learning은 한 걸음 더
세분화한 **Q-함수** \\(Q(s,a)\\) — "상태 \\(s\\)에서 행동 \\(a\\)를 한
뒤, 그 이후 최적으로 행동했을 때의 기대 누적 보상" — 를 학습한다.
\\(Q(s,a)\\)를 알면 최적 정책은 그냥 \\(\pi^*(s) = \arg\max_a
Q(s,a)\\)로 즉시 얻어진다 — "상태의 가치"만으로는 어떤 행동을 골라야
할지 알 수 없지만(전이확률을 알아야 함), "상태-행동의 가치"를 알면 바로
최선의 행동을 고를 수 있다는 게 핵심 이점이다.

## 6.3 Q-learning 업데이트 규칙

에이전트가 상태 \\(s\\)에서 행동 \\(a\\)를 하고, 보상 \\(r\\)을 받고,
새 상태 \\(s'\\)에 도달할 때마다:

\\[Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)\right]\\]

대괄호 안은 **TD 오차**(Temporal Difference error)라 부른다: "지금
추정한 \\(Q(s,a)\\)"와 "방금 관찰한 보상 + 다음 상태에서 최선을 다했을
때의 추정값"의 차이다. 이 오차만큼 \\(Q(s,a)\\)를 조금씩 보정해나간다 —
Chapter 2의 경사하강법처럼, "현재 추정치와 더 나은 추정치의 차이만큼
이동한다"는 같은 패턴이다.

```python
def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    max_next_q = max(Q[s_next].values())
    td_error = r + gamma * max_next_q - Q[s][a]
    Q[s][a] += alpha * td_error
    return Q
```

## 6.4 탐험과 활용의 딜레마: ε-greedy

새로운 식당을 고를 때를 생각해보자: 이미 맛있다고 아는 단골 식당에 또
갈 것인가(**활용, exploitation**), 아니면 아직 안 가본 식당을
시도해볼 것인가(**탐험, exploration**)? 강화학습 에이전트도 똑같은
딜레마에 빠진다 — 지금까지 가장 좋다고 알려진 행동만 계속하면(순수
활용) 더 좋은 행동을 영영 발견하지 못할 수 있고, 무작위로만
행동하면(순수 탐험) 학습한 지식을 전혀 활용하지 못한다.

행동을 고를 때, 확률 \\(1-\varepsilon\\)로는 현재까지 가장 좋다고 아는
행동을 고르고(활용), 확률 \\(\varepsilon\\)로는 무작위 행동을 고른다
(탐험):

```python
import random

def epsilon_greedy(Q, s, epsilon, actions):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: Q[s][a])
```

학습이 진행될수록 \\(\varepsilon\\)을 서서히 줄여가는 것(decay)이 흔한
전략이다 — 초반에는 많이 탐험하고, Q값이 신뢰할 만해지면 점점 활용
위주로 전환한다.

## 6.5 Q-learning이 왜 수렴하는가(직관)

Q-learning은 \\(Q(s,a)\\)가 실제로 어떤 행동으로 데이터를 모았는지
(탐험 정책)와 무관하게, **충분히 모든 상태-행동 쌍을 무한히 방문하고
학습률 \\(\alpha\\)를 적절히 줄여나가면** 참값 \\(Q^*(s,a)\\)로
수렴한다는 것이 증명돼 있다(Robbins-Monro 조건). 직관적으로는: TD
오차가 0이 되는 지점이 정확히 벨만 최적방정식 \\(Q^*(s,a) = R(s,a) +
\gamma \max_{a'} Q^*(s',a')\\)을 만족하는 지점이므로, TD 오차를 계속
줄여나가는 이 업데이트는 결국 그 고정점으로 수렴하게 된다.

## 6.6 Q-learning vs SARSA (참고)

Q-learning은 업데이트할 때 \\(\max_{a'} Q(s',a')\\)(**항상 최선의 다음
행동을 가정**)을 쓴다 — 실제로 탐험 때문에 그 최선의 행동을 안
골랐더라도. 이런 방식을 off-policy라 부른다. (SARSA는 실제로 고른
다음 행동의 Q값을 쓰는 on-policy 방법이다 — 이 학기에서는 다루지
않지만, Q-learning의 설계 선택을 이해하는 데 참고가 된다.)

**Q-learning은 "환경을 몰라도 배울 수 있다"는 것을 증명함으로써,
강화학습을 이론적 호기심에서 실제로 로봇·게임·추천시스템에 적용 가능한
도구로 바꾼 전환점이다.**

---

## 연습문제

**1. (코딩)** 다음과 같은 함수 `q_learning_train`을 완성하라(핵심 줄은
빈칸으로 남겨져 있다고 가정):

```python
import random

def epsilon_greedy(Q, s, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return max(range(n_actions), key=lambda a: Q[s][a])

def q_learning_train(transition, n_states, n_actions, n_episodes, alpha, gamma, epsilon):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    for episode in range(n_episodes):
        s = 0  # 매 에피소드는 state 0에서 시작한다고 가정
        for _ in range(20):  # 에피소드 최대 길이
            # ADD ADDITIONAL CODE HERE!!
            # 1. epsilon_greedy로 행동 a 선택
            # 2. transition(s, a)로 보상 r과 다음 상태 s_next 얻기
            # 3. Q-learning 업데이트 규칙 적용
            # 4. s_next가 터미널 상태(-1로 표시)면 에피소드 종료, 아니면 s = s_next
            if s_next == -1:
                break
            s = s_next
    return Q
```

**2. (손유도, Tier A — 자유 유도)** Q-learning의 업데이트
\\(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)]\\)가 참값 \\(Q^*(s,a)\\)로 수렴하기 위해 필요한 조건을 다음
세 가지로 나눠 논증하라: (1) TD 오차가 정확히 0이 되는 \\(Q\\)가 벨만
최적방정식 \\(Q^*(s,a) = R(s,a) + \gamma \max_{a'} Q^*(s',a')\\)의
해와 같다는 것(단순 대수적 정리). (2) \\(\alpha\\)가 너무 크면(예:
항상 \\(\alpha=1\\)) 왜 수렴이 불안정해지는지(Chapter 2의 학습률 문제와
연결지어). (3) 모든 상태-행동 쌍이 **무한히 많이 방문돼야** 수렴이
보장되는 이유(ε-greedy 탐험이 없다면 어떤 상태-행동 쌍은 영원히 시도되지
않을 수 있다는 점과 연결지어).

**정확성 확인**: 위 세 가지 논증이 각각 Q-learning 알고리즘의 어느
부분(TD 오차 계산, `alpha` 파라미터, `epsilon_greedy` 함수)에
대응하는지 짝지어라.
