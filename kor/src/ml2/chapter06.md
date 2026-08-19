# Chapter 6. 시간차 학습 (Temporal-Difference Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter06_q_learning.ipynb)

1989년, 크리스 왓킨스(Chris Watkins)는 박사 학위 논문에서 **Q-learning**
이라는 알고리즘을 제안했다. Chapter 5의 몬테카를로는 모델 없이도 배울 수
있었지만, 에피소드가 끝날 때까지 기다려야 리턴을 계산할 수 있었다.
**시간차 학습**(Temporal-Difference, TD)은 이 기다림 자체를 없앤다 — 한
스텝만 진행해보고, "지금 추정치"와 "방금 관찰한 보상 + 다음 상태의
추정치"의 차이만큼 즉시 갱신한다. 모델도 필요 없고, 에피소드가 끝나기를
기다릴 필요도 없는, 강화학습에서 가장 널리 쓰이는 절충안이다.

## 6.1 TD(0): 한 스텝만 보고 갱신한다

Chapter 4의 벨만방정식 \\(V^\pi(s) = R(s,\pi(s)) + \gamma V^\pi(s')\\)를
다시 보자. 정책평가(모델 기반)는 이 식의 우변을 **모든** 다음 상태에
대한 기댓값으로 계산했다. TD(0)는 그 기댓값을, 실제로 한 번 관찰한
**샘플 하나**로 대체한다:

\\[V(s) \leftarrow V(s) + \alpha\big[r + \gamma V(s') - V(s)\big]\\]

대괄호 안을 **TD 오차**(TD error)라 부른다. MC와 비교하면 핵심 차이가
분명해진다: MC는 실제 리턴 \\(G_t\\)(에피소드 끝까지의 실제 보상 합)를
목표로 쓰지만, TD는 \\(r + \gamma V(s')\\)(한 스텝의 실제 보상 + 다음
상태 가치의 **추정치**)를 목표로 쓴다 — "아직 확실하지 않은 추정치를
가지고 스스로를 갱신한다"는 뜻에서 **부트스트래핑**(bootstrapping)이라
부른다. 이 부트스트래핑 덕분에 에피소드가 끝나지 않아도, 심지어 끝이
없는 과업에서도 매 스텝 학습이 가능해진다.

| | 몬테카를로 | 시간차 학습(TD) |
|---|---|---|
| 목표값 | 실제 리턴 \\(G_t\\)(에피소드 끝까지) | \\(r + \gamma V(s')\\)(한 스텝 + 추정치) |
| 갱신 시점 | 에피소드가 끝난 뒤 | 매 스텝 즉시 |
| 편향/분산 | 편향 없음(불편), 분산 큼 | 편향 있음(추정치를 씀), 분산 작음 |

## 6.2 Q-learning: 모델 없이, 상태-행동 가치로

Chapter 4의 \\(V(s)\\)는 "상태의 가치"였다. 이것만으로는 다음 행동을
고를 수 없다 — 어느 행동이 좋은 다음 상태로 이어지는지 알려면 전이확률
\\(P\\)가 필요하기 때문이다(Chapter 5.3에서 이미 짚은 이유). **Q-learning**
은 한 걸음 더 세분화한 **Q-함수** \\(Q(s,a)\\) — "상태 \\(s\\)에서 행동
\\(a\\)를 한 뒤, 그 이후 최적으로 행동했을 때의 기대 누적 보상" — 를
학습한다. \\(Q(s,a)\\)를 알면 최적 정책은 그냥 \\(\pi^*(s) = \arg\max_a
Q(s,a)\\)로 즉시 얻어진다.

에이전트가 상태 \\(s\\)에서 행동 \\(a\\)를 하고, 보상 \\(r\\)을 받고, 새
상태 \\(s'\\)에 도달할 때마다:

\\[Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)\right]\\]

```python
def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    max_next_q = max(Q[s_next].values()) if isinstance(Q[s_next], dict) else max(Q[s_next])
    td_error = r + gamma * max_next_q - Q[s][a]
    Q[s][a] += alpha * td_error
    return Q
```

## 6.3 SARSA: 실제로 고른 행동을 쓴다

Q-learning의 목표값은 \\(\max_{a'} Q(s',a')\\) — **항상 최선의 다음
행동을 가정**한다. 실제로 탐험(Chapter 2의 \\(\varepsilon\\)-greedy)
때문에 그 최선의 행동을 안 골랐더라도 상관없이 말이다. **SARSA**
(State-Action-Reward-State-Action, 이름 자체가 갱신에 필요한 다섯
요소를 그대로 나열한 것이다)는 다르게 접근한다 — 실제로 다음에 **고른**
행동 \\(a'\\)의 Q값을 목표로 쓴다:

\\[Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma Q(s',a') -
Q(s,a)\right]\\]

이 차이 하나가 두 알고리즘을 근본적으로 갈라놓는다. Q-learning은
행동 정책이 무엇이든(예: \\(\varepsilon\\)-greedy로 무작위 탐험을 섞어도)
**목표 정책**(최적 정책)의 가치를 직접 학습한다 — Chapter 5.4의 언어로
**off-policy**다. SARSA는 지금 실제로 따르고 있는 정책(탐험을 포함한
\\(\varepsilon\\)-greedy 그 자체) 의 가치를 학습한다 — **on-policy**다.

```python
import random

def epsilon_greedy(Q, s, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return max(range(n_actions), key=lambda a: Q[s][a])

def sarsa_train(env_step, n_states, n_actions, n_episodes, alpha, gamma, epsilon, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    for _ in range(n_episodes):
        s = start_state
        a = epsilon_greedy(Q, s, epsilon, n_actions)
        for _ in range(200):
            ns, r, done = env_step(s, a)
            na = epsilon_greedy(Q, ns, epsilon, n_actions)
            Q[s][a] += alpha * (r + gamma * Q[ns][na] - Q[s][a])  # 실제로 고른 na를 사용
            s, a = ns, na
            if done:
                break
    return Q
```

## 6.4 절벽 걷기: 두 알고리즘이 실제로 다른 정책을 배운다

"절벽 걷기"(Cliff Walking) 환경 — 시작점에서 목표까지 가는 길 바로
옆에 떨어지면 큰 벌점(-100)을 받는 절벽이 있는 격자 — 에서 두
알고리즘을 각각 500 에피소드 학습시키면, 놀랍도록 다른 경로를
배운다:

- **Q-learning**: 절벽 바로 옆을 스치듯 지나가는 **가장 짧은** 경로를
  배운다 — 목표 정책(탐욕적) 관점에서는 이게 진짜 최적이기 때문이다.
- **SARSA**: 절벽에서 멀리 돌아가는 **더 안전한** 경로를 배운다 —
  학습 중에는 여전히 \\(\varepsilon\\)-greedy로 가끔 무작위 행동을
  섞는데, 절벽 바로 옆을 지나가는 정책을 학습하면 그 무작위 행동
  때문에 실제로 가끔 절벽에 떨어져 큰 손해를 본다. SARSA는 **자신이
  실제로 행동하는 방식**(탐험 포함)까지 감안해서 가치를 매기므로, 그
  위험을 피하는 경로를 선호하게 된다.

**Q-learning은 "이상적으로 항상 최선을 다한다면"이라는 가정 아래 최적
경로를 찾고, SARSA는 "실제로 가끔 실수(탐험)할 수 있다"는 현실을 감안한
경로를 찾는다** — 어느 쪽이 "더 낫다"가 아니라, 애초에 답하는 질문이
다르다는 것이 이 예제가 보여주는 핵심이다.

## 6.5 왜 Q-learning이 수렴하는가(직관)

Chapter 4.5에서 확인했듯이 \\(Q^*(s,a)\\)는 유한한 MDP라면 반드시
존재하고 유일한 숫자다(벨만 최적방정식, 바나흐 고정점 정리). Q-learning은
그 목표를 직접 계산하지 않고, 샘플 기반 업데이트만으로 근사하는데도
정확히 그 값에 도달한다는 것이 증명돼 있다: **충분히 모든 상태-행동
쌍을 무한히 방문하고 학습률 \\(\alpha\\)를 적절히 줄여나가면** 참값
\\(Q^*(s,a)\\)로 수렴한다(Robbins-Monro 조건). 직관적으로는: TD 오차가
0이 되는 지점이 정확히 벨만 최적방정식을 만족하는 지점이므로, TD 오차를
계속 줄여나가는 이 업데이트는 결국 그 고정점으로 수렴하게 된다.

**시간차 학습은 몬테카를로의 "모델이 필요 없다"는 장점과, 동적계획법의
"한 스텝만 보고도 갱신한다"는 장점을 하나로 합친 것이다 — 그리고
Q-learning과 SARSA의 차이는, 똑같은 부트스트래핑 아이디어를 "이상적인
목표 정책"에 적용하느냐 "실제 행동 정책"에 적용하느냐라는, 아주 작지만
결과는 크게 갈리는 선택에서 나온다.**

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

**2. (개념 서술)** 6.4절의 절벽 걷기 예제에서, 만약 학습이 끝난 뒤
\\(\varepsilon\\)을 0으로 낮춰서(더 이상 탐험하지 않고) SARSA가 배운
정책을 실행한다면, 그래도 여전히 먼 길로 돌아갈지, 아니면 짧은 길을
택하게 될지 예측하고 이유를 설명하라. (힌트: SARSA가 학습한 Q값
자체는 "탐험이 있는 상황"을 가정하고 계산된 값이라는 점을 생각하라.)

**3. (손유도, Tier A — 자유 유도)** Q-learning의 업데이트
\\(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)]\\)가 참값 \\(Q^*(s,a)\\)로 수렴하기 위해 필요한 조건을 다음
세 가지로 나눠 논증하라: (1) TD 오차가 정확히 0이 되는 \\(Q\\)가 벨만
최적방정식 \\(Q^*(s,a) = R(s,a) + \gamma \max_{a'} Q^*(s',a')\\)의
해와 같다는 것(단순 대수적 정리). (2) \\(\alpha\\)가 너무 크면(예:
항상 \\(\alpha=1\\)) 왜 수렴이 불안정해지는지(Chapter 2의 증분 평균
갱신과 연결지어). (3) 모든 상태-행동 쌍이 **무한히 많이 방문돼야**
수렴이 보장되는 이유(ε-greedy 탐험이 없다면 어떤 상태-행동 쌍은 영원히
시도되지 않을 수 있다는 점과 연결지어).

**정확성 확인**: 위 세 가지 논증이 각각 Q-learning 알고리즘의 어느
부분(TD 오차 계산, `alpha` 파라미터, `epsilon_greedy` 함수)에
대응하는지 짝지어라.
