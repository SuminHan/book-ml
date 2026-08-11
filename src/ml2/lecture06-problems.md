# Problem Set

난이도 등급: **Tier A (자유 유도)**

**1.** (코딩) 다음과 같은 함수 `q_learning_train`을 작성하라:

- input parameter: 결정론적 3-state MDP를 나타내는 `transition(s, a) -> (r, s_next)`
  함수, 상태 개수 `n_states`, 행동 개수 `n_actions`, 에피소드 수 `n_episodes`,
  학습률 `alpha`, 할인율 `gamma`, 탐험률 `epsilon`
- return value: 학습된 Q-테이블 (`Q[s][a]` 형태의 2차원 리스트)

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

**2.** ε을 학습 초반에는 크게(예: 0.9), 후반으로 갈수록 점점 작게(decay) 줄여가는
전략이 왜 고정된 ε(예: 항상 0.1)보다 실무에서 더 흔히 쓰이는지 한 문단으로
설명하라.

---

## 손유도 과제 (실습시간, Tier A — 자유 유도)

### Q-learning 수렴 조건(Robbins-Monro) 개념적 증명

Q-learning의 업데이트 \\(Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'}
Q(s',a') - Q(s,a)]\\)가 참값 \\(Q^*(s,a)\\)로 수렴하려면, 직관적으로 다음이 필요하다:

**논증할 것**:

1. TD 오차 \\(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\\)가 정확히 0이 되는
   \\(Q\\)가 벨만 최적방정식 \\(Q^*(s,a) = R(s,a) + \gamma \max_{a'}
   Q^*(s',a')\\)의 해와 같다는 것을 보여라 (단순 대수적 정리).
2. \\(\alpha\\)가 너무 크면(예: 항상 \\(\alpha=1\\)) 왜 수렴이 불안정해지는지
   설명하라 — 매번 새 관측치로 \\(Q\\)를 완전히 덮어써버리면, 하나의 (운 나쁜)
   관측치에 지나치게 흔들리게 된다는 점을 W02의 학습률 문제와 연결지어 논하라.
3. 모든 상태-행동 쌍이 **무한히 많이 방문돼야** 수렴이 보장되는 이유를, ε-greedy의
   탐험 없이(순수 활용만) 학습한다면 어떤 상태-행동 쌍은 영원히 시도되지 않을 수
   있다는 점과 연결지어 설명하라.

**정확성 확인**: 위 세 가지 논증(수렴 조건이 만족하는 방정식, 학습률의 역할, 탐험의
필요성)이 각각 Q-learning 알고리즘의 어느 부분(TD 오차 계산, `alpha` 파라미터,
`epsilon_greedy` 함수)에 대응하는지 짝지어라.
