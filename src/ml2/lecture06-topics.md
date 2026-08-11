# Topics Covered

## Q-함수: 상태와 행동 둘 다에 대한 가치

W05의 \\(V(s)\\)는 "상태의 가치"였다. Q-learning은 한 걸음 더 세분화한 **Q-함수**
\\(Q(s,a)\\) — "상태 \\(s\\)에서 행동 \\(a\\)를 한 뒤, 그 이후 최적으로 행동했을
때의 기대 누적 보상" — 를 학습한다. \\(Q(s,a)\\)를 알면 최적 정책은 그냥
\\(\pi^*(s) = \arg\max_a Q(s,a)\\)로 즉시 얻어진다 — "상태의 가치"만으로는 어떤
행동을 골라야 할지 알 수 없지만(전이확률을 알아야 함), "상태-행동의 가치"를 알면
바로 최선의 행동을 고를 수 있다는 게 핵심 이점이다.

## Q-learning 업데이트 규칙

에이전트가 상태 \\(s\\)에서 행동 \\(a\\)를 하고, 보상 \\(r\\)을 받고, 새 상태
\\(s'\\)에 도달할 때마다:

\\[Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') -
Q(s,a)\right]\\]

대괄호 안은 **TD 오차**(Temporal Difference error)라 부른다: "지금 추정한
\\(Q(s,a)\\)"와 "방금 관찰한 보상 + 다음 상태에서 최선을 다했을 때의 추정값"의
차이다. 이 오차만큼 \\(Q(s,a)\\)를 조금씩 보정해나간다 — W02의 경사하강법처럼,
"현재 추정치와 더 나은 추정치의 차이만큼 이동한다"는 같은 패턴이다.

```python
def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    max_next_q = max(Q[s_next].values())
    td_error = r + gamma * max_next_q - Q[s][a]
    Q[s][a] += alpha * td_error
    return Q
```

## ε-greedy 탐험

행동을 고를 때, 확률 \\(1-\varepsilon\\)로는 현재까지 가장 좋다고 아는 행동을
고르고(활용), 확률 \\(\varepsilon\\)로는 무작위 행동을 고른다(탐험):

```python
import random

def epsilon_greedy(Q, s, epsilon, actions):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda a: Q[s][a])
```

학습이 진행될수록 \\(\varepsilon\\)을 서서히 줄여가는 것(decay)이 흔한 전략이다 —
초반에는 많이 탐험하고, Q값이 신뢰할 만해지면 점점 활용 위주로 전환한다.

## Q-learning이 왜 수렴하는가(직관)

Q-learning은 \\(Q(s,a)\\)가 실제로 어떤 행동으로 데이터를 모았는지(탐험 정책)와
무관하게, **충분히 모든 상태-행동 쌍을 무한히 방문하고 학습률 \\(\alpha\\)를
적절히 줄여나가면** 참값 \\(Q^*(s,a)\\)로 수렴한다는 것이 증명돼 있다(Robbins-Monro
조건). 직관적으로는: TD 오차가 0이 되는 지점이 정확히 벨만 최적방정식
\\(Q^*(s,a) = R(s,a) + \gamma \max_{a'} Q^*(s',a')\\)을 만족하는 지점이므로,
TD 오차를 계속 줄여나가는 이 업데이트는 결국 그 고정점으로 수렴하게 된다. 이번 주
손유도 과제는 이 수렴 조건을 개념적으로(엄밀한 확률론적 증명 없이) 논증하는 것이다.

## Q-learning vs SARSA (참고)

Q-learning은 업데이트할 때 \\(\max_{a'} Q(s',a')\\)(**항상 최선의 다음 행동을
가정**)을 쓴다 — 실제로 탐험 때문에 그 최선의 행동을 안 골랐더라도. 이런 방식을
off-policy라 부른다. (SARSA는 실제로 고른 다음 행동의 Q값을 쓰는 on-policy
방법이다 — 이 학기에서는 다루지 않지만, Q-learning의 설계 선택을 이해하는 데
참고가 된다.)
