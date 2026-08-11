# Problem Set

난이도 등급: **Tier B (적정하나 힌트 제공)**

**1.** (코딩) 3-state MDP (state 0,1,2)가 주어질 때, 고정된 정책 하에서 각 상태의
가치함수 \\(V(s)\\)를 반복적 정책평가(iterative policy evaluation)로 계산하라.

```python
def policy_evaluation(P, R, policy, gamma, theta=1e-6):
    # P[s][a] = [(prob, next_state), ...]  전이확률
    # R[s][a] = 즉시 보상(스칼라)
    # policy[s] = 결정론적 정책이 선택하는 행동 a
    # gamma = 할인율
    n_states = len(P)
    V = [0.0] * n_states
    while True:
        # ADD ADDITIONAL CODE HERE!!
        # delta = 0으로 초기화
        for s in range(n_states):
            a = policy[s]
            # v_new = R[s][a] + gamma * sum(prob * V[next_s] for ...)
            # delta = max(delta, |v_new - V[s]|)
            V[s] = v_new
        # theta보다 delta가 작으면 종료

    return V
```

**2.** 아래 3-state MDP를 손으로 풀 것이므로, 먼저 전이확률과 보상을 표로 정리하라
(문제 3에서 사용).

- State 0: 항상 State 1로 이동, 보상 -1
- State 1: 항상 State 2로 이동, 보상 -1
- State 2: 종료 상태(terminal), 보상 0

---

## 손유도 과제 (실습시간, Tier B — 힌트 제공)

### 3-state MDP 벨만방정식 연립방정식으로 직접 풀기

위 MDP에서 \\(\gamma = 0.9\\)일 때, 벨만방정식

\\[V(s) = R(s) + \gamma V(s')\\]

을 State 2(터미널, \\(V(2)=0\\))부터 거꾸로 대입하여 \\(V(1), V(0)\\)을 **직접
계산**하라.

**힌트**: 터미널 상태부터 거꾸로 풀면(backward induction) 연립방정식을 한 번에 풀
필요 없이 대입만으로 답이 나온다. \\(V(2) = 0\\) → \\(V(1) = -1 + 0.9 \times V(2)
= ?\\) → \\(V(0) = -1 + 0.9 \times V(1) = ?\\)

**정확성 확인**: 코드 문제 1로 같은 MDP를 실행했을 때 나오는 값과 손계산 값이
일치하는지 비교하라.
