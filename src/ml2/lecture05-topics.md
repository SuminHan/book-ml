# Topics Covered

## MDP(Markov Decision Process)의 정식 정의

MDP는 다섯 가지 요소로 정의된다: \\((\mathcal{S}, \mathcal{A}, P, R, \gamma)\\)

- \\(\mathcal{S}\\): 가능한 상태들의 집합
- \\(\mathcal{A}\\): 가능한 행동들의 집합
- \\(P(s'|s,a)\\): 상태 \\(s\\)에서 행동 \\(a\\)를 했을 때 상태 \\(s'\\)로 전이될
  확률
- \\(R(s,a)\\): 상태 \\(s\\)에서 행동 \\(a\\)를 했을 때 받는 즉시 보상
- \\(\gamma \in [0,1)\\): 할인율(discount factor)

**마르코프 성질(Markov property)**: 다음 상태는 오직 **현재** 상태와 행동에만
의존한다 — 어떻게 지금 상태에 도달했는지(과거 전체 이력)는 상관없다. 예:
체스판의 현재 배치만 알면, 거기까지 어떤 수순을 거쳐왔는지는 다음 수를 정하는 데
필요 없다.

## 가치함수(Value Function)

정책 \\(\pi\\)를 따를 때, 상태 \\(s\\)의 가치는 그 상태에서 시작해 앞으로 받을
**할인된 누적 보상의 기댓값**이다:

\\[V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) \,\middle|\,
s_0 = s\right]\\]

## 벨만방정식: 재귀적 정의

가치함수를 직접 저 무한합으로 계산하는 대신, 재귀적 관계로 다시 쓸 수 있다:

\\[V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s,\pi(s)) V^\pi(s')\\]

직관: "지금 상태의 가치 = 지금 당장 받는 보상 + 할인된, 다음에 갈 상태들의 가치의
기댓값." 이 식이 성립하는 이유는 무한합 \\(\sum_{t=0}^\infty \gamma^t r_t = r_0 +
\gamma(r_1 + \gamma r_2 + \cdots)\\)를 "첫 항 + 할인된 나머지"로 다시 묶을 수
있기 때문이다 — 괄호 안이 바로 \\(V^\pi(s')\\)의 정의와 같다.

## 반복적 정책평가(Iterative Policy Evaluation)

결정론적 정책(각 상태에서 항상 같은 행동을 선택)이라면, 벨만방정식을 **연립방정식**으로
직접 풀 수도 있지만(상태가 적을 때), 상태가 많으면 대신 다음을 계속 반복해서 근사한다:

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

매 반복마다 모든 상태의 \\(V(s)\\)를 벨만방정식 우변으로 갱신하는데, 우변에 쓰이는
\\(V(s')\\)도 (아직 정확한 값이 아니라) 지금까지 추정한 값이다 — 그런데도 이 반복은
\\(\gamma < 1\\)이기만 하면 정확한 \\(V^\pi\\)로 수렴한다는 것이 증명돼 있다
(수축 사상, contraction mapping 성질에 근거하며, 이 학기에서는 결과만 받아들인다).

## 터미널 상태에서 거꾸로: 특수한 경우

만약 MDP가 종료 상태(terminal state, \\(V=0\\))를 가진 경로형 구조라면, 반복 계산
없이도 터미널부터 거꾸로 대입만으로 정확한 답을 즉시 구할 수 있다 — 이번 주 손유도
과제가 다루는 경우다. 이건 사실 반복적 정책평가가 "이미 정답을 아는 상태(터미널)"에서
시작해 한 방향으로만 전파되는 특수 케이스로 볼 수 있다.
