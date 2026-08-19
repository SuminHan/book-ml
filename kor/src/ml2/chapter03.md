# Chapter 3. MDP 정식화 (Markov Decision Processes)

1950년대, 수학자 리처드 벨만(Richard Bellman)은 "차원의 저주(curse of
dimensionality)"라는 용어를 만들면서 동적계획법(dynamic programming)이라는
최적화 기법을 고안했다. 그의 핵심 통찰 — "복잡한 문제를 한 번에 풀지 말고,
더 작은 부분 문제로 쪼개서 그 답을 재귀적으로 조합하라" — 는 지금 강화학습
이론 전체의 뼈대를 이루고 있다. 이번 장은 Chapter 2의 밴딧에 "상태"를
더해서, 강화학습 문제를 수학적으로 엄밀하게 정의하는 틀 — **MDP**(Markov
Decision Process) — 를 세운다.

## 3.1 MDP의 다섯 요소

MDP는 다섯 가지로 정의된다: \\((\mathcal{S}, \mathcal{A}, P, R, \gamma)\\)

- \\(\mathcal{S}\\): 가능한 상태들의 집합
- \\(\mathcal{A}\\): 가능한 행동들의 집합
- \\(P(s'|s,a)\\): 상태 \\(s\\)에서 행동 \\(a\\)를 했을 때 상태 \\(s'\\)로
  전이될 확률
- \\(R(s,a)\\): 상태 \\(s\\)에서 행동 \\(a\\)를 했을 때 받는 즉시 보상
- \\(\gamma \in [0,1)\\): 할인율(discount factor)

Chapter 2의 밴딧과 비교하면 정확히 무엇이 추가됐는지 보인다 — 밴딧에는
\\(\mathcal{S}\\)도 \\(P\\)도 없었다(팔을 당겨도 "다음 상태"가 없었다).
MDP는 지금 한 행동이 다음 상태 \\(P(s'|s,a)\\)를 통해 미래 전체에 영향을
준다는 사실을 명시적으로 담는다.

## 3.2 마르코프 성질

**마르코프 성질**(Markov property): 다음 상태는 오직 **현재** 상태와
행동에만 의존한다 — 어떻게 지금 상태에 도달했는지(과거 전체 이력)는
상관없다:

\\[P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0) = P(s_{t+1} | s_t, a_t)\\]

예: 체스판의 현재 배치만 알면, 거기까지 어떤 수순을 거쳐왔는지는 다음
수를 정하는 데 필요 없다. 이 성질이 성립한다는 가정 덕분에, "지금까지의
전체 이력"이 아니라 "현재 상태" 하나만 기억하면 충분해진다 — 뒤에서
배울 알고리즘들이 상태 하나만 보고 결정을 내릴 수 있는 이유가 여기서
나온다.

## 3.3 누적 보상과 할인율

강화학습의 목표는 한 스텝의 보상이 아니라, 앞으로 받을 보상들의 합 —
**리턴**(return) \\(G_t\\) — 을 최대화하는 것이다:

\\[G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k}\\]

**왜 할인율 \\(\gamma\\)가 필요한가**: (1) 수학적으로, \\(\gamma < 1\\)이면
보상이 유계(bounded)인 한 이 무한급수가 항상 수렴한다(기하급수). \\(\gamma=1\\)
이면 끝나지 않는 과업에서 리턴이 무한대로 발산할 수 있다. (2) 직관적으로,
"지금 당장의 보상 1"이 "10스텝 뒤의 보상 1"보다 더 가치 있다고 보는
경우가 많다 — 사람의 시간 선호나, 금융의 이자율 개념과 같은 방향이다.
\\(\gamma\\)가 0에 가까우면 에이전트는 "근시안적"(당장의 보상만 중시)이고,
1에 가까우면 "장기적"(먼 미래의 보상까지 충분히 고려)이 된다.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)

# 이 환경에서 MDP의 다섯 요소를 실제로 확인해보자
print("상태 공간 S:", env.observation_space)   # 4차원 연속 벡터
print("행동 공간 A:", env.action_space)         # 2개의 이산 행동 (왼쪽/오른쪽 밀기)

total_return, gamma, discount = 0.0, 0.95, 1.0
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_return += discount * reward   # G_t를 직접 누적
    discount *= gamma
    if terminated or truncated:
        break
print("10스텝 동안의 할인된 리턴 G_0:", round(total_return, 3))
env.close()
```

이 코드에서 `reward`가 매 스텝 즉시 보상 \\(R(s,a)\\)이고, 환경이
내부적으로 다음 상태로 전이시키는 규칙이 곧 \\(P(s'|s,a)\\)다 —
Gymnasium 환경은 이 전이확률을 코드로 감춰두고 `step()` 함수 하나로
캡슐화한 것뿐, MDP의 정의와 정확히 대응된다.

## 3.4 가치함수를 향해

이제 "누적 보상을 최대화하는 행동을 고른다"는 목표는 명확해졌지만,
아직 계산할 방법은 없다 — 무한히 먼 미래까지 내다봐야 할 것 같다.
다음 장(동적계획법)은 이 무한합을 재귀적인 형태로 다시 쓰는
**벨만방정식**을 통해, 실제로 계산 가능한 절차로 바꾸는 방법을 다룬다.

**MDP는 "지금 한 행동이 미래의 상태 자체를 바꾼다"는 사실 하나를
Chapter 2의 밴딧에 추가한 것뿐이지만, 이 한 가지 차이가 강화학습을
훨씬 더 어렵고, 훨씬 더 흥미로운 문제로 만든다.**

---

## 연습문제

**1. (코딩)** 위 코드를 참고해서, `FrozenLake-v1`(`is_slippery=False`)
환경을 만들고 `env.observation_space`와 `env.action_space`를 출력해서
상태·행동 공간이 각각 이산(discrete)인지 연속(continuous)인지, 몇
가지인지 확인하라. CartPole과 비교했을 때 어떤 차이가 있는지 한 문장으로
서술하라.

```python
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
# ADD ADDITIONAL CODE HERE!!
# observation_space, action_space를 출력하고 CartPole과 비교
```

**2. (개념 서술)** 마르코프 성질이 실제로는 성립하지 않을 것 같은
상황을 하나 떠올려보라(예: 자율주행에서 "지금 화면 한 장"만으로 판단하는
경우, 안개 속에서 방금 지나간 표지판 정보가 필요할 수 있다). 그런
경우에도 마르코프 성질을 다시 성립시키려면 상태를 어떻게 다시 정의해야
할지(예: 최근 몇 프레임을 함께 상태에 포함시키는 등) 한 문단으로
제안하라.

**3. (손유도, Tier B — 힌트 제공)** 보상이 매 스텝 상수 \\(R\\)로
고정된 경우(즉 \\(R_t = R\\) for all \\(t\\)), 리턴 \\(G_t = \sum_{k=0}^\infty
\gamma^k R\\)이 \\(\gamma < 1\\)일 때 닫힌 형태 \\(G_t = \frac{R}{1-\gamma}\\)로
수렴함을 보여라.

**힌트**: 등비급수의 합 공식 \\(\sum_{k=0}^\infty x^k = \frac{1}{1-x}\\)
(\\(|x|<1\\)일 때)을 그대로 적용하면 된다. \\(R=1, \gamma=0.9\\)일 때
\\(G_t\\)의 값을 직접 계산하고, \\(\gamma=0.99\\)로 바꿨을 때 값이 어떻게
달라지는지도 계산해보라.

**정확성 확인**: \\(\gamma\\)가 1에 가까워질수록 \\(G_t\\)가 왜 점점
커지는지(발산에 가까워지는지) 계산 결과를 바탕으로 한 문장으로
설명하라.
