# Chapter 5. 몬테카를로 방법 (Monte Carlo Methods)

Chapter 4의 동적계획법은 강력하지만 치명적인 전제 하나에 기대고 있었다 —
전이확률 \\(P(s'|s,a)\\)를 정확히 알고 있어야 한다는 것이다. 카드 게임을
생각해보자: 딜러가 다음에 어떤 카드를 낼지에 대한 "확률표"를 미리 손에 쥐고
있는 사람은 없다. 그런데도 사람은 게임을 반복해서 해보는 것만으로 점점
나은 전략을 배운다. **몬테카를로**(Monte Carlo, MC) 방법은 바로 이 방식을
알고리즘으로 옮긴 것이다 — 모델을 전혀 몰라도, **에피소드를 끝까지
실제로 플레이해보고, 그 결과(실제로 받은 리턴)를 그대로 가치의 추정치로
쓴다.**

## 5.1 모델 없이 배운다: 경험으로부터의 학습

가치함수의 정의를 다시 떠올려보자: \\(V^\pi(s) = \mathbb{E}\_\pi[G_t \mid
s_t = s]\\) — "이 상태에서 시작해서 정책을 따랐을 때 받을 리턴의
**기댓값**"이다. 기댓값을 정확히 계산하려면 전이확률을 알아야 하지만,
기댓값은 **많은 샘플의 평균**으로도 근사할 수 있다는 것이 통계학의 기본
원리(대수의 법칙)다. MC는 정확히 이 원리를 쓴다 — 정책 \\(\pi\\)를 따라
에피소드를 여러 번 플레이해서, 상태 \\(s\\)를 방문했을 때 그 이후
실제로 받은 리턴들을 모아 평균 내면, 그게 곧 \\(V^\pi(s)\\)의 추정치가
된다.

## 5.2 첫방문 MC 예측

한 상태를 한 에피소드 안에서 여러 번 방문할 수도 있다(예: 게임에서
같은 위치로 되돌아오는 경우). **첫방문**(first-visit) MC는 각
에피소드에서 그 상태를 **처음** 방문했을 때의 리턴만 카운트한다(모든
방문을 다 쓰는 **모든방문**(every-visit) MC도 있다 — 이번 학기에서는
첫방문 방식만 다룬다):

```python
def mc_prediction(policy, env_sample_episode, n_episodes, gamma):
    # env_sample_episode(policy) -> [(state, action, reward), ...] 한 에피소드
    returns_sum = {}
    returns_count = {}
    for _ in range(n_episodes):
        episode = env_sample_episode(policy)
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if s not in visited:  # 첫방문만 카운트
                visited.add(s)
                returns_sum[s] = returns_sum.get(s, 0.0) + G
                returns_count[s] = returns_count.get(s, 0) + 1
    return {s: returns_sum[s] / returns_count[s] for s in returns_sum}
```

에피소드를 **뒤에서부터 앞으로** 훑는 이유: 리턴 \\(G_t = R_t + \gamma
G_{t+1}\\)은 재귀적으로 정의되므로, 맨 마지막 스텝부터 거꾸로 누적해
가면 매 스텝의 \\(G_t\\)를 한 번의 순회로 전부 계산할 수 있다 — Chapter
4에서 본 "터미널부터 거꾸로 대입" 패턴과 같은 아이디어다.

## 5.3 MC 제어: 정책 개선까지

예측(주어진 정책의 가치를 구하는 것)에서 제어(더 나은 정책을 찾는
것)로 넘어가려면, Chapter 4의 정책 반복과 같은 틀을 쓴다 — 다만 이제는
\\(V(s)\\) 대신 **Q(s,a)**를 추정한다(전이확률을 모르니 \\(V\\)만으로는
"어느 행동이 최선인지" 판단할 수 없기 때문 — Chapter 6에서 다시 강조할
이유다). 각 상태에서 Q값이 최대인 행동을 고르는 탐욕적 정책 개선을,
\\(\varepsilon\\)-greedy(Chapter 2에서 배운 그 전략)와 함께 반복한다 —
순수 탐욕적으로만 행동하면 한 번도 시도하지 않은 (상태, 행동) 쌍은
영원히 그 가치를 알 수 없기 때문이다.

```python
import random

def mc_control(env_step, n_states, n_actions, n_episodes, epsilon, gamma, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    counts = [[0] * n_actions for _ in range(n_states)]

    def generate_episode():
        s, episode = start_state, []
        for _ in range(100):
            a = random.randrange(n_actions) if random.random() < epsilon \
                else max(range(n_actions), key=lambda x: Q[s][x])
            ns, r, done = env_step(s, a)
            episode.append((s, a, r))
            s = ns
            if done:
                break
        return episode

    for _ in range(n_episodes):
        episode = generate_episode()
        G, visited = 0.0, set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                counts[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / counts[s][a]  # Chapter 2의 증분 평균과 같은 갱신
    return Q
```

작은 1차원 격자 환경(양 끝에 종료 보상 -1, +1)에서 5000 에피소드를
학습시키면, 목표(+1) 쪽에 가까운 상태일수록 그 방향으로 가는 행동의
Q값이 더 크게 학습된다 — 예를 들어 목표 바로 앞 상태에서는 목표 방향
행동의 Q값이 거의 1.0에 가깝게 수렴한다.

## 5.4 중요도 샘플링: off-policy로 가는 다리

지금까지의 MC는 "행동에 쓰는 정책"과 "가치를 평가하려는 정책"이
같았다(**on-policy**) — \\(\varepsilon\\)-greedy로 행동하면서 그
\\(\varepsilon\\)-greedy 정책 자체의 가치를 학습했다. 그런데 만약
"무작위로 행동해서 모은 데이터"로 "탐욕적 정책의 가치"를 알고 싶다면
(**off-policy**), 그냥 평균을 내면 안 된다 — 데이터를 모을 때 쓴 정책
(**행동 정책**, behavior policy \\(b\\))과 평가하려는 정책(**목표
정책**, target policy \\(\pi\\))이 다르면, 관찰된 리턴을 그대로 평균
내는 것은 편향된(biased) 추정이 된다.

**중요도 샘플링**(importance sampling)은 각 에피소드의 리턴에
"이 궤적이 \\(\pi\\)에서 나왔을 확률 대비 \\(b\\)에서 나왔을 확률의
비율"을 가중치로 곱해서 이 편향을 보정한다:

\\[\rho = \prod_{t} \frac{\pi(a_t|s_t)}{b(a_t|s_t)}\\]

직관: 목표 정책 \\(\pi\\)라면 거의 선택하지 않았을 행동을 행동 정책
\\(b\\)가 우연히 선택해서 얻은 궤적은, 그 희소함(비율 \\(\rho\\)가
작음)만큼 가중치를 낮춰서 반영한다. 이 아이디어는 Chapter 9(DQN)의
경험재현 버퍼(과거 정책들이 모은 데이터를 재사용)와, Chapter 11(PPO)의
확률비(probability ratio)에서 형태를 바꿔 다시 등장한다.

## 5.5 MC의 한계와 다음 장

MC는 모델이 필요 없다는 강력한 장점이 있지만, **에피소드가 끝나야만**
학습할 수 있다는 근본적인 제약이 있다 — 리턴 \\(G_t\\) 자체가 에피소드
끝까지의 보상을 다 더한 값이기 때문이다. 게임이 아주 길거나 끝나지 않는
과업(continuing task)이라면 MC는 아예 쓸 수 없다. Chapter 6의 시간차
학습은 "에피소드가 끝나기를 기다리지 않고, 한 스텝만 보고도 즉시
갱신한다"는 완전히 다른 절충안으로 이 문제를 푼다.

**몬테카를로는 "모델을 몰라도, 충분히 많이 시도해보면 평균이 진실에
수렴한다"는 통계학의 가장 기본적인 원리를 강화학습에 그대로 옮긴
것이다 — 다만 그 대가로, 결과를 알기 위해 항상 게임이 끝날 때까지
기다려야 한다.**

---

## 연습문제

**1. (코딩)** 다음 1차원 격자 환경(`step` 함수, state 0~4, action
0=왼쪽/1=오른쪽, 상태 0과 4가 종료 상태)에 대해 위 `mc_control`(핵심
줄은 빈칸으로 남겨져 있다고 가정)을 완성하라:

```python
import random

def step(s, a):
    ns = s - 1 if a == 0 else s + 1
    ns = max(0, min(4, ns))
    if ns == 0:
        return ns, -1.0, True
    if ns == 4:
        return ns, 1.0, True
    return ns, 0.0, False

def mc_control(env_step, n_states, n_actions, n_episodes, epsilon, gamma, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    counts = [[0] * n_actions for _ in range(n_states)]
    # ADD ADDITIONAL CODE HERE!!
    # generate_episode 내부 함수 정의(epsilon-greedy로 행동 선택, env_step으로 전이)
    # 에피소드 생성 후 뒤에서부터 첫방문 리턴 계산, Q 증분 갱신

    return Q

random.seed(1)
Q = mc_control(step, 5, 2, 5000, 0.1, 0.9, start_state=2)
for s in range(5):
    print(s, [round(v, 2) for v in Q[s]])
# state 3에서 action 1(오른쪽, 목표 방향)의 Q값이 가장 커야 함
```

**2. (개념 서술)** 첫방문 MC와 모든방문 MC는 같은 데이터에서 다른
추정치를 낼 수 있다. 한 에피소드 안에서 같은 상태를 여러 번 방문하는
구체적인 상황을 하나 예로 들고, 왜 두 방식의 추정치가 달라질 수 있는지
설명하라.

**3. (손유도, Tier B — 힌트 제공)** 행동 정책 \\(b\\)가 두 행동을
각각 50%씩 고르는 무작위 정책이고, 목표 정책 \\(\pi\\)는 항상 행동
0만 고르는 결정론적 정책이라 하자. 길이 2인 궤적
\\((s_0,a_0{=}0),(s_1,a_1{=}0)\\)에 대해 중요도 샘플링 비율 \\(\rho =
\prod_t \frac{\pi(a_t|s_t)}{b(a_t|s_t)}\\)를 계산하라. 만약 궤적의
두 번째 행동이 \\(a_1{=}1\\)이었다면(\\(\pi\\)라면 절대 고르지 않았을
행동) \\(\rho\\)가 어떻게 되는지도 계산하고, 그 결과가 왜 "이 궤적은
목표 정책의 가치 추정에 전혀 기여하지 않는다"는 뜻인지 한 문장으로
설명하라.
