# Chapter 7. n-step 부트스트래핑, 적격흔적, 그리고 계획 (n-Step Bootstrapping, Eligibility Traces & Planning)

Chapter 5의 몬테카를로와 Chapter 6의 TD(0)는 언뜻 서로 다른 두 극단처럼
보인다 — MC는 에피소드 **끝까지**의 실제 보상을 목표로 쓰고, TD(0)는
**한 스텝**만 보고 나머지는 추정치로 대체한다. 이번 장은 이 둘이 사실
"몇 스텝을 실제로 볼 것인가"라는 하나의 다이얼 위의 양 끝일 뿐임을
보이고, 그 사이의 모든 지점을 아우르는 방법을 다룬다. 마지막으로는
경험뿐 아니라 학습된 모델도 함께 활용하는 계획(planning)까지 살펴본다.

## 7.1 n-step TD: MC와 TD(0) 사이의 다이얼

TD(0)의 목표값은 \\(r + \gamma V(s')\\)(1 스텝의 실제 보상 + 나머지는
추정치)였다. 이걸 일반화해서, \\(n\\) 스텝만큼 실제로 관찰한 보상을 쓰고
그 이후는 추정치로 대체하면:

\\[G_t^{(n)} = R_t + \gamma R_{t+1} + \cdots + \gamma^{n-1} R_{t+n-1} +
\gamma^n V(s_{t+n})\\]

\\(n=1\\)이면 정확히 TD(0)이고, \\(n \to \infty\\)(에피소드 끝까지)이면
정확히 몬테카를로다. \\(n\\)을 늘릴수록 실제 관찰에 더 의존하게 되어
**편향은 줄고 분산은 커진다**(Chapter 6.1의 표에서 본 그 트레이드오프의
연속선상에 있다) — \\(n\\)은 그 사이 어딘가에서 상황에 맞게 고르는
하이퍼파라미터다.

```python
def n_step_td(env_step, n, n_episodes, alpha, gamma, n_states, start_state):
    V = [0.0] * n_states
    for _ in range(n_episodes):
        states, rewards = [start_state], []
        s, T, t = start_state, float('inf'), 0
        while True:
            if t < T:
                a = 0 if False else __import__('random').randrange(2)  # 데모용 무작위 정책
                ns, r, done = env_step(s, a)
                states.append(ns)
                rewards.append(r)
                if done:
                    T = t + 1
                s = ns
            tau = t - n + 1  # tau번째 상태를 지금 갱신
            if tau >= 0:
                G = sum(gamma ** (i - tau - 1) * rewards[i]
                        for i in range(tau, min(tau + n, len(rewards))))
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1:
                break
            t += 1
    return V
```

`tau`(갱신 대상 시점)가 항상 현재 시점보다 \\(n-1\\)만큼 뒤처져 있다는
점에 주목하라 — \\(n\\) 스텝만큼의 실제 보상이 다 모여야 그 목표값을
계산할 수 있기 때문이다. 7개 상태짜리 1차원 격자에서 `n=3`으로 학습시켜
보면, 목표(오른쪽 끝, +1)에 가까운 상태일수록 가치가 매끄럽게 커지는
것을 확인할 수 있다(예: 왼쪽 끝 -0.84 근처에서 오른쪽으로 갈수록 0.69
근처까지 단조 증가).

## 7.2 적격흔적: 모든 n을 동시에

n-step TD는 매번 \\(n\\)을 하나로 고정해야 한다. **적격흔적**
(eligibility trace)은 다른 접근을 쓴다 — 모든 상태에 "최근에 얼마나
자주, 얼마나 최근에 방문했는가"를 나타내는 흔적 \\(e(s)\\)를 붙여두고,
TD 오차가 발생할 때마다 **그 흔적의 크기에 비례해서 모든 상태를 동시에**
갱신한다:

\\[e(s) \leftarrow \gamma\lambda \, e(s) + \mathbb{1}[s = s_t], \qquad
V(s) \leftarrow V(s) + \alpha \, \delta_t \, e(s) \; \text{(모든 } s \text{에 대해)}\\]

\\(\lambda \in [0,1]\\)이 새로운 다이얼이다 — \\(\lambda=0\\)이면 흔적이
매 스텝 거의 사라져서 TD(0)와 같아지고, \\(\lambda=1\\)이면 흔적이
전혀 안 줄어들어 몬테카를로에 가까워진다. **TD(\\(\lambda\\))**라는
이름은 정확히 이 다이얼에서 왔다 — n-step TD가 "\\(n\\) 하나를 고정해서
전방(forward)으로 내다보는" 방식이라면, 적격흔적은 "과거 방문 이력에
흔적을 남겨서 후방(backward)으로 갱신을 퍼뜨리는" 방식으로 사실상 같은
효과를 낸다는 것이 알려져 있다(전방 관점과 후방 관점의 등가성 — 자세한
증명은 이 학기 범위를 넘는다).

## 7.3 계획과 학습: Dyna-Q

지금까지의 모델-프리 방법(MC, TD)은 경험을 한 번 쓰고 나면 버렸다.
그런데 그 경험에는 사실 "이 상태에서 이 행동을 하면 대략 이런 보상과
다음 상태가 나오더라"라는, 환경에 대한 정보가 담겨 있다. **Dyna-Q**는
이 정보를 버리지 않고 간단한 **모델**로 저장해뒀다가, 실제 환경과
상호작용하는 사이사이에 그 모델을 이용해 **가상의** 추가 학습(계획,
planning)을 끼워 넣는다:

```python
import random

def dyna_q(env_step, n_states, n_actions, n_episodes, planning_steps, alpha, gamma, epsilon, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    model = {}  # (s, a) -> (r, s') 실제로 관찰한 것을 그대로 저장(결정론적 환경 가정)
    for _ in range(n_episodes):
        s = start_state
        for _ in range(100):
            a = random.randrange(n_actions) if random.random() < epsilon \
                else max(range(n_actions), key=lambda x: Q[s][x])
            ns, r, done = env_step(s, a)
            Q[s][a] += alpha * (r + gamma * max(Q[ns]) - Q[s][a])  # 1) 실제 경험으로 학습
            model[(s, a)] = (r, ns)                                 # 2) 모델 갱신
            for _ in range(planning_steps):                         # 3) 모델로 가상 학습(계획)
                (ps, pa), (pr, pns) = random.choice(list(model.items()))
                Q[ps][pa] += alpha * (pr + gamma * max(Q[pns]) - Q[ps][pa])
            s = ns
            if done:
                break
    return Q
```

`planning_steps`가 클수록 실제 환경과 상호작용을 한 번 할 때마다 그
경험을 모델을 통해 몇 번이고 "복습"하는 셈이다. 실제로 같은 격자
환경에서, 목표 상태(state 3, 오른쪽으로 가는 행동)의 Q값이 0.5를
넘어서기까지 필요한 에피소드 수를 비교하면 차이가 극적이다:
`planning_steps=0`(순수 Q-learning)은 253 에피소드가 걸리는데,
`planning_steps=10`은 단 9 에피소드만에 도달한다 — 매 실제 스텝마다
모델을 이용해 10번씩 추가로 학습하니, 같은 실제 경험으로 훨씬 더 많은
것을 뽑아내는 것이다.

## 7.4 세 가지 축을 한자리에

이번 장에서 다룬 것들을 한 표로 정리하면, 이번 학기 Block A 전체가
사실 "부트스트래핑을 얼마나 할 것인가"라는 하나의 질문에 대한 서로
다른 답이었다는 게 보인다:

| 방법 | 목표값 | 특징 |
|---|---|---|
| MC (Chapter 5) | 실제 리턴 \\(G_t\\) | 편향 없음, 분산 큼, 에피소드가 끝나야 함 |
| TD(0) (Chapter 6) | \\(r + \gamma V(s')\\) | 편향 있음, 분산 작음, 매 스텝 학습 |
| n-step TD | \\(n\\) 스텝 실제 + 나머지 추정 | 둘 사이의 절충, \\(n\\)이 다이얼 |
| TD(\\(\lambda\\)) | 모든 \\(n\\)을 흔적으로 가중평균 | \\(n\\)을 고정할 필요 없음, \\(\lambda\\)가 다이얼 |
| Dyna-Q | TD(0) + 학습된 모델로 계획 | 모델-프리와 모델-기반을 결합 |

**부트스트래핑(추정치로 스스로를 갱신하는 것)을 아예 안 하는 것(MC)과
매 스텝 하는 것(TD)은 두 극단일 뿐이다 — 실전에서는 그 사이 어딘가가,
그리고 경험을 재사용하는 계획까지 더하는 것이 대체로 더 낫다는 것이
이번 장의 결론이다.**

---

## 연습문제

**1. (코딩)** 위 `dyna_q`(핵심 줄은 빈칸으로 남겨져 있다고 가정)를
완성하고, `planning_steps`를 0과 10으로 바꿔가며 같은 상태-행동의
Q값이 0.5를 넘는 데 걸리는 에피소드 수를 비교하라:

```python
import random

def dyna_q(env_step, n_states, n_actions, n_episodes, planning_steps, alpha, gamma, epsilon, start_state):
    Q = [[0.0] * n_actions for _ in range(n_states)]
    model = {}
    # ADD ADDITIONAL CODE HERE!!
    # 1) epsilon-greedy로 행동 선택, 환경과 상호작용, Q-learning 갱신
    # 2) model[(s,a)] = (r, ns)로 모델 저장
    # 3) planning_steps번, model에서 무작위로 (s,a)->(r,ns)를 뽑아 같은 방식으로 Q 갱신

    return Q
```

**2. (개념 서술)** Dyna-Q의 모델은 환경이 결정론적이라고(같은 (s,a)는
항상 같은 (r,s')를 낸다고) 가정하고 있다. 만약 환경이 확률적이라면
(같은 (s,a)에서도 매번 다른 s'가 나올 수 있다면) `model[(s,a)] = (r,
ns)`처럼 마지막 경험 하나만 저장하는 방식에 어떤 문제가 생길지
설명하고, 이를 개선할 방법을 한 가지 제안하라.

**3. (손유도, Tier B — 힌트 제공)** \\(n\\)-step 리턴 \\(G_t^{(n)} =
\sum_{k=0}^{n-1} \gamma^k R_{t+k} + \gamma^n V(s_{t+n})\\)에서
\\(n=1\\)로 놓으면 정확히 TD(0)의 목표값 \\(R_t + \gamma V(s_{t+1})\\)이
됨을 확인하고, \\(n \to \infty\\)이면서 에피소드가 유한한 길이 \\(T\\)에서
끝난다고 할 때(\\(V(s_T) = 0\\)) \\(G_t^{(n)}\\)이 몬테카를로의 실제
리턴 \\(G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k}\\)와 같아짐을
보여라.

**정확성 확인**: 이 결과가 왜 "TD(0)와 MC가 n-step TD의 두 특수한
경우"라는 주장을 뒷받침하는지 한 문장으로 설명하라.
