# Chapter 11. 고급 정책 최적화: PPO (Proximal Policy Optimization)

Chapter 10의 REINFORCE와 Actor-Critic에는 공통된 한계가 있다: 궤적 하나를
모아서 경사 상승 한 번에 쓰고 나면 그 데이터는 버려진다(온-폴리시,
on-policy) — 정책이 바뀌는 순간, 옛 데이터로 계산한 확률/그래디언트는
더 이상 정확하지 않기 때문이다. 게다가 경사 상승 스텝이 한 번이라도
너무 크면 정책이 갑자기 나쁜 방향으로 확 바뀌어서 회복이 안 되는
경우도 있다 — 로봇 제어처럼 "한 번의 잘못된 업데이트로 정책이 완전히
무너지면 다시 좋아지기 어려운" 상황에서는 치명적이다. 2017년 OpenAI가
제안한 **PPO**(Proximal Policy Optimization, Schulman 외)는 데이터를
몇 번 더 재사용하면서도, 정책이 한 번에 너무 멀리 움직이지 못하게 막는
실용적인 해법이다.

## 11.1 신뢰 영역이라는 직관

산을 내려갈 때 안개가 껴서 한 치 앞만 보인다고 하자. 경사가 가장 가파른
방향으로 무작정 큰 걸음을 내디디면, 그 방향의 지형이 어떻게 바뀌는지도
모른 채 절벽으로 떨어질 수 있다. 안전한 전략은 "한 걸음의 크기를 믿을
수 있는 범위(신뢰 영역, trust region) 안으로 제한하는 것"이다 —
PPO의 핵심 아이디어가 바로 이것이다. 정책이 한 번의 업데이트로 너무
크게 바뀌지 않도록 제약을 걸어서, 매 스텝이 실제로 개선을 보장하는
좁은 범위 안에서만 이동하게 한다.

## 11.2 확률비와 중요도 샘플링

핵심 도구는 **확률비**(probability ratio)다 — 지금 업데이트하려는
정책 \\(\pi_\theta\\)와, 데이터를 모을 때 썼던 정책
\\(\pi_{\theta_{\text{old}}}\\)가 같은 행동에 부여하는 확률의 비다:

\\[r_t(\theta) := \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)},
\quad r_t(\theta_{\text{old}}) = 1\\]

\\(r_t\\)가 1보다 크면 그 행동을 전보다 더 선호하게 됐다는 뜻, 1보다
작으면 덜 선호한다는 뜻이다. 이 비율을 쓰면 "옛 정책이 모은 데이터"를
"새 정책 관점에서" 재사용할 수 있다(**중요도 샘플링** — Chapter 5.4에서
MC의 off-policy 학습에 처음 등장했던 바로 그 도구가 여기서 다시
쓰인다):

\\[L^{IS}(\theta) = \mathbb{E}_t[r_t(\theta) A_t]\\]

\\(A_t\\)는 Chapter 10.7에서 정의한 어드밴티지, \\(A_t := G_t -
V(s_t)\\)다. 문제는 이 목적함수를 그냥 최대화하면 \\(r_t\\)가
한없이 커지도록(같은 행동만 계속 더 밀어주도록) 학습이 폭주할 수
있다는 것이다 — 데이터는 재사용되지만, "정책이 너무 멀리 움직이지
않는다"는 보장은 어디에도 없다.

## 11.3 GAE: 어드밴티지를 더 안정적으로 추정하기

Chapter 10.7의 어드밴티지 \\(A_t = G_t - V(s_t)\\)는 실제 리턴 \\(G_t\\)를
그대로 쓰므로 여전히 분산이 크다. **GAE**(Generalized Advantage
Estimation)는 Chapter 7의 n-step/적격흔적과 정확히 같은 아이디어를
어드밴티지 추정에 적용한다 — 여러 \\(n\\)-step 어드밴티지 추정치를
\\(\lambda\\)로 가중평균해서, 분산은 낮추되 편향은 적당히만 감수하는
절충점을 찾는다:

\\[A_t^{\text{GAE}} = \sum_{k=0}^\infty (\gamma\lambda)^k \delta_{t+k},
\qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\\]

\\(\delta_t\\)는 Chapter 6.1의 TD 오차 그 자체다 — GAE는 결국 "여러
시점의 TD 오차를 적격흔적처럼 지수적으로 가중합해서 하나의 어드밴티지
추정치를 만든다"는 것으로 요약된다. \\(\lambda=0\\)이면 TD(0) 기반
어드밴티지(분산 작고 편향 큼), \\(\lambda=1\\)이면 몬테카를로 기반
어드밴티지(편향 없고 분산 큼)에 가까워진다 — Chapter 7.2의 다이얼이
여기서도 그대로 등장한다. 실전 PPO 구현은 거의 항상 GAE로 계산한
\\(A_t\\)를 쓴다.

## 11.4 클리핑: 너무 멀리 못 가게 막기

PPO의 해법은 단순하다 — \\(r_t\\)가 \\([1-\epsilon, 1+\epsilon]\\)
(보통 \\(\epsilon = 0.2\\)) 범위를 벗어나면, 그 구간 밖으로 나간
만큼의 이득을 깎아버린다:

\\[L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]\\]

min을 쓰는 이유: clip 안 한 값과 clip한 값 중 더 "비관적인"(작은)
쪽을 목적함수로 삼아서, 정책이 한쪽으로 너무 밀려가는 것에 대한
보상을 스스로 제한한다. 직관: 어드밴티지가 양수(좋은 행동)인데
\\(r_t\\)가 이미 \\(1+\epsilon\\)을 넘었으면, 더 밀어붙여도
그래디언트가 사라진다(이미 충분히 밀었으니 그만) — 반대로 어드밴티지가
음수(나쁜 행동)인데 \\(r_t\\)가 \\(1-\epsilon\\) 밑으로 떨어졌으면
역시 그래디언트가 사라진다(이미 충분히 눌렀으니 그만). 그 결과 "좋은
행동을 과도하게 밀어주지도, 나쁜 행동을 과도하게 눌러버리지도 않는"
암묵적인 신뢰 영역이 생긴다 — 11.1절에서 말한 "믿을 수 있는 범위"를
명시적으로 제약하는 식을 따로 풀 필요 없이, 클리핑 한 번으로 같은
효과를 낸다.

```python
def ppo_clip_loss(ratio, advantage, epsilon=0.2):
    unclipped = ratio * advantage
    clipped = max(min(ratio, 1 + epsilon), 1 - epsilon) * advantage
    return min(unclipped, clipped)  # 목적함수(최대화 대상)의 한 샘플분
```

## 11.5 왜 PPO가 표준이 됐는가

PPO 이전에 같은 문제(너무 큰 업데이트로 인한 정책 붕괴)를 풀던
방법은 **TRPO**(Trust Region Policy Optimization, 2015)였다 — KL
divergence 제약을 명시적으로 걸고, 2차 미분(피셔 정보 행렬)과
conjugate gradient, line search까지 동원하는 무거운 최적화였다. PPO는
그 복잡한 기계장치를 클리핑 한 줄로 대체하면서도 비슷한 안정성을
얻는다 — "이론적으로 완벽한 보장"(TRPO의 단조 개선 보장) 대신
"구현하기 쉽고 실전에서 잘 작동하는 근사"를 택한 선택이다.

이 실용성 덕분에 PPO는 지금 가장 널리 쓰이는 정책기반 알고리즘이
됐다: 로봇 보행 제어, OpenAI Five(Dota 2), AlphaStar(스타크래프트
II)뿐 아니라, 사람 피드백에 맞게 언어모델을 조정하는 RLHF의 핵심
도구로도 쓰인다 — 연속 제어(로봇의 관절 힘 같은)와 이산 선택(다음
토큰 고르기) 양쪽 모두에 그대로 적용될 만큼 범용적이다.

**Q-learning이 "얼마나 좋은지 평가한 뒤 최선을 고른다"는 간접적
전략이라면, 정책기반 방법은 "무엇을 할지" 자체를 직접 학습한다. GAE가
어드밴티지 추정의 분산을 줄이고, 클리핑이 그 추정을 바탕으로 한
업데이트가 너무 커지지 않게 막는다 — Chapter 13~14에서 다룰 로봇
시뮬레이션의 연속 제어 문제에 PPO가 표준으로 쓰이는 이유가 바로 이
안정성이다.**

---

## 연습문제

**1. (코딩)** 위 `ppo_clip_loss`(핵심 줄은 빈칸으로 남겨져 있다고
가정)와, GAE의 한 항 \\(\delta_t = r_t + \gamma V(s_{t+1}) -
V(s_t)\\)를 계산하는 `td_error`를 완성하라:

```python
def td_error(r, V_s, V_s_next, gamma):
    # ADD ADDITIONAL CODE HERE!!

def ppo_clip_loss(ratio, advantage, epsilon=0.2):
    # ADD ADDITIONAL CODE HERE!!
    # unclipped = ratio * advantage
    # clipped = ratio를 [1-epsilon, 1+epsilon]로 clip한 뒤 advantage 곱
    # 둘 중 최솟값 반환

print(ppo_clip_loss(ratio=1.5, advantage=1.0, epsilon=0.2))  # 1.2 (clip된 값이 더 작음)
print(ppo_clip_loss(ratio=0.5, advantage=1.0, epsilon=0.2))  # 0.5 (clip 안 된 값이 더 작음)
print(ppo_clip_loss(ratio=1.5, advantage=-1.0, epsilon=0.2)) # -1.5 (advantage가 음수면 부호가 바뀜에 주의)
```

**2. (개념 서술)** GAE의 \\(\lambda\\)를 0으로 두는 것과 1로 두는 것이
각각 Chapter 7에서 배운 어떤 방법과 대응되는지 설명하고, 왜 그
극단들이 각각 편향과 분산 중 어느 쪽 문제를 안고 있는지 서술하라.

**3. (손유도, Tier B — 힌트 제공)** \\(A_t^{\text{GAE}} =
\sum_{k=0}^\infty (\gamma\lambda)^k \delta_{t+k}\\)에서 \\(\lambda = 0\\)을
대입하면 \\(A_t^{\text{GAE}} = \delta_t\\)(TD 오차 하나)만 남음을
보여라. \\(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\\)의 정의를
이용해서, 이것이 Chapter 10.7의 1-step 어드밴티지 근사
\\(A_t \approx r_t + \gamma V(s_{t+1}) - V(s_t)\\)와 정확히 같은
형태임을 확인하라.

**정확성 확인**: \\(\lambda\\)를 0에서 1로 점점 키우면 GAE가 더 먼
미래의 TD 오차들까지 포함하게 되는데, 이게 왜 "몬테카를로 방향으로
다가간다"는 뜻인지 한 문장으로 설명하라.
