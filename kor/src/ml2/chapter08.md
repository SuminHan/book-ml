# Chapter 8. 정책기반 강화학습 (Policy-Based Reinforcement Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter08_reinforce_ppo.ipynb)

로봇 팔의 관절에 가할 힘을 정하는 문제를 생각해보자. 이 "행동"은
-10Nm부터 +10Nm까지 **연속적인 값** 중 아무거나 될 수 있다. Chapter
6의 Q-learning은 매 스텝 \\(\max_{a'} Q(s',a')\\)를 계산해야 하는데,
행동이 연속값이면 "가능한 모든 행동"을 나열해서 최댓값을 찾는다는 것
자체가 불가능하다 — 무한히 많은 후보를 다 계산해볼 수는 없다.

## 8.1 Q를 거치지 않고, 정책을 직접 학습한다

지금까지(Chapter 6, 7)는 \\(Q(s,a)\\)를 먼저 학습하고, 그로부터
간접적으로 정책 \\(\pi(s) = \arg\max_a Q(s,a)\\)을 얻었다.
**정책기반 방법**(Policy-Based Methods)은 이 중간 단계를 건너뛰고,
정책 \\(\pi_\theta(a|s)\\)(파라미터 \\(\theta\\)를 가진, 행동의
확률분포를 직접 내놓는 함수) 자체를 신경망으로 표현하고 직접 학습한다.
연속적인 행동공간에서도 "이 행동의 확률밀도가 얼마인가"는 잘
정의되므로, 앞의 문제가 자연스럽게 해결된다.

## 8.2 정책을 파라미터화하기

이산 행동공간에서는 정책을 신경망 출력에 softmax를 적용해 확률분포로
만든다:

\\[\pi_\theta(a|s) = \text{softmax}(f_\theta(s))_a\\]

\\(f_\theta(s)\\)는 상태 \\(s\\)를 입력받아 각 행동에 대한
원점수(logit)를 내는 신경망이다. 목표는 기대 누적 보상을 최대화하는
\\(\theta\\)를 찾는 것:

\\[J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}[R(\tau)]\\]

\\(\tau\\)는 궤적(에피소드 전체), \\(R(\tau)\\)는 그 궤적의 총
보상이다.

## 8.3 "보상을 미분한다"는 이상한 문제

문제는 "기대 보상을 최대화하는 \\(\theta\\)를 찾는다"는 목표를 그대로
미분하려고 하면 이상한 벽에 부딪힌다는 것이다: 기대 보상은 "무작위로
행동해서 얻은 결과"에 대한 평균인데, 그 무작위성 자체가 \\(\theta\\)에
의존한다. \\(J(\theta)\\)를 \\(\theta\\)로 직접 미분하려면 확률분포
\\(\pi_\theta\\) 자체를 미분해야 해서 기댓값(적분) 형태가 깨진다.

## 8.4 로그미분 트릭 (Log-Derivative Trick)

**로그미분 트릭**은 다음 항등식을 이용한다:

\\[\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta\\]

(이는 \\(\nabla \log f = \nabla f / f\\)라는 미분 공식에서 바로
나온다.) 이 치환을 통해, 미분이 확률 **밖으로** 빠져나오는 대신 로그
확률의 그래디언트로 바뀌면서, 다시 기댓값 형태로 정리할 수 있게 된다.
최종 결과가 **Policy Gradient Theorem**이다:

\\[\nabla\_\theta J(\theta) = \mathbb{E}\_\tau\left[\sum\_t \nabla\_\theta \log
\pi\_\theta(a\_t|s\_t) \\, G\_t\right]\\]

\\(G_t\\)는 시점 \\(t\\)부터의 할인된 누적 보상(Chapter 5의 return)이다.
직관: "결과가 좋았던(\\(G_t\\)가 큰) 궤적에서 실제로 골랐던 행동의
확률(\\(\log \pi_\theta(a_t|s_t)\\))을 더 높이는 방향으로, 결과가
나빴던 행동의 확률은 낮추는 방향으로 \\(\theta\\)를 옮긴다."

이 유도(대학원 수준에서도 까다롭게 여겨지는 유도)는 이번 장 연습문제의
핵심이며, 워크시트 버전으로 핵심 아이디어 하나("왜 로그를 취하면
문제가 풀리는가")만 확실히 잡는 것을 목표로 한다.

## 8.5 REINFORCE 알고리즘

Policy Gradient Theorem을 그대로 경사 상승법(gradient ascent —
최대화이므로 `+=`)으로 구현한 것이 REINFORCE다:

```python
import math

def softmax_policy(theta, state_feature):
    logits = [theta[0]*state_feature, theta[1]*state_feature]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

def reinforce_update(theta, episode, alpha, gamma):
    # episode: [(state_feature, action, reward), ...]
    T = len(episode)
    G = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = episode[t][2] + gamma * running
        G[t] = running
    for t, (s, a, r) in enumerate(episode):
        probs = softmax_policy(theta, s)
        if a == 0:
            grad_log_pi = [(1 - probs[0]) * s, -probs[1] * s]
        else:
            grad_log_pi = [-probs[0] * s, (1 - probs[1]) * s]
        theta[0] += alpha * G[t] * grad_log_pi[0]
        theta[1] += alpha * G[t] * grad_log_pi[1]
    return theta
```

## 8.6 왜 이게 model-free인가

Policy Gradient Theorem의 유도에서 핵심 단계는, 궤적의 확률
\\(P(\tau;\theta) = \prod_t \pi_\theta(a_t|s_t) \cdot
P(s_{t+1}|s_t,a_t)\\)를 로그로 바꾸면, **환경의 전이확률
\\(P(s_{t+1}|s_t,a_t)\\) 항은 \\(\theta\\)와 무관해서 미분하면
사라진다**는 것이다. 즉 최종 그래디언트 식에는 정책 \\(\pi_\theta\\)만
남고 환경 모델은 전혀 등장하지 않는다 — 환경이 어떻게 작동하는지
몰라도 정책을 학습할 수 있다는, model-free RL의 핵심 성질이 여기서
나온다.

## 8.7 Actor-Critic: 분산을 줄이는 개선

REINFORCE는 \\(G_t\\)(실제로 관찰된 누적 보상)를 그대로 쓰는데, 이건
하나의 에피소드를 샘플링한 결과라 노이즈가 크다(분산이 높다).
**Actor-Critic**은 \\(G_t\\) 대신, Chapter 5~6에서 배운 가치함수
(Critic)로 추정한 기준값을 빼서 분산을 줄인다(\\(G_t - V(s_t)\\), 이
차이를 **어드밴티지**(advantage)라 부른다) — 정책(Actor)과
가치함수(Critic)를 동시에 학습하는 구조다. 자세한 유도는 이 학기
범위를 넘어서지만, "정책 하나만 학습하는 것보다, 가치함수의 도움을
받으면 더 안정적으로 학습된다"는 아이디어는 기억해둘 만하다.

## 8.8 PPO: 한 걸음 더 안전하게 (Proximal Policy Optimization)

REINFORCE와 Actor-Critic에는 공통된 한계가 있다: 궤적 하나를 모아서
경사 상승 한 번에 쓰고 나면 그 데이터는 버려진다(온-폴리시,
on-policy) — 정책이 바뀌는 순간, 옛 데이터로 계산한 확률/그래디언트는
더 이상 정확하지 않기 때문이다. 게다가 경사 상승 스텝이 한 번이라도
너무 크면 정책이 갑자기 나쁜 방향으로 확 바뀌어서 회복이 안 되는
경우도 있다. 2017년 OpenAI가 제안한 **PPO**(Proximal Policy
Optimization, Schulman 외)는 데이터를 몇 번 더 재사용하면서도, 정책이
한 번에 너무 멀리 움직이지 못하게 막는 실용적인 해법이다.

핵심 도구는 **확률비**(probability ratio)다 — 지금 업데이트하려는
정책 \\(\pi\_\theta\\)와, 데이터를 모을 때 썼던 정책
\\(\pi\_{\theta\_{\text{old}}}\\)가 같은 행동에 부여하는 확률의 비다:

\\[r\_t(\theta) := \frac{\pi\_\theta(a\_t|s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t|s\_t)},
\quad r\_t(\theta\_{\text{old}}) = 1\\]

\\(r\_t\\)가 1보다 크면 그 행동을 전보다 더 선호하게 됐다는 뜻, 1보다
작으면 덜 선호한다는 뜻이다. 이 비율을 쓰면 "옛 정책이 모은 데이터"를
"새 정책 관점에서" 재사용할 수 있다(importance sampling):

\\[L^{IS}(\theta) = \mathbb{E}\_t[r\_t(\theta) A\_t]\\]

(\\(A\_t\\)는 8.7에서 정의한 어드밴티지, \\(A\_t := G\_t -
V(s\_t)\\)다.) 문제는 이 목적함수를 그냥 최대화하면 \\(r\_t\\)가
한없이 커지도록(같은 행동만 계속 더 밀어주도록) 학습이 폭주할 수
있다는 것이다 — 데이터는 재사용되지만, "정책이 너무 멀리 움직이지
않는다"는 보장은 어디에도 없다.

## 8.9 클리핑: 너무 멀리 못 가게 막기

PPO의 해법은 단순하다 — \\(r\_t\\)가 \\([1-\epsilon, 1+\epsilon]\\)
(보통 \\(\epsilon = 0.2\\)) 범위를 벗어나면, 그 구간 밖으로 나간
만큼의 이득을 깎아버린다:

\\[L^{CLIP}(\theta) = \mathbb{E}\_t\left[\min\left(r\_t(\theta) A\_t,
\text{clip}(r\_t(\theta), 1-\epsilon, 1+\epsilon) A\_t\right)\right]\\]

min을 쓰는 이유: clip 안 한 값과 clip한 값 중 더 "비관적인"(작은)
쪽을 목적함수로 삼아서, 정책이 한쪽으로 너무 밀려가는 것에 대한
보상을 스스로 제한한다. 직관: 어드밴티지가 양수(좋은 행동)인데
\\(r\_t\\)가 이미 \\(1+\epsilon\\)을 넘었으면, 더 밀어붙여도
그래디언트가 사라진다(이미 충분히 밀었으니 그만) — 반대로 어드밴티지가
음수(나쁜 행동)인데 \\(r\_t\\)가 \\(1-\epsilon\\) 밑으로 떨어졌으면
역시 그래디언트가 사라진다(이미 충분히 눌렀으니 그만). 그 결과 "좋은
행동을 과도하게 밀어주지도, 나쁜 행동을 과도하게 눌러버리지도 않는"
암묵적인 **신뢰 영역**(trust region)이 생긴다 — "정책이 얼마나
바뀌어도 되는지"를 명시적으로 제약하는 식을 따로 풀 필요 없이,
클리핑 한 번으로 같은 효과를 낸다.

```python
def ppo_clip_loss(ratio, advantage, epsilon=0.2):
    unclipped = ratio * advantage
    clipped = max(min(ratio, 1 + epsilon), 1 - epsilon) * advantage
    return min(unclipped, clipped)  # 목적함수(최대화 대상)의 한 샘플분
```

## 8.10 왜 PPO가 표준이 됐는가

PPO 이전에 같은 문제(너무 큰 업데이트로 인한 정책 붕괴)를 풀던
방법은 **TRPO**(Trust Region Policy Optimization, 2015)였다 — KL
divergence 제약을 명시적으로 걸고, 2차 미분(피셔 정보 행렬)과
conjugate gradient, line search까지 동원하는 무거운 최적화였다. PPO는
그 복잡한 기계장치를 클리핑 한 줄로 대체하면서도 비슷한 안정성을
얻는다 — "이론적으로 완벽한 보장"(TRPO의 단조 개선 보장) 대신
"구현하기 쉽고 실전에서 잘 작동하는 근사"를 택한 선택이다.

이 실용성 덕분에 PPO는 지금 가장 널리 쓰이는 정책기반 알고리즘이
됐다: ChatGPT류 LLM을 사람 피드백에 맞게 조정하는 **RLHF**
(Reinforcement Learning from Human Feedback) 단계, OpenAI Five(Dota
2), AlphaStar(스타크래프트 II), 로봇 보행 제어 등에 실제로 널리
쓰인다(RLHF는 Chapter 9에서 직접 다룬다).

**Q-learning이 "얼마나 좋은지 평가한 뒤 최선을 고른다"는 간접적
전략이라면, 정책기반 방법은 "무엇을 할지" 자체를 직접, 그리고
연속적인 행동 공간에서도 학습한다. Chapter 6에서 증명했듯이 유한한
MDP라면 최적 정책 \\(\pi^\*\\)는 항상 존재한다 — policy gradient와
PPO는 그 존재가 보장하는 목표를 향해 실제로 나아가는, 지금 가장
널리 쓰이는 방법이다.**

---

## 연습문제

**1. (코딩)** 간단한 이산 행동공간(2개 행동) softmax 정책에 대해, 한
에피소드의 REINFORCE 업데이트를 완성하라(핵심 줄은 빈칸으로 남겨져
있다고 가정):

```python
import math

def softmax_policy(theta, state_feature):
    # ADD ADDITIONAL CODE HERE!!
    # logits = [theta[0]*state_feature, theta[1]*state_feature]
    # softmax 확률 계산 (수치안정성을 위해 max 빼고 exp)
    return probs

def reinforce_update(theta, episode, alpha, gamma):
    # episode: [(state_feature, action, reward), ...]
    # ADD ADDITIONAL CODE HERE!!
    # 리턴(return) G_t를 뒤에서부터 누적 계산 (할인 적용)
    for t, (s, a, r) in enumerate(episode):
        probs = softmax_policy(theta, s)
        # ADD ADDITIONAL CODE HERE!!
        # grad_log_pi: action=0이면 [1-probs[0], -probs[1]]*s, action=1이면 [-probs[0], 1-probs[1]]*s
        # theta 갱신: theta += alpha * G[t] * grad_log_pi
    return theta
```

**2. (손유도, Tier C — 최우선 폴백 대상)** 정책 \\(\pi_\theta\\) 하에서
기대 리턴 \\(J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}[R(\tau)]\\)의
그래디언트가

\\[\nabla\_\theta J(\theta) = \mathbb{E}\_{\tau}\left[\sum\_t \nabla\_\theta \log
\pi\_\theta(a\_t|s\_t) \\, G\_t\right]\\]

가 됨을, 로그미분 트릭을 사용하여 유도하라. (수학 상위권/희망자
대상 심화 — 대부분은 아래 워크시트를 기본값으로 한다.)

**빈칸채움형 워크시트 버전** (기본값):

```
목표: J(theta) = sum_tau P(tau; theta) * R(tau) 를 theta로 미분하고 싶다.
문제: P(tau; theta)를 직접 미분하면 기댓값(적분) 형태가 깨져서 샘플로 추정할 수 없다.

로그미분 트릭: grad(f) = f * grad(log f)

Step 1: grad_theta P(tau;theta) = P(tau;theta) * ______________  [로그미분 트릭 적용]

Step 2: grad_theta J(theta) = sum_tau ______________ * R(tau)
                             = E_tau[ grad_theta log P(tau;theta) * R(tau) ]

Step 3: 궤적 확률 P(tau;theta) = prod_t pi_theta(a_t|s_t) * (환경전이확률, theta와 무관)
        따라서 grad_theta log P(tau;theta) = ______________

결론: grad_theta J(theta) = E_tau[ (sum_t grad_theta log pi_theta(a_t|s_t)) * R(tau) ]
```

**정확성 확인**: Step 3의 "환경전이확률은 theta와 무관하다"는 사실이
왜 중요한지 한 문장으로 설명하라(힌트: 이게 없으면 환경 모델을
몰라도 정책만으로 학습 가능하다는 model-free RL의 핵심 성질이
깨진다).
