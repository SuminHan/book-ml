# Topics Covered

## 정책을 파라미터화하기

이산 행동공간에서는 정책을 신경망 출력에 softmax를 적용해 확률분포로 만든다:

\\[\pi_\theta(a|s) = \text{softmax}(f_\theta(s))_a\\]

\\(f_\theta(s)\\)는 상태 \\(s\\)를 입력받아 각 행동에 대한 원점수(logit)를 내는
신경망이다. 목표는 기대 누적 보상을 최대화하는 \\(\theta\\)를 찾는 것:

\\[J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]\\]

\\(\tau\\)는 궤적(에피소드 전체), \\(R(\tau)\\)는 그 궤적의 총 보상이다.

## 로그미분 트릭(Log-Derivative Trick)

\\(J(\theta)\\)를 \\(\theta\\)로 직접 미분하려면 확률분포 \\(\pi_\theta\\) 자체를
미분해야 해서 기댓값(적분) 형태가 깨진다. **로그미분 트릭**은 다음 항등식을
이용한다:

\\[\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta\\]

(이는 \\(\nabla \log f = \nabla f / f\\)라는 미분 공식에서 바로 나온다.) 이
치환을 통해, 미분이 확률 **밖으로** 빠져나오는 대신 로그 확률의 그래디언트로
바뀌면서, 다시 기댓값 형태로 정리할 수 있게 된다. 최종 결과가
**Policy Gradient Theorem**이다:

\\[\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_t \nabla_\theta \log
\pi_\theta(a_t|s_t) \, G_t\right]\\]

\\(G_t\\)는 시점 \\(t\\)부터의 할인된 누적 보상(W05의 return)이다. 직관: "결과가
좋았던(\\(G_t\\)가 큰) 궤적에서 실제로 골랐던 행동의 확률(\\(\log
\pi_\theta(a_t|s_t)\\))을 더 높이는 방향으로, 결과가 나빴던 행동의 확률은 낮추는
방향으로 \\(\theta\\)를 옮긴다."

## REINFORCE 알고리즘

Policy Gradient Theorem을 그대로 경사 상승법(gradient ascent — 최대화이므로
`+=`)으로 구현한 것이 REINFORCE다:

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

## 왜 이게 model-free 인가

Policy Gradient Theorem의 유도에서 핵심 단계는, 궤적의 확률
\\(P(\tau;\theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t,a_t)\\)를
로그로 바꾸면, **환경의 전이확률 \\(P(s_{t+1}|s_t,a_t)\\) 항은 \\(\theta\\)와
무관해서 미분하면 사라진다**는 것이다. 즉 최종 그래디언트 식에는 정책
\\(\pi_\theta\\)만 남고 환경 모델은 전혀 등장하지 않는다 — 환경이 어떻게
작동하는지 몰라도 정책을 학습할 수 있다는, model-free RL의 핵심 성질이 여기서
나온다.

## Actor-Critic: 분산을 줄이는 개선

REINFORCE는 \\(G_t\\)(실제로 관찰된 누적 보상)를 그대로 쓰는데, 이건 하나의
에피소드를 샘플링한 결과라 노이즈가 크다(분산이 높다). **Actor-Critic**은
\\(G_t\\) 대신, W05~W06에서 배운 가치함수(Critic)로 추정한 기준값을 빼서 분산을
줄인다(\\(G_t - V(s_t)\\), 이 차이를 **어드밴티지**(advantage)라 부른다) — 정책
(Actor)과 가치함수(Critic)를 동시에 학습하는 구조다. 자세한 유도는 이 학기
범위를 넘어서지만, "정책 하나만 학습하는 것보다, 가치함수의 도움을 받으면 더
안정적으로 학습된다"는 아이디어는 기억해둘 만하다.
