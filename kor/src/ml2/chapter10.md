# Chapter 10. 정책기반 강화학습 (Policy-Based Reinforcement Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter10_reinforce_ppo.ipynb)

로봇 팔의 관절에 가할 힘을 정하는 문제를 생각해보자. 이 "행동"은
-10Nm부터 +10Nm까지 **연속적인 값** 중 아무거나 될 수 있다. Chapter
6·9의 Q-learning/DQN[^dqn]은 매 스텝 \\(\\max\_{a'} Q(s',a')\\)를 계산해야
하는데, 행동이 연속값이면 "가능한 모든 행동"을 나열해서 최댓값을 찾는다는
것 자체가 불가능하다 — 무한히 많은 후보를 다 계산해볼 수는 없다. 이번
장부터 다룰 로봇 시뮬레이션(Chapter 13~14)이 바로 이런 연속 행동공간을
쓰므로, 이 문제를 정면으로 풀어야 한다.

Chapter 9까지의 답 — Q를 신경망으로 배우고 그 argmax에서 정책을 뽑는
"가치기반" 경로 — 는 행동을 나열할 수 있는 공간에서는 훌륭했지만,
연속 공간에서는 그 argmax 자체가 불가능해진다. 이번 장은 "각 행동에
점수를 매긴 뒤 가장 좋은 것을 고른다"는 중간 단계를 건너뛰고, 행동
확률분포 \\(\\pi\_\theta(a|s)\\) 자체를 신경망으로 표현해 **정책을
직접 학습**한다. 최댓값을 "찾는" 대신 분포에서 행동을 **샘플링**하고,
기대 누적 보상을 높이는 방향으로 파라미터 \\(\\theta\\)를 움직이는
것이다. 그리고 이 장에서 세울 REINFORCE/Actor-Critic의 기초 위에
Chapter 11에서는 정책이 한 업데이트 만에 무너지지 않게 막는 PPO[^ppo]를
세운다.

## 학습 목표

이 챕터를 마치면 다음을 할 수 있다.

- DQN형 가치기반 방법이 연속 행동공간에서 바로 쓸 수 없는 이유를 설명하고, "확률분포에서 행동을 샘플링하는" 확률적 정책이 답인 이유를 설명할 수 있다.
- **로그미분 트릭**(log-derivative trick)이 Policy Gradient Theorem[^suttonbarto]에서 하는 일을 — "무작위 결과인데도 어떻게 미분하는가"를 — 설명할 수 있다.
- REINFORCE를 손으로 한 스텝 업데이트하고, PyTorch[^pytorch]로 CartPole을[^cartpole] 학습시키며, softmax 정책 그래디언트의 "one-vs-rest" 모양을 설명할 수 있다.
- REINFORCE의 학습 신호가 왜 높은 분산인지 진단하고, Actor-Critic이 **베이스라인/어드밴티지**(가치함수를 빼는 것)로 분산을 줄이는 아이디어를 설명할 수 있다.

## 이번 주 수업 블록

이번 주는 세 개의 수업 블록으로 진행된다:

- [10.1 정책 그래디언트 정리](chapter10/1.md) — Q를 거치지 않고 정책을 직접 파라미터화한다(이산 공간은 softmax, 연속 공간은 상태 조건부 정규분포). 로그미분 트릭으로 Policy Gradient Theorem에 도달하고, 탐색을 학습 목표에 넣는 엔트로피 보너스와 이 정리의 분산 문제[^gae]까지 짚는다.
- [10.2 REINFORCE와 실습](chapter10/2.md) — 정리를 알고리즘으로 번역한 REINFORCE를 손으로 한 스텝 갱신하고 2행동 밴딧 실험[^silvercourse][^gymnasium]으로 "좋은 행동의 확률이 오른다"를 확인한 뒤, PyTorch 실습에서 CartPole을 학습시키고 "리턴을 정규화하지 않고 그대로 쓰는" 함정을 다룬다.
- [10.3 Actor-Critic](chapter10/3.md) — 시드마다 다른 학습 곡선이라는 REINFORCE의 높은 분산 문제를 진단하고, 행동에 무관한 베이스라인(가치함수)을 빼도 그래디언트 기댓값은 안 바뀔 때 분산만 줄어든다는 성질을 숫자로 검증한다. 실습에서는 CartPole에서 REINFORCE와 A2C[^a3c]의 학습 곡선을 비교한다.

## 세 블록이 이어지는 하나의 라인

세 블록은 하나의 줄을 이룬다. 10.1이 정리를(그래디언트 방향을), 10.2가 그 정리를 그대로 실행하는 알고리즘을(그리고 거친 학습 곡선을) 주고, 10.3은 그 거칠기(분산)를 고친다. 이번 주를 마치면 손에 쥐게 되는 두 네트워크 — Actor와 Critic — 는 Chapter 11의 PPO가 그대로 재사용하는 구성 요소이고, Chapter 13~14의 로봇 시뮬레이션(Pendulum)은 그 위에 올라간다. 즉 10.1의 정규분포 정책 파라미터화, 10.2의 REINFORCE 갱신 식, 10.3의 어드밴티지는 한 주의 소재가 아니라, 학기 마지막까지 계속 만지는 부품이다. 이 주제(정책 그래디언트에서 PPO/Actor-Critic까지)를 더 깊이 다루는 자료로 스탠포드 CS234: Reinforcement Learning을 추천한다.[^cs234]

[^dqn]: Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529–533. (Earlier preprint: Mnih, V. et al. (2013). arXiv:1312.5602.)
[^ppo]: Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
[^suttonbarto]: Sutton, R. S., Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.). MIT Press. 저자 공식 무료 공개: http://incompleteideas.net/book/the-book-2nd.html
[^a3c]: Mnih, V. et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." arXiv:1602.01783.
[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/
[^cartpole]: Brockman, G. et al. (2016). "OpenAI Gym." arXiv:1606.01540. (CartPole은 이 라이브러리의 환경)
[^pytorch]: Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019. arXiv:1912.01703.
[^silvercourse]: Silver, D. (2015). "UCL Course on Reinforcement Learning," Advanced Topics (COMPM050/COMPGI13) — 10개 강의 슬라이드(PDF) 및 영상 강의가 공개되어 있다. https://www.davidsilver.uk/teaching/ (본 장과 가장 직접적으로 겹치는 Lecture 9: Exploration and Exploitation — 47쪽, ε-greedy·멀티암 밴딧·컨텍스트 밴딧: https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-9-exploration-and-exploitation.pdf, CC-BY-NC 4.0).
[^gae]: Schulman, J. et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438. (어드밴티지 함수 추정·그 분산 trade-off를 정립한 논문. 어드밴티지(이익) 추정 개념을 다룬 Chapter 11에서 PPO와 함께 직접 사용된다.)
[^gymnasium]: Towers, M. et al. (2024). "Gymnasium: A Standard Interface for Reinforcement Learning Environments." arXiv:2407.17032.
