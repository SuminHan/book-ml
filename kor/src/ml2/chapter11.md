# Chapter 11. 고급 정책 최적화: PPO (Proximal Policy Optimization)

Chapter 10의 REINFORCE와 Actor-Critic에는 공통된 한계가 있다: 궤적 하나를
모아서 경사 상승 한 번에 쓰고 나면 그 데이터는 버려진다(온-폴리시,
on-policy) — 정책이 바뀌는 순간, 옛 데이터로 계산한 확률/그래디언트는
더 이상 정확하지 않기 때문이다. 게다가 경사 상승 스텝이 한 번이라도
너무 크면 정책이 갑자기 나쁜 방향으로 확 바뀌어서 회복이 안 되는
경우도 있다 — 로봇 제어처럼 "한 번의 잘못된 업데이트로 정책이 완전히
무너지면 다시 좋아지기 어려운" 상황에서는 치명적이다. 2017년 OpenAI가
제안한 **PPO**[^ppo] (Proximal Policy Optimization, Schulman 외)는 데이터를
몇 번 더 재사용하면서도, 정책이 한 번에 너무 멀리 움직이지 못하게 막는
실용적인 해법이다.

![서로게이트 함수 L_CLIP의 한 항(단일 timestep)을 확률비 r의 함수로 그린 그래프 — 왼쪽은 이익이 양수(A>0), 오른쪽은 음수(A<0)인 경우. (원 논문 Figure 1)](../images/ref_ppo.png)

이 장은 Chapter 10이 남긴 두 공백을 정확히 메우는 장이다. Chapter 10에서
연속 행동공간을 위한 정책 그래디언트 기계(정책 그래디언트 정리,
REINFORCE, Actor-Critic)를 세웠지만, 모은 데이터는 "한 번 쓰고 버리는"
것이었고 업데이트 스텝의 크기는 "무제한"이었다. 이 장은 **신뢰 영역**과
**확률비**라는 두 도구로 이 두 문제를 한 번에 풀고, 그것이 클리핑이라는
한 줄의 식으로 압축되는 과정을 따라간다. 다음 장(Chapter 12, 모방학습과
인간 피드백)에서는 보상의 시행착오 대신 사람의 시연과 선호로부터 배우는
방법으로 방향을 돌린다 — Chapter 2~11의 보상 최적화 알고리즘 중 가장
강한 것인 PPO가, Chapter 12에서 "왜 보상 최적화만으론 안 되는가"를
비교하기 위한 기준점이 된다.

## 학습 목표

- 이 챕터를 마치면 "신뢰 영역"의 뜻(한 번의 업데이트에서 정책이 움직일
  수 있는 크기를 KL 발산으로 잰 "믿을 수 있는 범위"로 제한하는 것)을
  설명할 수 있고, 중요도 샘플링 목적함수
  \\(L^{IS}(\theta) = \mathbb{E}\_t[r\_t(\theta) A\_t]\\)의 그래디언트가
  \\(r\_t = 1\\)에서는 평범한 정책 그래디언트와 정확히 같아지는 것을
  유도할 수 있다.
- 이 챕터를 마치면 PPO 클리핑 목적함수 \\(L^{CLIP}\\)의 모양을
  손으로 그려서 설명할 수 있다 — min이 만드는 "모서리"가, 좋은 행동을
  더 밀어주거나 나쁜 행동을 더 억제하는 **개선 방향에만** 그래디언트를
  0으로 만들고 악화 방향에는 발동하지 않는 비대칭을 어떻게 만들어내는지.
- 이 챕터를 마치면 GAE(Generalized Advantage Estimation[^gae], TD 오차를
  지수적으로 가중합해서 TD와 MC 사이를 \\(\lambda\\) 다이얼로 절충하는
  어드밴티지 추정)와 엔트로피 보너스가 탐험 유지에 왜 필요한지 설명할
  수 있다.
- 이 챕터를 마치면 PPO 한 반복의 전체 데이터 흐름(경험 수집 → GAE →
  미니배치 정규화 → M 에폭 미니배치 재사용 →
  \\(\theta\_{\text{old}}\\) 갱신)을 따라갈 수 있고, 연속 제어
  환경(Pendulum)에서 PPO를 학습시킬 수 있다.

![3D 워커 환경 학습 곡선 (원 논문 Figure 1) — PPO+GAE(ours, 청색)가 DDPG/TRPO/SAC 대비 더 적은 반복 횟수로 높은 보상을 달성함을 보여준다.](../images/ref_gae.png) [^ddpg][^sac]

이번 주는 세 개의 수업 블록으로 진행된다:

- [11.1 신뢰 영역과 확률비](chapter11/1.md) — 11.2의 클리핑으로 합쳐질
  "재료"를 준비하는 절이다. 신뢰 영역(정책의 한 번 움직임을 KL 발산으로
  잰 "믿을 수 있는 범위" 안으로 제한하는 직관)과 확률비
  \\(r\_t = \pi\_\theta / \pi\_{\text{old}}\\)를 소개하고, 확률비로
  목적함수를 다시 쓰면 옛 정책이 모은 데이터를 중요도 샘플링으로
  재사용할 수 있으며 그 그래디언트가 정책 그래디언트에 계수
  \\(r\_t\\)가 곱해진 형태라는 것을 유도한다. 2-행동 softmax 정책으로
  \\(r\_t\\)가 실제로 어떻게 움직이는지 손으로 계산해본다.
- [11.2 PPO 클리핑과 연속 제어 실습](chapter11/2.md) — 신뢰 영역 제약식을
  목적함수에 직접 박아 넣는 한 줄,
  \\(\min(r\_t A\_t,\ \text{clip}(r\_t, 1-\epsilon, 1+\epsilon)A\_t)\\),
  이 만들어내는 목적함수의 모양을 손으로 확인한다. \\(A\_t > 0\\)이면
  \\(r\_t > 1+\epsilon\\) 이후, \\(A\_t < 0\\)이면 \\(r\_t < 1-\epsilon\\)
  이후 "추가 이득"이 0이 되는 min의 비대칭을 그림과 함께 짚는다.
  이어서 Pendulum 환경(행동: -2~+2Nm 톱의 하나)의 연속 행동공간에서
  PPO를 통째로 구현해본다 — 가우시안 정책, 그리고 정책 손실 + 가치
  손실 − 엔트로피 보너스로 이루어진 전체 목적함수까지.
- [11.3 GAE와 PPO가 표준이 된 이유](chapter11/3.md) — PPO를 실전용으로
  만든 마지막 두 조각인 GAE와 엔트로피 보너스를 배운다. GAE가
  "여러 시점의 TD 오차를 적격흔적처럼 지수적으로 가중합"하는 것임을
  정리로 보이고, 3스텝 에피소드에서 \\(\lambda\\)만 바꾸면 어드밴티지
  추정치의 부호까지 달라지는 것을 손으로 계산한다. TRPO[^trpo]보다 이론적
  보장은 약하지만 구현이 훨씬 단순한 PPO가 왜 실제로 TRPO를 밀어내고
  표준이 됐는지 정리하고, "PPO를 돌린다"는 것이 실제로 어떤 코드
  묶음(경험 수집 → GAE → 정규화 → M 에폭 →
  \\(\theta\_{\text{old}}\\) 갱신)인지 전체 그림으로 마무리한다.[^cs234]

![TRPO의 데이터 생성 절차 (원 논문 Figure 1) — 왼쪽은 시뮬레이션한 단일 트래젝터리의 모든 상태-행동 쌍을 목적 함수에 사용하는 single path 절차, 오른쪽은 줄기(trunk) 트래젝터리에 도달 상태의 일부에서 가지(branch) 롤아웃을 퍼지는 방식(공통 무작위 수)으로 분산 감소 효과를 얻는 vine 절차를 보여준다.](../images/ref_trpo.png)

[^ppo]: Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
[^gae]: Schulman, J. et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438.
[^trpo]: Schulman, J. et al. (2015). "Trust Region Policy Optimization." ICML 2015. arXiv:1502.05477.
[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/ — 이 장의 주제(신뢰 영역/근접 정책 최적화, PPO, GAE)를 더 깊이 다루는 자료.
[^ddpg]: Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015). "Continuous Control with Deep Reinforcement Learning." arXiv:1509.02971.
[^sac]: Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." arXiv:1801.01290.
