# Chapter 9. 함수근사와 DQN (Function Approximation & Deep Q-Networks)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter09_dqn_tricks.ipynb)

2013년, 딥마인드는 Q-learning의 Q-테이블을 신경망으로 통째로 바꾼
알고리즘으로 Atari 2600 게임 여러 개를 학습시켰다 — 게임의 규칙을 전혀
알려주지 않고, 오직 화면 픽셀과 점수만 보고서. **DQN**(Deep
Q-Network)[^dqn]이라 불린 이 결과는 2015년 Nature에 게재되며, 여러 게임에서
사람 수준 또는 그 이상의 성능을 보였다. 표(tabular)로 배웠던 강화학습이
여기서 "배운 것을 한 번도 본 적 없는 상태에도 퍼뜨리는"
함수근사[^suttonbarto]와 만나면서, 실용적인 규모로 첫발을 내딛는 순간이다.

**이전 챕터와의 연결.** Chapter 8의 캡스톤에서 밴딧부터 n-step까지
표 기반 강화학습의 이론 전체를 Gymnasium[^gymnasium]의 작은 격자 환경에서 직접
구현해 보았다. 그런데 그 모든 알고리즘은 "상태-행동 쌍을 표에
하나하나 적어 넣는다"는 전제 위에 서 있었다. 이 챕터는 바로 그 전제를
버린다 — 상태가 너무 많거나 연속일 때 표로는 원리적으로 불가능하다는
한계를 직면하고, Q-함수를 함수(신경망)로 표현하는 길을 연다.

**다음 챕터와의 연결.** DQN은 Q-함수를 잘 근사할 수 있게 해주지만,
매 스텝 \\(\max\_{a'} Q(s',a')\\)를 계산하는 구조라는 점은 그대로다.
행동이 연속량이면 이 최댓값을 취하는 것 자체가 불가능해진다.
Chapter 10에서는 이 문제를 정면으로 마주하며, Q-함수를 우회하고
정책을 직접 학습하는 **정책기반 강화학습**(policy-based RL)으로
건너든다. 이 주제를 더 깊이 다루는 자료: [^cs234]

## 학습 목표

- 이 챕터를 마치면, Atari 화면이 만들 수 있는 상태(\\(\approx
  10^{16{,}993}\\)개)가 표로는 원리적으로도 저장 불가능하다는 사실을
  숫자로 설명하고, 함수근사의 진짜 목적이 "메모리"가 아니라
  **일반화**(만나지 못한 비슷한 상태에 배운 것을 퍼뜨리는 것)임을
  설명할 수 있다.
- 이 챕터를 마치면, DQN 손실함수가 "정답이 매 스텝 스스로 바뀌는"
  지도학습 회귀 문제라는 점과, **타겟 네트워크**가 움직이는 과녁을
  어떻게 고정하는지(목표값 계산에만 쓰는 두 번째 네트워크를 일정 주기
  마다 통째로 복사) 코드 수준에서 설명할 수 있다.
- 이 챕터를 마치면, 연속 경험이 서로 상관되어 미니배치 경사추정치의
  분산을 키운다는 두 번째 불안정 원인을 설명하고, **경험 재현**(replay
  buffer에서 무작위 미니배치를 샘플링)이 상관 제거와 데이터 재사용의
  두 이득을 동시에 주는 이유를 설명할 수 있다.
- 이 챕터를 마치면, 두 장치를 하나로 조립한 DQN 알고리즘을 CartPole에서
  돌려 실제 **학습곡선**[^cartpole]을 그리고, 버퍼가 차기 전에 학습을 시작하거나
  단일 시드로만 판단하는 실수를 피하며 여러 시드의 곡선을 함께 읽을
  수 있다.

세 수업 블록은 "한 가지씩 불안정 원인을 찾아 고친다"는 흐름을 이룬다.
**9.1**에서 표가 불가능한 이유와 함수근사의 핵심(일반화)을 잡고
Q-신경망의 손실함수를 세운 뒤, **9.2**에서 "목표가 움직인다"는 첫
불안정 원인에 타겟 네트워크를, **9.3**에서 "데이터가 상관된다"는 두
번째 원인에 경험 재현을 붙인다. 9.3의 마지막에서 이 모든 조각을
CartPole의 진짜 학습곡선으로 합친다.

## 이번 주 수업 블록

이번 주는 세 개의 수업 블록으로 진행된다:

- [9.1 Q-테이블의 한계와 함수근사](chapter09/1.md)
  Q-테이블이 Atari 화면(\\(256^{84\times84}\approx10^{16{,}993}\\))에는
  담기지도, 대부분의 상태를 방문하기도 어렵다는 한계를 숫자로 보여주고,
  그 대안으로 Q-함수를 신경망 \\(Q(s,a;\theta)\\)로 근사한다. 핵심은
  "함수근사의 진짜 이유는 일반화"라는 주장이며, 같은 데이터로 배운
  Q-테이블(칸마다 점프)과 신경망(보간된 매끄러운 곡선)을 1차원
  랜덤워크 장난감 환경에서 비교한다.
- [9.2 DQN의 안정화 장치와 실습](chapter09/2.md)
  "목표값이 매 스텝 스스로 바뀐다"는 첫 불안정 원인을 **타겟
  네트워크**(목표 계산에만 쓰고, 일정 주기마다 온라인 네트워크를
  통째로 복사하는 두 번째 네트워크)로 해결한다. `dqn_loss` 코드에서
  \\(\theta\\)와 \\(\theta^-\\)가 들어가는 두 자리를 분리해서 읽고,
  움직이는 목표가 학습을 실제로 얼마나 변질시키는지 실험한 뒤, DQN
  알고리즘 전체를 코드 한 루프로 조립한다.
- [9.3 경험 재현과 CartPole 학습곡선 실습](chapter09/3.md)
  "연속 경험이 서로 상관돼 있다"는 두 번째 원인을 **경험 재현**(replay
  buffer에서 무작위 미니배치 샘플링)으로 풀고, 상관 제거와 데이터
  재사용의 두 이득을 정리한다. 이어 9.1·9.2의 조각(`QNetwork`,
  `dqn_loss`+타겟 네트워크, `ReplayBuffer`)을 CartPole에서 합쳐
  진짜 학습곡선을 그리며, 버퍼 크기와 다중 시드 같은 실무 포인트까지
  다룬다.

[^dqn]: Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529–533. (Earlier preprint: Mnih, V. et al. (2013). arXiv:1312.5602.)
[^suttonbarto]: Sutton, R. S., Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.). MIT Press. 저자 공식 무료 공개: http://incompleteideas.net/book/the-book-2nd.html — 함수근사(function approximation)로 표 기반 강화학습을 확장하는 구도는 이 표준 교과서 8장 "Function Approximation and Generalization"에서 체계적으로 다뤄진다.
[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/
[^gymnasium]: Towers, M. et al. (2024). "Gymnasium: A Standard Interface for Reinforcement Learning Environments." arXiv:2407.17032.
[^cartpole]: Brockman, G. et al. (2016). "OpenAI Gym." arXiv:1606.01540. (CartPole은 이 라이브러리의 환경)
