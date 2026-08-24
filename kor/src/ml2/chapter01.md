# Chapter 1. 코스 소개와 신경망 미니 리뷰 (Course Introduction & Neural Network Mini-Review)

체스나 바둑을 둘 때, "이 수가 정답이다"라고 알려주는 사람은 아무도 없다 —
게임이 끝나야 "이겼다/졌다"는 신호 하나만 돌아온다. 지금까지의 지도학습(정답
\\(y\\)가 있는 데이터)과 비지도학습(구조만 있는 데이터)와는 전혀 다른 이
문제 — **행동하고, 그 결과로 나중에 돌아오는 보상만 보고 스스로 좋은 행동을
찾아내야 하는 문제** — 가 이번 학기의 주제인 **강화학습**[^suttonbarto] (Reinforcement
Learning, RL)이다. 그리고 이번 학기는 그 강화학습을, 화면 속 게임이 아니라
**물리 법칙이 지배하는 로봇 시뮬레이션**이라는 무대 위에서 다룬다.

## 이 챕터가 이 책의 출발점인 이유

이 챕터는 ML2의 첫 장으로, ML1과 ML2를 잇는 경계 자체를 다룬다 — ML1이
"주어진 데이터를 어떻게 모델로 변환하는가"였다면, ML2는 "행동하며 데이터를
스스로 만들어내는 에이전트를 어떻게 학습시키는가"를 다루며, 이 장은 그 경계에서
강화학습의 정체성(보상으로 배우는 학습 + 로봇 시뮬레이션 무대)과 앞으로 16개
챕터의 공통 언어(에이전트·환경·정책, 그리고 그 뒤를 받치는 신경망 도구)를
정해준다.

다음 장 Chapter 2("멀티암 밴딧 (Multi-Armed Bandits)")로 이어지는 다리도
여기서 놓인다 — 밴딧은 상태도, 여러 스텝에 걸친 전이도 없이 "탐험과 활용의
딜레마"만 순수하게 남긴 가장 단순한 RL 문제이므로, 이 장에서 보상·정책이라는
어휘와 Gymnasium 실습 환경을 먼저 장착해두면 2장에서
\\(\\varepsilon\\)-greedy와 UCB[^ucb]로 그 딜레마를 처음 풀 때 마찰 없이 들어갈 수
있다.

## 학습 목표

이 챕터를 마치면 다음과 같은 일이 가능해진다:

- 지도학습과 강화학습을 "데이터를 누가 만드느냐", "신호가 정답 \\(y\\)인가
  점수(보상)인가", "지금의 행동이 미래를 바꾸는가"의 세 축으로 구분하고,
  "탐험 vs 활용"이라는 강화학습만의 딜레마가 왜 근본적으로 피할 수 없는지
  설명할 수 있다.
- "순전파 → 손실 → 역전파 → 업데이트"라는 신경망 학습의 뼈대를, 활성
  함수가 없으면 두 층이 하나의 선형 변환으로 축소된다는 XOR의 예시와
  1-2-1 신경망의 정수 손계산까지 포함해 정확히 짚을 수 있다.
- "강화학습의 새로움은 학습 알고리즘이 아니라 손실함수의 정의에 있다"는
  관점에서, DQN[^dqn] (Ch. 9)이나 PPO[^ppo] (Ch. 10~11)에서도
  PyTorch[^pytorch]의 `zero_grad()` → `backward()` → `step()` 세 줄이
  그대로 쓰이는 이유를 미리 설명할 수 있다.
- Gymnasium[^gym][^gymnasium]의 `reset`/`step` 표준 계약과 관측·행동 공간(`Box`/`Discrete`),
  `terminated`와 `truncated`의 차이를 읽고, CartPole에서 무작위 정책의
  평균 생존 스텝을 시드 고정으로 재현성 있게 측정할 수 있다.

## 세 수업 블록의 흐름

- [1.1 강화학습이란 무엇인가, 그리고 이번 학기 로드맵](chapter01/1.md) —
  ML1의 "데이터의 세 갈래"에서 "보상이 있다"는 갈래를 꺼내, 지도학습과의
  차이를 네 행짜리 표(데이터를 스스로 만든다 / 할인된 리턴을 최대화한다 /
  지금의 행동이 미래를 바꾼다 / 탐험 vs 활용)로 정리하고, 1951년 쥐 실험
  특허부터 DQN·PPO·AlphaGo[^alphago]·RLHF[^rlhf]까지의 역사를 "이번 학기의 어느 장에
  나오나"로 이어 붙인다. "왜 하필 로봇 시뮬레이션인가"(연속 상태·행동이라
  표로는 안 된다)는 질문으로 Block A(Ch. 2~8, 표 기반의 완성)와 Block B
  (Ch. 9~16, 신경망 + 로봇)의 전체 로드맵을 흐름도로 보여준다.

![DQN의 합성곱 신경망 구조 -- Atari 화면을 입력받아 조이스틱 행동별 Q값을 출력한다 (원 논문 Extended Data Figure 1).](../images/ref_dqn.png)

![서로게이트 함수 L\_CLIP의 한 항(단일 timestep)을 확률비 r의 함수로 그린 그래프 — 왼쪽은 이익이 양수(A>0), 오른쪽은 음수(A<0)인 경우. (원 논문 Figure 1)](../images/ref_ppo.png)

![정책망·가치망 학습 파이프라인(왼쪽)과 두 신경망의 합성곱 구조(오른쪽) (원 논문 Figure 1).](../images/ref_alphago.png)

![Figure 1: RLHF 접근법의 구조를 보여주는 개략도 — 보상 예측기(reward predictor)가 트래젝터리 세그먼트의 인간 비교 피드백으로 비동기 학습된 뒤, 그 보상을 이용해 정책(policy)를 강화학습으로 학습하는 전체 파이프라인을 보여준다.](../images/ref_rlhf.png)

- [1.2 신경망 미니 리뷰](chapter01/2.md) — 강화학습을 새로 가르치지 않고,
  앞으로 16개 챕터 내내 쓸 신경망 뼈대를 한 번 더 정확히 짚는 "열쇠
  정리" 시간이다. 활성 함수 없이 두 층이 하나의 선형 변환으로 축소되는
  등식과 XOR 결정 경계로 비선형성의 필요성을 보이고, 1-2-1 신경망 역전파
  의 정수 손계산과 PyTorch 실습(\\(y=2x+1\\), \\(y=\\sin(x)\\) 근사)을
  거쳐, "정답을 보상 기반으로 바꾸는 것(정책 \\(\\pi(a|s)\\)이나
  Q-네트워크)이 전부일 뿐, 학습의 기계 자체는 변하지 않는다"는 한 문장으로
  마무리한다.[^cs230]
- [1.3 실습 환경 세팅: Gymnasium](chapter01/3.md) — 이번 학기 모든 실습의
  "무대"를 세팅한다. `gym.make`와 `reset`/`step` 표준 계약(다음 상태·보상·
  종료 신호를 포함한 5-튜플), CartPole의 4차원 `Box` 관측 공간(위치+속도)과
  이산 2행동 `Discrete(2)` — Pendulum의 연속 `Box`와 대비 — 과
  `terminated`와 `truncated`를 가치 업데이트에서 왜 구분해야 하는지를 다룬
  뒤, 200개 시드의 무작위 정책 실험("평균 24.1스텝")으로 베이스라인을
  수치로 잡아두는데, 이 숫자가 이후 \\(\\varepsilon\\)-greedy(Ch. 2)와
  DQN(Ch. 9)의 성과를 같은 기준으로 비교하는 잣대가 된다.

[^suttonbarto]: Sutton, R. S., Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.). MIT Press. 저자 공식 무료 공개: http://incompleteideas.net/book/the-book-2nd.html
[^dqn]: Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529–533. (Earlier preprint: Mnih, V. et al. (2013). arXiv:1312.5602.)
[^ppo]: Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
[^gym]: Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., Zaremba, W. (2016). "OpenAI Gym." arXiv:1606.01540.
[^alphago]: Silver, D. et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature 529, 484–489.
[^rlhf]: Christiano, P. et al. (2017). "Deep reinforcement learning from human preferences." NeurIPS 2017. arXiv:1706.03741.
[^pytorch]: Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019. arXiv:1912.01703.
[^gymnasium]: Towers, M., Kwiatkowski, A., Terry, J., et al. (2024). "Gymnasium: A Standard Interface for Reinforcement Learning Environments." arXiv:2407.17032.
[^cs230]: Stanford CS230: Deep Learning. https://cs230.stanford.edu/
[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/
[^ucb]: Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). "Using Confidence Bounds for Exploitation-Exploration Trade-off." JMLR 3, 397–422. (COLT 2002.)

더 깊이 보려면: [^cs234]
