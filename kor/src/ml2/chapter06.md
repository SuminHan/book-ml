# Chapter 6. 시간차 학습 (Temporal-Difference Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter06_q_learning.ipynb)

1989년, 크리스 왓킨스(Chris Watkins)는 박사 학위 논문에서 **Q-learning**
이라는 알고리즘을 제안했다. Chapter 5의 몬테카를로는 모델 없이도 배울 수
있었지만, 에피소드가 끝날 때까지 기다려야 리턴을 계산할 수 있었다.
**시간차 학습**(Temporal-Difference, TD)은 이 기다림 자체를 없앤다 — 한
스텝만 진행해보고, "지금 추정치"와 "방금 관찰한 보상 + 다음 상태의
추정치"의 차이만큼 즉시 갱신한다. 모델도 필요 없고, 에피소드가 끝나기를
기다릴 필요도 없는, 강화학습에서 가장 널리 쓰이는 절충안이다[^suttonbarto].

**이전 챕터와의 연결.** Chapter 5의 몬테카를로(MC)는 "모델이 필요
없다"는 장점을 확보했지만, 에피소드가 끝날 때까지 실제 리턴을 모아야
한다는 대가를 치렀다. 이 챕터의 시간차 학습은 바로 그 대가 — 기다림 — 를
없앤다. MC가 "에피소드 끝까지의 실제 보상"을 목표로 쓴다면, TD는
"한 스텝의 실제 보상 + 다음 상태의 현재 추정치"를 목표로 매 스텝 즉시
갱신한다. 아직 정확하지 않은 추정치로 스스로를 갱신한다는 이 방식을
**부트스트래핑**(bootstrapping)이라 부른다.

**다음 챕터와의 연결.** TD(0)는 한 스텝만 보고 부트스트래핑하고, MC는
에피소드 끝까지 본다. 이 둘은 사실 "몇 스텝을 실제로 볼 것인가"라는 하나의
다이얼 위의 양 끝일 뿐이다. Chapter 7에서는 이 다이얼의 모든 지점을 아우르는
n-step 부트스트래핑과 적격흔적(eligibility traces)을 다루고, 경험뿐 아니라
학습된 모델까지 함께 쓰는 계획(planning, Dyna-Q)으로 나아간다. 이 주제를 더 깊이 다루는 자료: [^cs234]

## 학습 목표

- 이 챕터를 마치면, TD(0) 갱신식과 TD 오차를 이해하고 7-state random
  walk에서 손으로 한 에피소드를 따라가며, MC(불편하지만 분산 큼)와
  TD(편향 있되 분산 작음)의 트레이드오프를 설명할 수 있다.
- 이 챕터를 마치면, 행동을 고르려면 \\(V(s)\\)만으론 부족하고
  \\(Q(s,a)\\)가 필요한 이유를 설명하고, 실제로 고른 다음 행동을
  쓰는 on-policy 알고리즘 SARSA를 코드로 구현할 수 있다.
- 이 챕터를 마치면, 다음 상태의 \\(Q(s',a')\\)를
  \\(\max\_{a'}Q(s',a')\\)로 바꾼 한 줄의 차이가 on-policy와
  off-policy를 가르는 근본적인 선택임을 설명할 수 있다.
- 이 챕터를 마치면, CliffWalking에서 SARSA와 Q-learning이 서로 다른
  경로(안전한 17스텝 vs 최단 13스텝)를 배우는 이유를 "답하는 질문이
  다르다"는 관점에서 설명하고, 학습 도중 리턴과 탐욕적 롤아웃 리턴을
  구분해서 읽을 수 있다.

세 수업 블록은 한 흐름을 이룬다. 먼저 **6.1**에서 가치 예측(V(s))의
부트스트래핑 골격을 세우고, **6.2**에서 그 골격에 행동을 얹어 Q(s,a)로
확장하고 on-policy(SARSA)를 만난 뒤, **6.3**에서는 목표값의 한 항
\\(Q(s',a')\\)을 \\(\max\\)으로 바꿔 off-policy(Q-learning)로 건너뛴다.
"어떤 가치함수를 갱신하느냐"는 6.1→6.2에서, "목표값의 다음 행동을
어떻게 채우느냐"는 6.2→6.3에서 바뀐다. 이 두 갈림을 붙잡으면, 이
챕터 이후 DQN(Chapter 9)[^dqn]·액터-크리틱(Chapter 11)까지 이어지는
on-policy/off-policy의 큰 축도 미리 놓기 쉬워진다.

## 이번 주 수업 블록

이번 주는 세 개의 수업 블록으로 진행된다:

- [6.1 TD(0): 한 스텝만 보고 갱신한다, TD vs MC](chapter06/1.md)
  TD(0) 갱신식 \\(V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]\\)
  를 정의하고, 7-state random walk에서 손계산으로 "방향은 맞췄지만
  크기는 틀린" 첫 에피소드를 끝까지 따라간다. MC와 TD의 편향-분산
  차이를 목표값 분포의 실제 숫자로 확인하고, "TD 오차 = 예측 오차"라는
  시각과 수렴 조건(Robbins-Monro)까지 다룬다.
- [6.2 SARSA와 CliffWalking 실습](chapter06/2.md)
  실제로 고른 다음 행동 \\(a'\\)를 목표값에 쓰는 on-policy 알고리즘
  SARSA를 배운다. 4×12 절벽 격자 CliffWalking에서 SARSA가 왜 절벽을
  피하는 17스텝 안전 경로를 배우는지(탐험 위험이 \\(\varepsilon\\)-greedy
  가치에 가격으로 들어와 있기 때문[^silvercourse]) 숫자로 풀어내고, \\(\varepsilon\\)
  감쇠가 배운 경로의 모양까지 바꾸는 것도 실험한다.
- [6.3 Q-learning과 SARSA·Q-learning 경로 비교](chapter06/3.md)
  \\(Q(s',a')\\)를 \\(\max\_{a'}Q(s',a')\\)로 바꾼 off-policy
  알고리즘 Q-learning을 다룬다. \\(\max\\)가 "아직 고르지 않은 행동의
  가치까지 끌어들이는" 낙관적 부트스트래핑임을 짚고, 같은 CliffWalking에서
  SARSA(안전 17스텝)와 Q-learning(최단 13스텝)가 답하는 질문이 다르다는
  것을 학습 도중/탐욕적 리턴의 비교 숫자로 보여준다(5.3절의 중요도
  샘플링과도 연결한다).

[^suttonbarto]: Sutton, R. S., Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.). MIT Press. 저자 공식 무료 공개: http://incompleteideas.net/book/the-book-2nd.html

[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/

[^silvercourse]: Silver, D. (2015). "UCL Course on Reinforcement Learning," Advanced Topics (COMPM050/COMPGI13) — 10개 강의 슬라이드(PDF) 및 영상 강의가 공개되어 있다. https://www.davidsilver.uk/teaching/ (본 장과 가장 직접적으로 겹치는 Lecture 9: Exploration and Exploitation — 47쪽, ε-greedy·멀티암 밴딧·컨텍스트 밴딧: https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-9-exploration-and-exploitation.pdf, CC-BY-NC 4.0).

[^dqn]: Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529–533. (Earlier preprint: Mnih, V. et al. (2013). arXiv:1312.5602.)
