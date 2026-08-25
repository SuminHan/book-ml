# Chapter 15. 모델기반 RL과 몬테카를로 트리 탐색 (Model-Based RL & Monte Carlo Tree Search)

2016년, 딥마인드의 **AlphaGo**[^alphago]는 세계 최정상급 프로 바둑기사 이세돌을
꺾어 큰 화제를 모았다. 바둑판의 경우의 수는 우주의 원자 수보다 많다고
알려져 있어서, Chapter 9의 DQN[^dqn]처럼 "신경망으로 가치를 근사하는 것"만으로는
충분하지 않았다 — AlphaGo의 핵심 무기 중 하나가 바로 이번 장의 주제,
**몬테카를로 트리 탐색**(Monte Carlo Tree Search, MCTS)[^uct]이다. 이번 장은
Chapter 7.3에서 잠깐 맛본 "모델을 활용한 계획"이라는 아이디어를,
게임처럼 명확한 규칙이 있는 문제에서 훨씬 강력하게 확장한 형태를 다룬다.

이 연결고리를 이전·다음 챕터와 함께 짚어보자. 그동안의 ML2 강의 —
DQN, REINFORCE/PPO[^ppo], 모방학습까지 — 는 대부분 모델이 없는(model-free)
환경에서 시행착오로 배우는 방법들이었다. 이번 장은 그 반대편,
환경의 규칙이 주어졌을 때 그 **모델**을 이용해 미리 계획하고 결정하는
방법을 다룬다. 이 두 축이 합쳐져야 비로소 AlphaGo/AlphaZero[^alphazero] 같은
게임 AI가 완성된다. 이전의 Chapter 14("고급 시뮬레이션: MuJoCo[^mujoco]와
Isaac Sim[^isaacsim]")에서 실전 물리엔진 두 가지를 소개했는데, 그 시뮬레이터
자체가 로봇 분야에서 모델이 *주어진* 경우, 즉 known model에 해당한다.
이번 장은 13~14장에서 시뮬레이터 위를 "배우며" 다닌 경험을, "모델로
계획한다"는 관점 위에서 한 번 더 짚어보는 장이다. 그리고 다음 장("Block
B 캡스톤: 팀 프로젝트와 학기 총정리")에서는 MCTS를 포함해 후반부에서
배운 모든 도구를 로봇 시뮬레이션 팀 프로젝트에 직접 적용한다 — 이번 장
은 그 캡스톤에 들어가기 전, 새 도구를 챙기는 마지막 장이다.

이번 챕터의 학습 목표는 다음과 같다.

- 모델기반 RL을 "모델이 어디에서 오느냐"(known model/learned model)와
  "모델을 무엇에 쓰느냐"(학습/계획)라는 두 축으로 정리하고, 가치반복·
  Dyna-Q[^suttonbarto]·MCTS·모델 예측 제어(MPC)가 각각 그 표의 어디에 앉는지 설명할
  수 있다.
- MCTS의 네 단계(선택·확장·시뮬레이션·역전파)가 한 번의 시뮬레이션에서
  각각 무슨 일을 하는지 단계별로 설명하고, 님 게임에서 MCTS를 직접
  구현해 시뮬레이션 횟수와 탐색 상수 \\(c\\)가 찾은 전략에 미치는
  영향을 실험으로 확인한다.
- 선택 규칙 UCT[^uct]가 Chapter 2.2의 UCB[^ucb]를 트리 탐색에 적용한 것임을
  설명하고, AlphaGo/AlphaZero가 MCTS에 신경망을 어떻게 결합해
  확장했는지 그 구조를 짚을 수 있다.
- (선택) 여러 에이전트가 함께 학습하면 "내 시점의 환경"이 학습 도중
  계속 바뀌는 비정상성(non-stationary) 문제 때문에 단일 에이전트의
  수렴 보장은 무너지는 이유를 설명하고, 죄수의 딜레마·조율 게임·
  바위-가위-보에서 알고리즘이 같아도 보상 구조에 따라 학습 결과가
  달라지는 이유를 말할 수 있다.

하나의 큰 그림을 먼저 놓는다. 모델이 있으면(또는 주어지면) 시행착오
횟수를 크게 줄일 수 있다는 이점이 있지만, 동시에 모델의 오차가 학습
결과에 그대로 전이되는 리스크도 생긴다. MCTS[^mcts]는 이 딜레마에 대해 "전체
상태 공간을 계산하는 대신, 지금 필요한 갈림길만 깊게 계산한다"는
절충을 보여주는 표준 사례다 — 이 "부분 계획"의 아이디어가 이번 장의
세 절을 관통하는 주제다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [15.1 모델기반 RL 개관: Dyna-Q 재조명과 "공짜 모델"](chapter15/1.md)
  Dyna-Q를 복습해 "known/learned 모델 × 학습/계획"의 두 축으로
  모델기반 RL을 정리한다. 체스·바둑처럼 규칙 자체가 모델인 "공짜
  모델"에서 완전탐색이 가능할 때와 불가능할 때를 비교하고, 알려진
  모델에 가치반형을(계획)을 한 스텝 손계산으로 확인한다. 학습된
  모델로 계획을 하는 Dyna-Q가 Q-learning보다 학습을 얼마나 빠르게
  하는지 숫자로 재보고, "모델이 있으니 학습이 필요 없다"는 흔한
  착각까지 짚어본다.
- [15.2 몬테카를로 트리 탐색: 이론과 실습](chapter15/2.md)
  MCTS의 네 단계 반복과 UCT 선택 규칙을 배우고, 직접 구현해 님 게임
  에서 최적 전략을 찾아본다. 더 큰 국면(돌 14개)에서 탐색 상수
  \\(c\\)의 효과를 보고, 틱택토 첫 수 실습으로 15.1절의 완전탐색과
  대조한다. 실제로 잘 발생하는 구현 실수 네 가지를 버그 투어로 점검한
  뒤, MCTS에 신경망을 결합한 AlphaGo/AlphaZero까지 이어간다.
- [15.3 멀티에이전트 RL 개관 (선택)](chapter15/3.md)
  여러 에이전트가 학습하면 상대가 환경이 되는 순간 "수렴해야 할 정답"
  자체가 움직이는 비정상성 문제를 다룬다. 알고리즘은 죄수의 딜레마,
  조율 게임, 바위-가위-보 세 게임 모두 똑같이 단순한 Q-learning인데,
  결과만 보상 구조에 따라 완전히 달라지는 세 가지 학습 동역학을
  관찰하고, 내쉬 균형과 MCTS·미니맥스·자기 자신과의 대국
  (self-play)과의 연결고리를 살펴본다.[^cs234]

[^alphago]: Silver, D. et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature 529, 484–489.
[^alphazero]: Silver, D. et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815.
[^uct]: Kocsis, L., Szepesvári, C. (2006). "Bandit Based Monte-Carlo Planning." ECML 2006, pp. 282–293.
[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/ — 이 장의 주제(모델기반 RL, 몬테카를로 트리 탐색)를 더 깊이 다루는 자료.
[^dqn]: Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529–533. (Earlier preprint: Mnih, V. et al. (2013). arXiv:1312.5602.)
[^ppo]: Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
[^suttonbarto]: Sutton, R. S., Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.). MIT Press. 저자 공식 무료 공개: http://incompleteideas.net/book/the-book-2nd.html — 이 장의 주제(모델로 계획하는 모델기반 RL, Dyna 계열 접근)는 이 표준 교과서의 "Model-Based RL" 장(모델 기반 강화학습)에 대응한다.
[^mujoco]: Todorov, E., Erez, T., Tassa, Y. (2012). "MuJoCo: A physics engine for model-based control." IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2012). DOI: 10.1109/IROS.2012.6386109.
[^isaacsim]: Makoviychuk, V. et al. (2021). "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning." arXiv:2108.10470. (Isaac Sim의 기반이 되는 GPU 병렬 물리 시뮬레이션의 원 논문)
[^ucb]: Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." Machine Learning 47(2/3), 235–256. — UCB1 알고리즘과 로그 후회 경계의 원 논문.
[^mcts]: Browne, C. B., Cowling, P. I., White, M., et al. (2012). "A Survey of Monte Carlo Tree Search Methods." IEEE Transactions on Computational Intelligence and AI in Games 4(1), 1–43. MCTS 방법론을 체계적으로 정리한 대표적 서베이 — "Monte-Carlo Tree Search"라는 이름과 선택·확장·평가·역전파의 표준 4단계 체계가 이 무렵 확립되었다.
