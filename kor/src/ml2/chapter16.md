# Chapter 16. Block B 캡스톤: 팀 프로젝트와 학기 총정리 (Block B Capstone & Semester Review)

Chapter 9~15는 표 기반 강화학습을 신경망으로 확장하고(DQN)[^dqn], 정책을
직접 학습하고(REINFORCE, PPO)[^ppo], 사람의 시연·선호로부터
배우고(모방학습)[^rlhf], 실제 물리 시뮬레이션(MuJoCo[^mujoco], Isaac Sim[^isaacsim])과 트리
탐색(MCTS)[^mcts]까지 다뤘다. 이번 장은 Chapter 8의 짝이다 — 이 두 번째 절반의
도구들을 로봇 시뮬레이션 환경에 직접 적용해보고, 학기 전체를 총정리한다.

![DQN의 합성곱 신경망 구조 -- Atari 화면을 입력받아 조이스틱 행동별 Q값을 출력한다 (원 논문 Extended Data Figure 1).](../images/ref_dqn.png)

![서로게이트 함수 L\_CLIP의 한 항(단일 timestep)을 확률비 r의 함수로 그린 그래프 — 왼쪽은 이익이 양수(A>0), 오른쪽은 음수(A<0)인 경우. (원 논문 Figure 1)](../images/ref_ppo.png)

![Figure 1: RLHF 접근법의 구조를 보여주는 개략도 — 보상 예측기(reward predictor)가 트래젝터리 세그먼트의 인간 비교 피드백으로 비동기 학습된 뒤, 그 보상을 이용해 정책(policy)을 강화학습으로 학습하는 전체 파이프라인을 보여준다.](../images/ref_rlhf.png)

왜 이 순서인가. 이전 장("모델기반 RL과 몬테카를로 트리 탐색")은
"모델이 있을 때"의 도구 상자 마지막 조각, MCTS를 보여줬다 — AlphaGo[^alphago]의
무기였던 UCT[^uct]로 트리를 탐색하고, 정책·가치 네트워크가 탐색을
어떻게 돕는지
짚은 장이다. Chapter 15가 이 두 번째 절반에서 새 도구를 추가한 마지막 장
인 셈이므로, 이 장부터는 배울 개념이 남지 않는다. 대신 이 책은 손에 든
도구들을 실제 로봇 시뮬레이션 환경에 직접 적용해, 그 결과를 정직하게
보고하는 것으로 막을 내린다. Chapter 8(Block A 캡스톤)과 쌍을 이루듯,
동일한 절차(환경 고르기 → 구현 → 평가 프로토콜 → 정직한 발표)를 완전히
다른 무대 — 연속 제어 + 신경망 — 에서 다시 집행한다. 세 절은 "프로젝트를
실행해 발표한다(16.1) → 서로를 심사한다(16.2) → 학기 전체를 자기 것으로
다진다(16.3)"라는 하나의 아크를 이룬다.

![정책망·가치망 학습 파이프라인(왼쪽)과 두 신경망의 합성곱 구조(오른쪽) (원 논문 Figure 1).](../images/ref_alphago.png)

이번 챕터의 학습 목표는 다음과 같다.

- 연속제어 환경에서 4주 파이프라인(환경 고르기 → 구현 → 평가 → 발표)으로
  팀 프로젝트를 설계·실행하고, "시드 ≥3, 무작위 정책 베이스라인,
  학습/평가 분리" 프로토콜을 실제로 수행할 수 있다.
- "학습 곡선이 올랐다"와 "배운 정책이 무작위보다 낫다"가 서로 다른
  문장임을 구분하고, 학습 도중(탐험 포함) 성과와 학습 후 결정적(a=μ)
  평가를 비교·보고하며 그 격차를 11.2절의 분산 원리로 해석할 수 있다.
- 다른 팀의 발표에 5항목 체크리스트를 적용하고, Ch9~15의 개념(장·절
  명시)을 근거로 담고 발표팀이 새 실험을 돌려야 답할 수 있는 3점급
  "반박 가능한 질문"을 쓴 peer-review를 작성할 수 있다.
- 학기 전체를 "모델 정의 → 틀린 정도를 수치화 → 개선" 3단계 구조로
  정리하고, 20문항 자가진단으로 준비도를 재며, 학기 이후로 이어질 방향
  (오프라인 RL, 범용 로봇 정책, sim-to-real)을 자기 언어로 말할 수 있다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [16.1 팀 프로젝트: 발표](chapter16/1.md)
  연속제어 환경(Pendulum, MuJoCo 기반 로봇 등)을 고르고 Chapter 9~15
  중 방법(DQN / PPO / 모방학습 / MCTS)으로 정책을 학습시키는, 8.1절의
  쌍(pair)에 해당하는 절이다. 무작위 정책의 베이스라인 리턴을 먼저
  재고(예: Pendulum −1274), 시드 3개 이상으로 실행한 뒤, 탐험이 섞인
  학습 도중 성과와 학습 후 결정적(a=μ) 평가를 비교한다 — 이 실습에서는
  결정적 평가가 오히려 베이스라인 아래라는 "12만 스텝 예산의 한계"가
  정직하게 드러난다. 그 위에 수식적 정당화 최소 1개와 "잘 안 된 부분과
  원인 진단"을 포함한 5~7분 발표를 한다. 채점 기준의 절반은 실험
  프로토콜의 정직성과 결과의 정직성·반성에 배정된다.
- [16.2 Peer-Review](chapter16/2.md)
  이번에는 네가 심사위원이다. 다른 두 팀의 발표를 듣고 5항목
  체크리스트(환경 설명, 알고리즘 선택, 수식적 정당화, 결과의 정직성,
  탐험-활용 균형)를 적용해 리뷰 두 통을 작성하고, 그 리뷰의 *질*로
  채점받는다. 이 절의 중심은 신경망 RL 특유의 "우연히 좋은" 패턴
  5가지 — 시드 1개 학습 곡선, 학습 도중 리턴으로 내린 수렴 결론,
  하이퍼파라미터 민감성 미검증, 이산용 알고리즘(DQN)을 연속 환경에 쓰는
  것, 베이스라인 없는 "잘 배웠다" — 과 1점(감상평) → 3점(반박 가능한
  질문) 3단계 채점 기준이다.
- [16.3 ML2 총정리와 학기를 넘어서](chapter16/3.md)
  새로운 개념이 하나도 없는 절이다. 16.1~16.2가 수렴하는 프로젝트 채점
  기준을 확정하고, 하나의 작은 MDP(달리거나 쉬거나)를 Chapter 4~6의 네
  경로(동적계획법, 몬테카를로, SARSA, Q-learning)로 풀어 "같은
  방정식, 다른 형태"를 숫자로 확인한다. 그 뒤 ML2 개념 지도, 20문항
  복습 자가진단, 학기 이후에 가져갈 세 가지와 최신 동향(오프라인 RL,
  범용 로봇 정책, sim-to-real)으로 이 학기를 마무리한다.[^cs234]

[^dqn]: Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529–533. (Earlier preprint: Mnih, V. et al. (2013). arXiv:1312.5602.)
[^ppo]: Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
[^rlhf]: Christiano, P. et al. (2017). "Deep reinforcement learning from human preferences." NeurIPS 2017. arXiv:1706.03741.
[^alphago]: Silver, D. et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature 529, 484–489.
[^cs234]: 이 장의 주제(연속제어 RL, 팀 프로젝트 실습, 시드·베이스라인·평가 분리 프로토콜)를 더 깊이 다루는 자료: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/
[^uct]: Kocsis, L., Szepesvári, C. (2006). "Bandit based Monte-Carlo Planning." ECML 2006, LNAI 4212, 282–293. — UCT(Upper Confidence bounds applied to Trees)의 원 논문.
[^mujoco]: Todorov, E., Eret, T., Tassa, Y. (2012). "MuJoCo: A physics engine for model-based control." IROS 2012, 5026–5033.
[^isaacsim]: Gao, S., Pagnucco, M., Bednarz, T., Song, Y. (2026). "NVIDIA Isaac Sim: Enabling Scalable, GPU-Accelerated Simulation for Robotics." arXiv:2606.03551.
[^mcts]: Browne, C. B., Cowling, P. I., White, M., et al. (2012). "A Survey of Monte Carlo Tree Search Methods." IEEE Transactions on Computational Intelligence and AI in Games 4(1), 1–43. MCTS 방법론을 체계적으로 정리한 대표적 서베이 — "Monte-Carlo Tree Search"라는 이름과 선택·확장·평가·역전파의 표준 4단계 체계가 이 무렵 확립되었다.
