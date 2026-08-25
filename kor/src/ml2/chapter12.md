# Chapter 12. 모방학습과 인간 피드백 (Imitation Learning & Learning from Human Feedback)

Chapter 2~11에서 배운 모든 알고리즘은 에이전트가 스스로 시행착오를 겪으며
보상 신호로부터 배웠다. 그런데 어떤 문제는 "시행착오" 자체가 위험하거나
비싸다 — 로봇 팔이 사람 옆에서 물건을 옮기는 법을 무작위로 시도하며
배우게 둘 수는 없다. 이번 장은 보상을 직접 최적화하는 대신, **사람의
시연**이나 **사람의 선호**로부터 배우는 두 가지 방법을 다룬다.

## 왜 이 순서인가

Chapter 11에서 PPO[^ppo] — 보상함수를 코드로 쓸 수 있고, 에이전트가 안전하게
탐험할 수 있는 환경이 주어지면 에이전트가 스스로 시행착오로 최적 정책을
찾는 알고리즘으로 끝났다. 이 장은 그 두 전제 — 자연스러운 운동은 어떤
스칼라 함수로도 쓸 수 없고, 실세계 탐험은 사람을 다치게 할 수 있다 — 가
동시에 무너지는 문제들에 대한 답이다. 다음 장 "로봇 시뮬레이션과 제어
기초"는 이 장의 방법들을 실제 로봇 시뮬레이션 환경으로 옮겨, 시연으로
만든 초기 정책을 PPO가 어떻게 미세조정하는지 구체적으로 본다. 이 주제
(모방학습과 RLHF)를 더 깊이 다루는 자료로 스탠포드 CS234: Reinforcement
Learning을 추천한다.[^cs234]

![서로게이트 함수 L_CLIP의 한 항(단일 timestep)을 확률비 r의 함수로 그린 그래프 — 왼쪽은 이익이 양수(A>0), 오른쪽은 음수(A<0)인 경우. (원 논문 Figure 1)](../images/ref_ppo.png)

## 세 절이 이어지는 방식

세 절은 "시연에서 배운다 (12.1) → 비교에서 배운다 (12.2) → 어떤 쪽을
쓸 것인가 (12.3)" 순서다. 12.1은 모방학습을 "시연 데이터를 지도학습에
꽂아 넣는 지름길"로 세우고 그 한계인 복합 오차를 드러낸다. 12.2는
사람의 신호를 "정답 행동"에서 "둘 중 나은 쪽"으로 일반화해, 보상모델[^rlhf]이라는
우회로로 전문가 상한을 넘을 수 있는 길을 연다. 12.3는 둘을 실험으로
통합해 원칙이 아니라 숫자로 선택 기준을 준다.

## 이 챕터를 마치면

- 행동 복제를 "(상태, 행동) 시연 쌍을 라벨로 쓰는 지도학습 문제"로
  설명하고, 학습 분포와 배포 시 분포가 어긋나며 오차가 눈덩이치는
  **복합 오차**[^covshift]의 본질을 한 문장으로 짚을 수 있다.
- DAgger[^dagger] 네 줄 — 에이전트가 자신의 정책으로 달리고, 전문가가 방문한
  상태에 라벨만 답하는 반복 — 을 장난감 환경에 적용해 보고, "시연을
  더 많이 모으는 것"이 그 실패를 해결하지 못하는 이유(수량이 아니라
  분포 문제)를 설명할 수 있다.
- Bradley-Terry 선호 쌍으로부터 보상모델을 학습하고 그 점수를 PPO의
  보상으로 쓰는 RLHF 두 단계를 설명하고, 스케일 불변성·보상 해킹 위험
  (한계)과 전문가 상한 초월 가능(장점)을 함께 짚을 수 있다.
- 시연·비교·보상함수·안전한 탐험 중 어떤 자원을 가진 문제인지 확인하고,
  BC / DAgger / 선호 기반 보상모델 / BC+RL 선택의 기준을 episode 길이
  와 \\(p^T\\) 감쇠로 정량적으로 따질 수 있다.

## 이번 주는 세 개의 수업 블록으로 진행된다

- [12.1 모방학습: 행동 복제와 DAgger](chapter12/1.md) — 행동 복제는
  Chapter 2의 로지스틱회귀 코드 한 줄도 바꾸지 않은 "시연 (상태, 행동)
  쌍을 라벨로 꽂아 넣기"다. 그런데 작은 실수 한 번이 에이전트를 시연에
  없던 상태로 빼내면 오차가 눈덩이처럼 불어난다 — 손으로 풀어 볼 수
  있는 3×4 회랑에서 그 실패를 보고, 에이전트가 자신의 정책으로 달리고
  전문가가 방문한 상태에 라벨만 답하는 DAgger가 2라운드 만에 실패를
  고치는 과정을 확인한다.
- [12.2 선호 기반 보상모델과 실습](chapter12/2.md) — "정답을 시연하는
  것"이 어렵고 "두 시도 중 나은 쪽을 고르는 것"이 쉬운 문제(자연스러운
  걸음걸이 등)에는, 비교 데이터만으로 Bradley-Terry 보상모델을 학습하고
  그 점수를 PPO의 보수로 쓴다 — LLM을 다듬는 RLHF[^instructgpt]와 같은 구조다.
  선호 쌍 300개로 숨겨진 보상함수의 방향(비율)을 복원하는 실습, 그리고
  이 대리 보상이 보상모델의 약점을 파고드는 보상 해킹[^rewardhacking] 위험을 안고
  있다는 점도 확인한다.
- [12.3 모방학습 vs 강화학습: 언제 무엇을 쓰는가](chapter12/3.md) —
  "어느 쪽이 낫나요?"는 잘못된 질문 — 선택은 가진 자원(시연, 비교,
  보상함수)이 정한다. 5×5 격자에서 순수 BC·순수 RL·BC+RL을 함께 돌려,
  순수 BC가 단 한 번도 전문가(10스텝 우회)를 못 넘고 함정에 26회
  들어가는 이유, 순수 RL이 8스텝 최적을 찾되 학습 중 위험 행동을 30회
  하는 이유, 하이브리드 BC+RL이 초기 안전성(9회)과 추월 속도(193 vs
  273 episode)를 동시에 얻는 이유를 숫자로 확인한다.

![Figure 1: RLHF 접근법의 구조를 보여주는 개략도 — 보상 예측기(reward predictor)가 트래젝터리 세그먼트의 인간 비교 피드백으로 비동기 학습된 뒤, 그 보상을 이용해 정책(policy)을 강화학습으로 학습하는 전체 파이프라인을 보여준다.](../images/ref_rlhf.png)

![InstructGPT 3단계 학습 파이프라인 (원 논문 Figure 2) — (1) 지도 미세조정(SFT), (2) 보상 모델(RM) 학습, (3) 이 보상 모델을 이용한 근접 정책 최적화(PPO) 강화학습.](../images/ref_instructgpt.png)

[^ppo]: Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/
[^rlhf]: Christiano, P. et al. (2017). "Deep reinforcement learning from human preferences." NeurIPS 2017. arXiv:1706.03741.
[^dagger]: Ross, S., Gordon, G. J., Bagnell, J. A. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS 2011. arXiv:1011.0686.
[^instructgpt]: Ouyang, L. et al. (2022). "Training language models to follow instructions with human feedback." arXiv:2203.02155.
[^covshift]: Sugiyama, M., Suzuki, T., Kanamori, T. (2012). "Metrics for Discriminative Domain Adaptation." NeurIPS 2012. arXiv:1206.6323. (학습·배포 분포 불일치 — 코바리에이트 시프트 — 를 측정하는 지표의 원 논문)
[^rewardhacking]: Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. (2022). "Defining and Characterizing Reward Hacking." arXiv:2209.13085.
