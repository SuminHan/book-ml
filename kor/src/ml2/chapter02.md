# Chapter 2. 멀티암 밴딧 (Multi-Armed Bandits)

카지노에는 손잡이(암, arm)가 여러 개 달린 슬롯머신들이 있다고 상상해보자.
각 손잡이는 서로 다른, 그러나 당신은 모르는 평균 보상을 가지고 있다.
제한된 횟수만 당길 수 있다면, 어떤 순서로 손잡이를 당겨야 총 보상을
최대화할 수 있을까? 이 단순한 질문 — **멀티암 밴딧**(Multi-Armed
Bandit) 문제[^suttonbarto] — 이 강화학습이 다루는 모든 문제의 가장 작은 뼈대다. 다음
장부터 배울 상태(state)도, 여러 스텝에 걸친 전이(transition)도 없이,
"탐험과 활용의 딜레마"라는 강화학습의 핵심 문제 하나만 순수하게 남겨둔
가장 단순한 형태이기 때문이다.

## 이전·다음 챕터와 어떻게 이어지는가

이전 장에서는 이번 학기의 무대를 마련했다 — "행동하고, 그 결과로
나중에 돌아오는 보상만 보고 스스로 좋은 행동을 찾아내야 하는 문제"를
강화학습으로 정의하고, 앞으로 함수 근사기로 쓸 신경망과 Gymnasium
[^gymnasium] 실습 환경을 확인했다. 이 장의 밴딧 문제는 바로 그 무대의 첫 수업에
해당한다 — 강화학습의 정의가 구체적으로 드러나는 가장 작은 문제다.
상태도 전이도 없고, "행동 → 보상"이라는 반복 구조와 매 순간 무엇을
할지 결정하는 문제만 남는다. 다음 장(MDP 정식화)에서는 바로 이 밴딧에
**상태**라는 재료를 하나 더해서, 강화학습 문제를 수학적으로 엄밀하게
정의하는 틀을 세운다. 즉, 이 장은 다음 장부터 이어질 MDP·가치함수·
벨만방정식이라는 산에 오르기 전, "탐험-활용의 딜레마" 하나만 먼저
낱으로 다뤄 직감을 기르는 출발점인 셈이다. 이 주제를 더 깊이 다루는 자료: [^cs234]

## 이 챕터를 마치면

- k개의 팔(arm)이 각자 모르는 진짜 평균 보상 \\(q^\\*(a)\\)을 가진
  밴딧 문제를 총보상과 후회(regret)라는 두 잣대로 정식화하고, 순수
  탐욕적 선택이 "단 한 번의 나쁜 운에 영영 박힐 수 있다"는 이유를
  한 문장으로 설명할 수 있다.
- ε-greedy[^suttonbarto]와 증분 갱신식 \\(Q \\leftarrow Q + \\frac{1}{N}(R-Q)\\)을
  직접 코드로 구현하고, 이 갱신식이 "지금까지 관찰한 보상의 평균을
  하나씩 온라인으로 갱신하는 것"과 수학적으로 같은 이유를 손으로
  확인할 수 있다.
- ε-greedy, 낙관적 초기화[^suttonbarto], UCB[^ucb] 세 전략을 "탐험이 어떻게 만들어지는가"
  (고정 확률 / 초기값 예약 / 매 스텝 불확실성 재계산) 기준으로 비교하고,
  고정 환경과 시간이 지나면 팔의 순위가 뒤바뀌는 비정상(non-stationary)
  환경에서 각각 어떤 전략이 유리한지 설명할 수 있다.[^silvercourse]
- 밴딧을 "상태가 하나뿐인 MDP"로 다시 해석하고, "상태"가 들어오는
  순간 문제의 무엇이 커지는지(좋은 **팔** 1개 찾기 → 좋은 **정책**
  찾기) 한 문장으로 말할 수 있다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [2.1 탐험과 활용: ε-greedy](chapter02/1.md) —
  밴딧 문제를 정식화(총보상, 후회)하고, 이 책에서 첫 정책이 되는
  ε-greedy("탐험의 양을 하나의 확률 \\(\\varepsilon\\)로 직접 조종한다")
  를 구현한다. 증분 갱신식이 온라인 평균과 같은 것임을 손으로 증명하고,
  시뮬레이션으로 순수 탐욕(ε=0)이 "한 번의 나쁜 운"에 막혀 2000스텝
  후회가 1754까지 폭주하는 반면, ε를 0→1로 훑으면 총보상 최대치가
  양쪽 극단이 아닌 가운데(ε≈0.05)에 있다는 "탐험에도 비용이 있다"는
  감각을 숫자로 만든다.
- [2.2 낙관적 초기화와 UCB: 불확실성을 이용한 탐험](chapter02/2.md) —
  탐험을 알고리즘 내부에서 자동으로 유도하는 전략 두 가지다.
  낙관적 초기화는 모든 \\(Q\_0(a)\\)를 실제 보상보다 훨씬 크게
  (예: 5.0) 걸어둠으로써 순수 탐욕만으로도 각 팔이 정확히 한 번씩
  당겨지게 하고, UCB는 추정치에 \\(c\\sqrt{\\ln t/N}\\)에 비례하는
  "불확실성 보너스"를 더해 덜 당겨본 팔을 자동으로 재시험하게 한다.
  같은 문제로 세 전략을 붙여 싸우면, 정보와 무관하게 탐험하는
  ε-greedy가 가장 낭비하고, 팔의 순위가 뒤바뀌는 비정상 환경에서는
  매 스텝마다 보너스를 재계산하는 UCB만이 새 최선을 다시 찾아
  적응하는 것을 확인할 수 있다.
- [2.3 밴딧 vs 완전한 MDP: 다음 장으로 가는 다리](chapter02/3.md) —
  밴딧을 "상태가 하나뿐인 MDP"(자기 전이뿐)로 다시 쓰고,
  1-상태 MDP의 가치 \\(V^\\pi(s^\\*) = q^\\*(a)/(1-\\gamma)\\)와
  두 상태짜리 장난감 MDP(\\(\\gamma=0.9\\)에서
  \\(V(s\_0)=28\\), \\(V(s\_1)=30\\))의 벨만방정식을 손으로 풀어,
  "한 상태의 가치는 다른 상태의 가치에 의존한다"는 연립 구조를
  처음 목격한다. 임상시험(적응적 배정)과 A/B 테스트라는 실제 사례로,
  "지금 한 행동이 다음 상태 자체를 바꾸는가?"라는 판정 기준으로
  "밴딧인가, MDP인가?"를 구분하는 법을 배운다.

[^suttonbarto]: Sutton, R. S., Barto, A. G. (2018). "Reinforcement Learning: An Introduction" (2nd ed.). MIT Press. 저자 공식 무료 공개: http://incompleteideas.net/book/the-book-2nd.html

[^cs234]: Stanford CS234: Reinforcement Learning. https://web.stanford.edu/class/cs234/

[^gymnasium]: Towers, M., Kwiatkowski, A., Terry, J., et al. (2024). "Gymnasium: A Standard Interface for Reinforcement Learning Environments." arXiv:2407.17032.

[^ucb]: Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." Machine Learning 47(2/3), 235–256. — UCB1 알고리즘과 로그 후회 경계의 원 논문.

[^silvercourse]: Silver, D. (2015). "UCL Course on Reinforcement Learning," Advanced Topics (COMPM050/COMPGI13) — 10개 강의 슬라이드(PDF) 및 영상 강의가 공개되어 있다. https://www.davidsilver.uk/teaching/ (본 장과 가장 직접적으로 겹치는 Lecture 9: Exploration and Exploitation — 47쪽, ε-greedy·멀티암 밴딧·컨텍스트 밴딧: https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-9-exploration-and-exploitation.pdf, CC-BY-NC 4.0).
