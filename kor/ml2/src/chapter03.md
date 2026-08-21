# Chapter 3. MDP 정식화 (Markov Decision Processes)

1950년대, 수학자 리처드 벨만(Richard Bellman)은 "차원의 저주(curse of
dimensionality)"라는 용어를 만들면서 동적계획법(dynamic programming)이라는
최적화 기법을 고안했다. 그의 핵심 통찰 — "복잡한 문제를 한 번에 풀지 말고,
더 작은 부분 문제로 쪼개서 그 답을 재귀적으로 조합하라" — 는 지금 강화학습
이론 전체의 뼈대를 이루고 있다. 이번 장은 Chapter 2의 밴딧에 "상태"를
더해서, 강화학습 문제를 수학적으로 엄밀하게 정의하는 틀 — **MDP**(Markov
Decision Process) — 를 세운다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [3.1 MDP의 다섯 요소와 마르코프 성질](chapter03/1.md)
- [3.2 누적 보상, 할인율, 그리고 실습](chapter03/2.md)
- [3.3 가치함수와 벨만 기대방정식](chapter03/3.md)
