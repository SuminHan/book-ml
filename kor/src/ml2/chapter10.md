# Chapter 10. 정책기반 강화학습 (Policy-Based Reinforcement Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter10_reinforce_ppo.ipynb)

로봇 팔의 관절에 가할 힘을 정하는 문제를 생각해보자. 이 "행동"은
-10Nm부터 +10Nm까지 **연속적인 값** 중 아무거나 될 수 있다. Chapter
6·9의 Q-learning/DQN은 매 스텝 \\(\max_{a'} Q(s',a')\\)를 계산해야
하는데, 행동이 연속값이면 "가능한 모든 행동"을 나열해서 최댓값을 찾는다는
것 자체가 불가능하다 — 무한히 많은 후보를 다 계산해볼 수는 없다. 이번
장부터 다룰 로봇 시뮬레이션(Chapter 13~14)이 바로 이런 연속 행동공간을
쓰므로, 이 문제를 정면으로 풀어야 한다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [10.1 정책 그래디언트 정리](chapter10/1.md)
- [10.2 REINFORCE와 실습](chapter10/2.md)
- [10.3 Actor-Critic](chapter10/3.md)
