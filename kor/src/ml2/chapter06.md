# Chapter 6. 시간차 학습 (Temporal-Difference Learning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter06_q_learning.ipynb)

1989년, 크리스 왓킨스(Chris Watkins)는 박사 학위 논문에서 **Q-learning**
이라는 알고리즘을 제안했다. Chapter 5의 몬테카를로는 모델 없이도 배울 수
있었지만, 에피소드가 끝날 때까지 기다려야 리턴을 계산할 수 있었다.
**시간차 학습**(Temporal-Difference, TD)은 이 기다림 자체를 없앤다 — 한
스텝만 진행해보고, "지금 추정치"와 "방금 관찰한 보상 + 다음 상태의
추정치"의 차이만큼 즉시 갱신한다. 모델도 필요 없고, 에피소드가 끝나기를
기다릴 필요도 없는, 강화학습에서 가장 널리 쓰이는 절충안이다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [6.1 TD(0): 한 스텝만 보고 갱신한다, TD vs MC](chapter06/1.md)
- [6.2 SARSA와 CliffWalking 실습](chapter06/2.md)
- [6.3 Q-learning과 SARSA·Q-learning 경로 비교](chapter06/3.md)
