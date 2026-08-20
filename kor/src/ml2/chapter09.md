# Chapter 9. 함수근사와 DQN (Function Approximation & Deep Q-Networks)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter09_dqn_tricks.ipynb)

2013년, 딥마인드는 Q-learning의 Q-테이블을 신경망으로 통째로 바꾼
알고리즘으로 Atari 2600 게임 여러 개를 학습시켰다 — 게임의 규칙을 전혀
알려주지 않고, 오직 화면 픽셀과 점수만 보고서. **DQN**(Deep
Q-Network)이라 불린 이 결과는 2015년 Nature에 게재되며, 여러 게임에서
사람 수준 또는 그 이상의 성능을 보였다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [9.1 Q-테이블의 한계와 함수근사](chapter09/1.md)
- [9.2 DQN의 안정화 장치와 실습](chapter09/2.md)
- [9.3 경험 재현과 CartPole 학습곡선 실습](chapter09/3.md)
