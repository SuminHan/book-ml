# Chapter 4. 동적계획법 (Dynamic Programming)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuminHan/book-ml/blob/main/notebooks/ml2/chapter04_policy_evaluation.ipynb)

Chapter 3에서 MDP를 정식화했지만, "누적 보상을 최대화하는 행동을 고른다"는
목표는 여전히 계산 불가능해 보인다 — 무한히 먼 미래까지 내다봐야 하는
것처럼 보이기 때문이다. 벨만의 통찰은 이 무한합을 **재귀적 관계**로 다시
쓸 수 있다는 것이었다(Chapter 3.3에서 이미 벨만 기대방정식으로 확인했다).
이번 장은 그 재귀식을 실제로 계산하는 절차(정책평가)와, MDP의 전이확률
\\(P\\)를 정확히 알고 있다는 전제 하에(모델 기반, model-based) 최적
정책을 찾는 절차(정책반복·가치반복)를 다룬다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [4.1 벨만 최적방정식과 정책평가](chapter04/1.md)
- [4.2 정책 반복과 GridWorld 실습](chapter04/2.md)
- [4.3 가치 반복, 수렴 증명, 그리고 모델 기반이라는 전제](chapter04/3.md)
