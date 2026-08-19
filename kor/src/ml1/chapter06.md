# Chapter 6. 정규화와 모델 선택 (Regularization & Model Selection)

1996년, 통계학자 로버트 팁시라니(Robert Tibshirani)는 "Lasso"(Least Absolute
Shrinkage and Selection Operator)라는 방법을 제안했다. 아이디어는 단순했다 —
회귀의 손실함수에 가중치 절댓값의 합을 페널티로 더하면, 놀랍게도 중요하지 않은
특징의 가중치가 **정확히 0**이 되어버린다. 특징이 수천 개인 문제에서 "어떤
특징이 진짜 중요한가"를 사람이 일일이 고르는 대신, 손실함수 하나를 바꾸는 것만
으로 모델이 스스로 걸러낸다는 뜻이다. 이번 장은 Chapter 4.1에서 본 편향-분산
트레이드오프를 실제로 **조절하는 손잡이**를 다룬다.

이번 주는 세 개의 수업 블록으로 진행된다:

- [6.1 정규화: 편향-분산 복습과 L1/L2 페널티](chapter06/1.md)
- [6.2 교차검증: \\(\lambda\\)를 데이터로 정한다](chapter06/2.md)
- [6.3 train/val/test 분리 원칙 실전](chapter06/3.md)
