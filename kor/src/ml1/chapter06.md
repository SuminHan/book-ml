# Chapter 6. 정규화와 모델 선택 (Regularization & Model Selection)

1996년, 통계학자 로버트 팁시라니(Robert Tibshirani)는 "Lasso"(Least Absolute
Shrinkage and Selection Operator)[^tibshirani1996]라는 방법을 제안했다. 아이디어는 단순했다 —
회귀의 손실함수에 가중치 절댓값의 합을 페널티로 더하면, 놀랍게도 중요하지 않은
특징의 가중치가 **정확히 0**이 되어버린다. 특징이 수천 개인 문제에서 "어떤
특징이 진짜 중요한가"를 사람이 일일이 고르는 대신, 손실함수 하나를 바꾸는 것
으로 모델이 스스로 걸러낸다는 뜻이다. 이번 장은 Chapter 4.1에서 본 편향-분산
트레이드오프를 실제로 **조절하는 손잡이**를 다룬다.

## 이미 써본 그 손잡이: Chapter 5의 \\(C\\)

앞 장(Chapter 5)의 소프트 마진 SVM[^cortespapnikas95]은 사실 이 장의 아이디어를 이미 쓰고 있었다.
목적함수 \\(\frac{1}{2}\\|w\\|^2 + C\sum\_{i}\xi^{(i)}\\)는 "가중치 \\(w\\)를
작게 유지하라"는 항과 "제약 위반을 허용하되 벌을 물린다"는 항의 합인데,
\\(\\|w\\|^2\\) 자체가 L2 페널티이며 \\(C\\)는 이 장의 \\(\\lambda\\)와
정확히 같은 종류의 손잡이다. 즉 Chapter 5에서는 정규화의 이름을 모른 채
그 손잡이를 돌리고 있었던 셈이다. 다만 Chapter 5가 보여준 것은 "그런 손잡이가
있고, 돌리면 모델이 바뀐다"는 사실뿐이었고, **손잡이의 크기를 앞의 데이터만으로
어떻게 정할 것인가**라는 질문에는 답하지 않았다. 이번 장은 바로 그 질문에 답한다.

그리고 다음 장(트리 기반 모델: 결정트리에서 GBDT[^gbdt]까지)에서 트리의 복잡도를
조절하는 손잡이는 가중치 페널티가 아니라 "트리 깊이", "분할 횟수" 같은 것들이
된다. 하지만 "수많은 후보 중 어떤 설정이 최적인가?"라는 질문은 똑같이 남고,
그 답을 만드는 틀 — 검증, 교차검증, train/val/test 분리 — 이 바로 이 장에서
짓는 것이다. 이번 장은 다음 장의 트리·앙상블 튜닝이 올라설 초석이라고
생각하면 된다.

## 학습 목표

이 챕터를 마치면 다음과 같은 일이 가능해진다:

- "가중치가 작다 = 모델이 단순하다 = 분산이 작다"는 연결고리를 설명할 수
  있고, L2 페널티가 매끄러운 축소(shrinkage)를, L1 페널티가 중요하지 않은
  가중치를 **정확히 0**으로 만들어 특징 선택까지 해내는 이유를 짚을 수 있다.
- \\(\\lambda\\)를 편향-분산 트레이드오프 위를 이동시키는 "손잡이"로
  해석하고, 계수 경로(coefficient path)[^lars2004] 같은 숫자 실험으로
  그 효과를 확인할 수 있다.
- k-겹 교차검증으로 \\(\\lambda\\) 후보들을 검증 성능 기준으로 비교해
  최적값을 고를 수 있고, U자형 검증 곡선의 왼쪽을 과적합·오른쪽을
  과소적합으로 읽을 수 있다.
- 실제 데이터 파이프라인에 train/val/test 3분할 원칙을 적용할 수 있고,
  주어진 코드에 "정보 흐름의 역주행"(leakage)이 있는지를 진단할 수 있다.

## 세 수업 블록의 흐름

- [6.1 정규화: 편향-분산 복습과 L1/L2 페널티](chapter06/1.md) —
  Chapter 4.1의 편향-분산 복습으로 "가중치 작음 = 분산 작음"이라는
  메커니즘을 먼저 세운 뒤, 페널티의 모양이 달라(L2는 원, L1은 마름모)
  결과가 어떻게 갈리는지 — 매끄러운 축소 대 "정확히 0" — 대조한다.
  Lasso가 왜 0으로 "뭉개지는"지 0에서의 서브그래디언트 조건
  (soft-thresholding)과 기하학적 그림으로 증명하고, \\(\\lambda\\)를
  키우며 계수 경로를 기록해
  잡음 특징이 한 순간에 0으로 스냅되는 현상을 숫자로 확인한다.
- [6.2 교차검증: \\(\\lambda\\)를 데이터로 정한다](chapter06/2.md) —
  \\(\\lambda\\)는 학습으로 정해지는 \\(w\\)와 달리 학습 전에 값을
  정해줘야 하는 하이퍼파라미터이므로, 검증 데이터로 잴 수밖에 없다.
  아무것도 배우지 못하는 평균모델의 5-fold CV를 손으로 계산해 "CV 추정값은
  어떤 숫자다"라는 감각을 만든 뒤, k-겹 교차검증으로 \\(\\lambda\\) 후보를
  훑으며 검증 MSE가 실제로 U자(좌측 과적합, 우측 과소적합)를 그리는지
  Ridge 회귀로 확인한다.
- [6.3 train/val/test 분리 원칙 실전](chapter06/3.md) — 데이터를 세
  조각으로 나눠 각 조각의 역할을 정하는 원리를 세운다: train은 배운다, val은
  하이퍼파라미터를 고르고, test는 최종 점수만 **딱 한 번** 준다. "정보
  흐름의 방향"을 기준으로, test를 훔쳐보고 모델 고르기와 val을 생략한
  튜닝, 전처리 통계량을 전체 데이터로 fit하는 누수라는 세 가지 흔한
  위반이 각각 점수를 어떻게 부풀리는지 숫자로 추적한다.

이 장의 주제(정규화, 교차검증, 모델 선택)를 더 깊이 다루는 자료: [^cs229]

[^cs229]: Stanford CS229: Machine Learning, Lecture Notes. https://cs229.stanford.edu/main_notes.pdf
[^tibshirani1996]: Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B (Methodological)* 58(1), 267–284. — Lasso(L1 정규화)의 원 논문.
[^gbdt]: Friedman, J. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics 29(5), 1189–1232. — 그래디언트 부스팅(GBDT)의 원 논문.
[^cortespapnikas95]: Cortes, C., Vapnik, V. (1995). "Support-Vector Networks." Machine Learning 20(3), 273–297. — 소프트 마진 SVM(여유 변수 + \\(C\\) 트레이드오프)의 원 논문.
[^lars2004]: Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). "Least Angle Regression." *Annals of Statistics* 32(2), 407–451. — LARS 알고리즘 원 논문. LASSO 계수 경로(coefficient path)를 O(p²)로 효율적으로 계산하는 방법을 제시한다.
