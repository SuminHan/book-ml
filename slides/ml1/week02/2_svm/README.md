# 서포트 벡터 머신

- 원본 쇼츠: [`lessons/ml1/svm/`](../../../../../lessons/ml1/svm/README.md)
- 구성 (2026-11-16, CME 250 Lecture 5 흐름): 초평면 (1 + 2x1 + 3x2) → 분리 초평면 → 최대 마진 (M 1.59, 폭 3.18) · 서포트 벡터 · 값과 거리 → max M s.t. ‖β‖ = 1 (β = (−0.71, 0.71), β0 = −0.18) → w = β/M, min ½‖w‖² → 학습 (힌지 손실 경사하강, 2만 걸음, 176 · 278 · 7567걸음) → 약점 (겹침 · 점 하나 M 0.46) → 슬랙 ε · 예산 B (0 · 1.87 · 2.64, sklearn C 는 반대) → 특징 늘리기 → 점곱 꼴 α (8/81, 8/81, −16/81) → 커널 · 25 = 5² → 선형 · 다항 · RBF (56 · 99 · 99%) → γ → 장단점 → 일대일 · 일대다 → 문제풀이
- 참고: S. Wang, CME 250 Lecture 5: Support Vector Machines (Stanford, Winter 2019) https://web.stanford.edu/class/cme250/files/cme250_lecture5.pdf · ISL 9장 · Goodfellow et al. Deep Learning 5.7.2. 그림은 모두 직접 그림 (ISL 그림 안 씀)
- 상태: 28쪽. 점 20개는 손으로 고른 예시. 이전 자체 구성판 = _이전/svm_v1_자체구성.tex · .pdf
- 실습: 하드 마진·먼 점·소프트 마진 · Banknote 선형 98.6% vs RBF 100% · 반달 GridSearchCV (C, γ)
