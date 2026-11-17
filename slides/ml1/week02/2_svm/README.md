# 서포트 벡터 머신

- 원본 쇼츠: [`lessons/ml1/svm/`](../../../../../lessons/ml1/svm/README.md)
- 구성 (2026-11-17, 기준은 ‖w‖ = 1 하나): 질문 → 두 평균의 한가운데 → SVM 은 가장 가까운 점 기준 (1.41 / 1.71 → 1.59) → 로지스틱 vs SVM → 초평면 (1 + 2x1 + 3x2) → 분리 초평면 → 최대 마진 (m 1.59, 폭 3.18) · 서포트 벡터 → find w, b that maximize m > 0, ‖w‖ = 1 (w = (−0.71, 0.71), b = −0.18) → 참고: 점곱 · 코사인 → 학습 (가장 가까운 점이 밀고 길이 1 로, 학습률 0.05/√t · b 5배, 1걸음 = 평균 방향 (−0.75, 0.66), 3걸음 (7.8, 5) 되밀기, 2만 걸음 m 1.58) → 걸음별 · m 곡선 · 애니메이션 2개 (0걸음부터 · 엉뚱한 선에서) → 약점 (겹침 · 점 하나 m 0.46) → 슬랙 ε · 예산 B (0 · 1.87 · 2.64, sklearn C 는 반대) → 특징 늘리기 → 점곱 꼴 α (0.16, 0.16, −0.31) → 커널 · 25 = 5² → 선형 · 다항 · RBF (56 · 99 · 99%) → γ → 장단점 → 일대일 · 일대다 → 문제풀이
- 참고: S. Wang, CME 250 Lecture 5: Support Vector Machines (Stanford, Winter 2019) https://web.stanford.edu/class/cme250/files/cme250_lecture5.pdf · ISL 9장 · Goodfellow et al. Deep Learning 5.7.2. 그림은 모두 직접 그림 (ISL 그림 안 씀)
- 상태: 28쪽. 점 20개는 손으로 고른 예시. 이전 자체 구성판 = _이전/svm_v1_자체구성.tex · .pdf
- 실습: 하드 마진·먼 점·소프트 마진 · Banknote 선형 98.6% vs RBF 100% · 반달 GridSearchCV (C, γ)
