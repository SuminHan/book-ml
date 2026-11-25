# SVM 2: 소프트 마진

- 2026-11-25 SVM 을 세 덱으로 나눔: [1 선형](../2_svm/README.md) · **2 라그랑주 · 소프트 마진 (이 덱)** · [3 커널](../4_svm_kernel/README.md). 나누기 전 한 덱 = ../4_svm_kernel/_이전/v_세덱나누기전/
- 순서: 지난 시간 → 오늘의 질문 (겹치면?) → 식 다시 쓰기 (M‖w‖ = 1) → α 로 학습 (참고: 라그랑주 함수 일반형 · 라그랑주 · 안장점 · J · 학습 · SMO) → 소프트 마진 (α ≤ C) → α 의 뜻 → 정리 → 실습 → Appendix (소프트 w 학습 · ISL · 논문 따라 읽기 1–6) → 참고 문헌. 39쪽
- 표기: 원 논문 (Boser · Guyon · Vapnik 1992), 소프트 마진은 Cortes · Vapnik 1995
- 숫자 data.py (assert) · 그림 make_figs.py (공용 함수는 ../2_svm/make_figs.py) · 실습 lab/make_lab.py · 퀴즈 quiz/svm_soft_quiz(_solution).pdf (Q1–Q3)
