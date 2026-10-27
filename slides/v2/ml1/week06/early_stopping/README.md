# Early Stopping 조기 종료

- 원본 쇼츠: [`lessons/ml1/early_stopping/`](../../../../../lessons/ml1/early_stopping/README.md)
- 구성: 300번 학습 곡선 → 실험 (digits 500/500, 정답 20% 바꿈) → 31번째 92% vs 300번째 83% → 틀린 정답 외움 2→86장 → 맞히던 그림 59장 틀림 → 조기 종료 + 참을성 (41번째 멈춤, 31번째로) → 알고리즘 → 참을성 표 → 학습 횟수도 손잡이 → 멈춘 시점도 검증을 본 결과 + 문제풀이
- 상태: 13쪽. 실험 결과 캐시 assets/digits_run2.npz (쇼츠와 같음)
- 실습: 학습 곡선 재현 · 참을성 조기 종료 구현 + 시험 데이터 90.2% vs 82.2% · 깨끗한 정답 + sklearn early_stopping
