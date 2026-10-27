# GBDT 와 SHAP

- 원본 쇼츠: [`lessons/ml1/gbdt_shap/`](../../../../../lessons/ml1/gbdt_shap/README.md)
- 구성: 26살 두 승객 96% vs 13% → 부스팅 (남은 오차를 다음 나무가) → 작은 나무 50그루 MSE 0.48→0.04 → F ← F + ηh → 타이타닉 GBDT (200그루, CV 80.7%) → RF vs GBDT → SHAP 순서별 오른 만큼 → 섀플리 값 (34.0 · 22.6 · 1.3, 합 96.0) → 모르는 정보 = 양쪽 길 섞기 (93 → 44%) → 두 승객 폭포 → 전체 평균 |몫| + 나이 → 주의 + 문제풀이
- 상태: 15쪽. 타이타닉 캐시 assets/titanic.csv (OpenML)
- 실습: 부스팅 직접 구현 · GBDT 학습률/나무 수/조기 종료 · 섀플리 값 직접 계산 + shap 확인
