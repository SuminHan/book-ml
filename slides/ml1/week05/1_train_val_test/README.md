# train / val / test 와 데이터 누수

- 원본 쇼츠: [`lessons/ml1/data_leakage/`](../../../../../lessons/ml1/data_leakage/README.md)
- 원본: notebooks/ml1/chapter06_3 (E[max], 표준화 누수, 환자 · 시간 누수, diabetes 3분할) + 쇼츠 lessons/ml1/data_leakage 의 동전 실험 (87% vs 49%)
- 구성: 동전 87% → 세 데이터 역할 → test 로 고르면 +1.16σ/+2.25σ → 누수 정의 → 특징 고르기 87 vs 49 → 표준화 z → 환자 1.04 vs 3.65 → 시간 0.79 vs 44.1 → 규칙 → diabetes 265/88/89, λ 0.1, test 53.79 + 문제풀이
- 상태: 13쪽
- 실습: 동전 실험 SelectKBest 누수 vs Pipeline · GroupKFold · TimeSeriesSplit · diabetes 3분할
