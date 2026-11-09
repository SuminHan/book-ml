# 데이터 직접 다루기 (1주 3차시)

- 원본 쇼츠: [`lessons/ml1/dirty_data/`](../../../../../lessons/ml1/dirty_data/README.md) — 더티 데이터의 세 가지 모양 · [`lessons/ml1/feature_scaling/`](../../../../../lessons/ml1/feature_scaling/README.md) — 와인 그대로 72% vs 정규화 98%
- 옛 덱 `week01/dirty_data` + `week01/feature_scaling` 을 합친 것 (옛 폴더는 `week01/_이전/`)
- 구성: ① 더티 데이터 (결측치 · 중복 · 이상치) → ② 정규화 · 표준화 (와인 뉴런 포화, Min-Max · Z-score, 데이터 종류별 "무엇마다 맞추나") → 문제풀이 → 실습
- 상태: 덱 30쪽 (표지 + 29) · 실습 파트 8개
  - 1 더티 iris 청소 86% → 96% · 2 와인 72% → 98% · 3 MNIST uint8 · ÷255 (63 → 90 → 92%) · 4 CIFAR-10 채널별
  - 5 SMS 표기 15,733 → 8,713 · 6 항공 승객 창마다 표준화 (kNN 오차 74 → 10천 명) · 7 숫자 말소리 피크 정규화 · dB · 8 말 GIF 프레임마다 vs 영상 전체
- 숫자: `data.py` (반 10명 키 + `assets/scale.json` · `assets/modal.json`), `exp_scale.py` · `exp_modal.py` 가 json 을 만든다 (`lab/data/` 사용)
