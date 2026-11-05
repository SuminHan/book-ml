# 정규화와 표준화

- 1주 2차시 · 원본 쇼츠: [`lessons/ml1/feature_scaling/`](../../../../../lessons/ml1/feature_scaling/README.md) — 와인 그대로 72% vs 정규화 98%
- 구성: 강의 20~30분 (쇼츠의 와인 · 뉴런 포화 장면 + 데이터 종류별 "무엇마다 맞추나") + 문제풀이 / Colab 실습
- 상태: 덱 24쪽 (표지 + 23) · 실습: 일곱 가지 데이터를 직접 열어 보기
  - 표 (UCI Wine): Min-Max · Z-score 구현, 72% → 98% 재현
  - 흑백 이미지 (MNIST): uint8 함정, ÷255, 63% → 90% → 92%
  - 컬러 이미지 (CIFAR-10 시험 1만 장): 채널별 평균 · 표준편차, (H, W, C) → (C, H, W)
  - 언어 (UCI SMS): 표기 맞추기 15,733 → 8,713, 단어 번호 · 임베딩, 길이로 나누기
  - 시계열 (항공 승객): 학습 기간 하나로 vs 12달 창마다 표준화, kNN 오차 74 → 10천 명
  - 소리 (Free Spoken Digit 6명): ÷32768, 음량 37배 → 피크 정규화, dB 스펙트로그램
  - 비디오 (Muybridge 말 GIF): 프레임마다 vs 영상 전체 표준화
- 숫자: `exp_scale.py` → `assets/scale.json` (쇼츠와 같은 와인 실험), `exp_modal.py` → `assets/modal.json` (데이터 종류별, `lab/data/` 사용)
