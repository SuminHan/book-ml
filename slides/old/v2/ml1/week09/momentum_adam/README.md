# 모멘텀과 Adam

- 원본 쇼츠: [`lessons/ml1/momentum_adam/`](../../../../../lessons/ml1/momentum_adam/README.md)
- 구성: 길쭉한 골짜기 경사하강 111걸음 → 왜 지그재그 (학습률 < 0.04) → 모멘텀 식 · v 1→1.9→2.71 → 33걸음 → Adam 식 (m, s, 보정) → 27걸음 · 세 방법 비교 → 기본값 · AdamW → digits 실제 (SGD 0.817 · 모멘텀 0.066 · Adam 0.186) → 무엇을 쓸까 + 문제풀이
- 상태: 12쪽. 경로 assets/paths.npz (쇼츠와 같음, exp_paths.py) · 실제 비교 exp_digits.py
- 실습: 세 방법 numpy 구현 · 학습률 격자 · torch.optim 비교
