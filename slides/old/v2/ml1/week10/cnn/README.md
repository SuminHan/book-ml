# CNN 합성곱 신경망

- 원본 쇼츠: [`lessons/ml1/cnn/`](../../../../../lessons/ml1/cnn/README.md)
- 구성: 125만 가중치 → 완전연결 두 문제 → 3×3 합성곱 손 계산 3.0 → 특징 지도 (가로 · 세로) · 파라미터 공유 → 출력 크기 식 (26 · 28 · 14) → 채널 · 가중치 (40+296+1,168) → 1,504 vs 1,256,000 + 실제 학습 (CNN 5,258개 97.5% vs MLP 101,770개 96.4%) → 최대 풀링 → 구조 (LeNet · AlexNet) → 학습된 필터 + 문제풀이
- 상태: 13쪽. exp_cnn.py → assets/cnn.json · filters.npy
- 실습: 합성곱 · 풀링 구현 · 출력 크기/가중치 vs PyTorch · CNN vs MLP + 필터
