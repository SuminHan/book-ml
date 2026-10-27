# VAE 변분 오토인코더

- 원본 쇼츠: [`lessons/ml1/vae/`](../../../../../lessons/ml1/vae/README.md)
- 구성: 새 손글씨를 만들려면? (AE 의 '아무 숫자' 문제) → AE 지도 평균 (1.1, 3.5) 제멋대로 vs VAE 평균 0 · 퍼짐 0.98 → 점 대신 구름 (Kingma & Welling 2013) → 손실 = 복원 + KL (μ 1, σ 0.5 → 0.82) → 재매개변수화 손 계산 (z = 0.8) → N(0,1) 에서 뽑기 AE 5종류 vs VAE 10종류 → 격자 · 0 → 1 걷기, 대가 복원 오차 1.85 vs 2.10 → 쓰임 + 문제풀이
- 상태: 12쪽. exp_vae.py → assets/vae.json · vae.npz, 걷기 그림은 쇼츠의 assets/vae_shorts.npz
- 실습: KL · 재매개변수화 · AE vs VAE 학습 · N(0,1) 뽑기 종류 수 · 격자
