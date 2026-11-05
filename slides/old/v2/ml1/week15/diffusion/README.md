# Diffusion 확산 모델

- 원본 쇼츠: [`lessons/ml1/diffusion/`](../../../../../lessons/ml1/diffusion/README.md)
- 구성: 노이즈에서 고양이 (Stable Diffusion 50단계) → 앞으로 섞기 1,000번 → 식 x_t (t = 500: 0.28 : 0.96) → 학습 = 노이즈 맞히기 (DDPM 2020) → 뒤로 조금씩 지우기 → 숫자 실험 확신 VAE 31% · GAN 36% · 확산 55% → DDIM 50단계 약 25배 빠르게 (49%) → Stable Diffusion (48분의 1 · CLIP · 어텐션 · guidance) → 역사 + 문제풀이 + 이미지 출처
- 상태: 14쪽. exp_ddpm.py → assets/ddpm.json · ddpm.npz, assets/sd/ 는 쇼츠에서 복사
- 실습: 스케줄 · 섞기 · DDPM 30,000걸음 · DDPM vs DDIM 생성 판정
