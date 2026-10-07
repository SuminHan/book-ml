# 출처
- 슬라이드: week02 p65(섀넌 정보량 −log₂p) · p66(엔트로피 · 교차 엔트로피 · KL = H(p,q) − H(p)) · week15 p09(VAE ELBO 의 KL 정규화 항) · week17 p48(RLHF/PPO 목적함수의 β·KL 벌점) · p49(KL 항 없으면 보상 해킹, Gao et al. 2022)
- 이름: S. Kullback & R. A. Leibler, "On Information and Sufficiency", Ann. Math. Stat. 1951
- 예시 데이터(v4, 현재): 게임 뽑기 공지 당첨 20 · 꽝 80 vs 실제 4 · 96 (특정 게임 아님; 3등급판 v4a 는 data_v4a.py), 사진 분류 예측 70% → 95% — 모두 설명용 예시
  - 이전 판: v1 한국 vs 브라질 · v2 날씨 앱 · v3 대출 심사 (data_v1~v3.py, lines_v1~v3.json, shorts_v1~v3.py)
- 계산(assert, 비트): src/data.py — 평균 뽑기 5 vs 25(1/p), 놀라움 −log₂: 4.64·0.06 / 2.32·0.32, H 0.24, H(p,q) 0.40, KL 0.16, 공지 = 실제면 0, 정답 0/1 이면 교차 엔트로피 = KL (0.51 → 0.07)
- s9 의 로봇 사이 'KL' 글자 크기는 개념 그림(수치 아님)
- 이모지: Twemoji (CC BY 4.0) — 🎁 💨 🐱 🐶 🤖 (v3: 사람 이모지)
