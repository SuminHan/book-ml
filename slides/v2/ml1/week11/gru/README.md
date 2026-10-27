# GRU 게이트 순환 유닛

- 원본 쇼츠: 없음 (LSTM 덱의 GRU 한 쪽을 덱 하나로 늘림). 덱 tex 는 `slides/v2_cs/ml1/week11/3_gru/gru.tex` 에만 있다 (이 폴더는 그림 · 숫자 · 실습 원본)
- 구성: 문 3개 · 기억 2개 다 필요할까? → GRU 그림 (z 지키기 · r 리셋) → 식 (원 논문 · PyTorch 방향, z = 지키는 비율) → 손 계산 갱신 문 (0.8, z 0.9, h~ -0.2 → 0.70) · 리셋 문 (0.97 vs 0.46) → LSTM 대응표 → z ≈ 1 이면 0.999¹⁰⁰ ≈ 90% → 가중치 3묶음 (99,072 = LSTM 의 ¾) → 셰익스피어 RNN / LSTM / GRU (검증 1.71 / 1.61 / 1.61) → 무엇을 쓸까 (Chung 2014, Jozefowicz 2015) + 문제풀이
- 상태: 13쪽. exp_char.py → assets/char_gru.json (RNN · LSTM 은 ../lstm/assets/char_cmp.json, 같은 조건)
- 실습: 손 계산 · numpy GRU 한 칸 (nn.GRUCell 과 같은 값) · LSTM vs GRU 1,000걸음 (검증 1.70 / 1.70, 가중치 ¾) · 갱신 문 z 글자별 평균 · 글 생성
