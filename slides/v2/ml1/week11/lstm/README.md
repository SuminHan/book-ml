# LSTM 장단기 메모리

- 원본 쇼츠: [`lessons/ml1/lstm/`](../../../../../lessons/ml1/lstm/README.md)
- 구성: 0.9¹⁰⁰ ≈ 0.003% → 셀 상태 컨베이어 (1997 · 잊는 문 2000) → 식 f · i · o · c̃ → 손 계산 (c 2.2, h 0.68) → ∂c/∂c = f, 0.999¹⁰⁰ ≈ 90% → 문의 뜻 (The cat … was) → 셰익스피어 RNN vs LSTM (검증 손실 1.71 → 1.61) → GRU → 쓰임 (Seq2Seq · 트랜스포머) + 문제풀이
- 상태: 13쪽. exp_char.py → assets/char_cmp.json (exp_memory.py 는 쓰지 않음: 짧은 기억 문제라 RNN 도 풀었다)
- 실습: numpy LSTM 한 칸 (손 계산 · nn.LSTMCell 과 같은 값) · RNN / LSTM / GRU 1,000걸음 (검증 1.77 / 1.70 / 1.70) · 가중치 수 · 글 생성
