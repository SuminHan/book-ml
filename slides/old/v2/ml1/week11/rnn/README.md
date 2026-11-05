# RNN 순환 신경망

- 원본 쇼츠: [`lessons/ml1/rnn/`](../../../../../lessons/ml1/rnn/README.md)
- 구성: Dog bites man vs Man bites dog → 더하면 같다 → RNN 식 · 같은 가중치 → 펼치기 → 손 계산 (0.762 · 0.363 · 0.828) → 두 문장 마지막 메모 다름 → 셰익스피어 글자 언어 모델 (손실 4.18 → 1.58) → 생성 글 200 vs 2,000걸음 → BPTT → 기억 감쇠 0.5¹⁰ → 쓰임 + 문제풀이
- 상태: 14쪽. exp_charrnn.py → assets/charrnn.json (tiny Shakespeare 는 assets/ 에 복사)
- 실습: numpy RNN 한 칸 · nn.RNN 글자 언어 모델 1,000걸음 · 온도 0.5/1.0/1.5 생성
