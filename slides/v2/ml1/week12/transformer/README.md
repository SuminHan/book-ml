# 트랜스포머

- 원본 쇼츠: [`lessons/ml1/transformer/`](../../../../../lessons/ml1/transformer/README.md)
- 구성: ChatGPT 의 T → RNN 5걸음 vs 한 번에 (n² 계산) → 셀프 어텐션 W_Q · W_K · W_V, 멀티헤드 8 × 64 → 위치 정보 sin · cos → 블록 (어텐션 → 피드포워드, 지름길 · LayerNorm) → 가중치 12d² (GPT-3 ≈ 174B) → GPT 앞만 보기 마스크 · BERT / GPT → 논문 BLEU 28.4 · 41.8 → 셰익스피어 RNN 1.74 / LSTM 1.64 / 트랜스포머 1.61 (위치 뺌 1.68) → BERT · GPT-3 · ViT + 문제풀이
- 상태: 14쪽. exp_tf.py → assets/tf.json (tiny Shakespeare 는 assets/ 에 복사)
- 실습: 위치 정보 · 마스크 · 블록 직접 쌓기 (12d² 비교) · 작은 GPT 1,000걸음 (검증 1.70) · 글 생성
