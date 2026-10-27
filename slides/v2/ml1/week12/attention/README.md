# 어텐션

- 원본 쇼츠: [`lessons/ml1/attention/`](../../../../../lessons/ml1/attention/README.md)
- 구성: 그것은 너무 피곤했다 → Seq2Seq 벡터 하나 vs 모든 칸 (Bahdanau 2014) → 도서관 비유 Q · K · V → 식 softmax(QKᵀ/√d)V → 손 계산 (4.85 → 56% · 43%, 새 벡터 (2.88, 0.15, 0.01)) → 왜 √d (1.7 vs 22.6) → 실험 베끼기 길이 30 24% → 97%, 길이 50 88% → 어텐션 지도 대각선 → 2014 · 2015 · 2017 셀프 어텐션 + 문제풀이
- 상태: 13쪽. exp_attention.py → assets/attn.json
- 실습: numpy 어텐션 · √d 실험 · PyTorch Seq2Seq + 어텐션 베끼기 길이별 · 어텐션 지도
