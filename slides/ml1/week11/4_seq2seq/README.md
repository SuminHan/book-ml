# Seq2Seq 시퀀스투시퀀스

- 원본 쇼츠: [`lessons/ml1/seq2seq/`](../../../../../lessons/ml1/seq2seq/README.md)
- 구성: I love you → Je t'aime (길이가 다르다) → 인코더 · 벡터 · 디코더 (2014) → 식 · 확률 곱 → teacher forcing vs greedy · 빔 서치 → 실험 숫자 말 → 숫자 (시험 99.5%, seventy → 77) → 논문 BLEU 34.8 vs 33.3, 8,000개 벡터 → 거꾸로 넣기 25.9 → 30.6 → 실험 베끼기 길이 5 93% → 30 24% (병목) → 어텐션 예고 + 문제풀이
- 상태: 13쪽. exp_seq2seq.py → assets/s2s.json
- 실습: 영어 숫자 말 함수 · GRU 인코더-디코더 학습/번역 (199/200) · 숫자열 베끼기 길이별 정답률 그대로 vs 거꾸로
