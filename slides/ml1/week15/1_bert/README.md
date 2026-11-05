# BERT 빈칸 채우기

- 원본 쇼츠: [`lessons/ml1/bert/`](../../../../../lessons/ml1/bert/README.md)
- 구성: I went to the ___ (bread → market 78% · doctor → hospital 81%) → 양방향 인코더 (GPT 와 비교) → 15% 빈칸 (80 / 10 / 10), 33억 단어 자기 지도 → 가중치 세기 2,344만 + 12 × 708만 = 1억 889만 → 실험 빈칸 (paris 41.7%, seoul 99.8%) → bank 문맥 벡터 (0.83 · 0.71 vs 0.42 ~ 0.54) → 사전학습 + 미세조정 (11개 과제, GLUE 80.5%) → 편향 (she → nurse 60.9%, he → doctor 28.7%) → BERT 이후 + 문제풀이
- 상태: 13쪽. exp_bert.py → assets/bert.json (HF 캐시 bert-base-uncased)
- 실습: 15% 가리기 직접 구현 · 실제 BERT 빈칸 · bank 문맥 벡터 · 가중치 수
