# Dropout 드롭아웃

- 원본 쇼츠: [`lessons/ml1/dropout/`](../../../../../lessons/ml1/dropout/README.md)
- 구성: 뉴런 절반 끄기 (논문 160→130→110) → 2,000장 · 67만 가중치 외우기 (770 오답) → 배치마다 무작위로 끄기 → 짝꿍 · 앙상블 2¹²=4096 → 시험 때 크기 맞추기 (inverted, ×2) → 우리 실험 770→726 → p 별 (663 · 726 · 680) → train()/eval() + 문제풀이
- 상태: 12쪽. 실험 exp_dropout.py → assets/dropout.json (MNIST 2,000장, 100 에폭)
- 실습: 드롭아웃 직접 구현 · MNIST 2,000장 p=0 vs 0.5 (30 에폭) · eval() 깜빡하기
