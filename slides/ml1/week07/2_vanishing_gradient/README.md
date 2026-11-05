# 기울기 소실

- 원본 쇼츠: [`lessons/ml1/vanishing_gradient/`](../../../../../lessons/ml1/vanishing_gradient/README.md)
- 구성: MNIST 10층 시그모이드 10% vs ReLU 95% → 역전파는 곱한다 → σ' ≤ 0.25, 0.25¹⁰ → 층별 기울기 119만분의 1 → ReLU → 폭발 · 기울기 자르기 → 초기화 (0.01 · 1 · Xavier · He) → Xavier 1/n · He 2/n → 죽은 ReLU · Leaky · GELU + 문제풀이
- 상태: 12쪽. 실험값 assets/vg.json (쇼츠와 같음, exp_vg.py)
- 실습: 곱하기 · 초기화별 층 출력 · MNIST 10층 층별 기울기 · 1에폭 정답률 (sigmoid 9.8% · ReLU 94%)
