# BatchNorm 배치 정규화

- 원본 쇼츠: [`lessons/ml1/batchnorm/`](../../../../../lessons/ml1/batchnorm/README.md)
- 구성: 시그모이드 8층 10% vs BN 250번 96% → 층마다 그림 사이 퍼짐 0.15 → 1.8e-7 → 식 (μ, σ², x̂, γ, β) → 손 계산 2·4·6·8 → γ·β 이유 → 효과 (논문 14배, 설명 논쟁) → 시험 때 running mean · eval() → 위치 · LayerNorm + 문제풀이
- 상태: 11쪽. 학습 곡선 assets/bn.npz (쇼츠와 같음, exp_bn.py)
- 실습: BN 직접 구현 vs nn.BatchNorm1d · 층별 퍼짐 + 학습 곡선 · eval 과 그림 한 장
