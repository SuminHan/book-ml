# t-SNE 데이터 지도

- 원본 쇼츠: [`lessons/ml1/tsne/`](../../../../../lessons/ml1/tsne/README.md)
- 구성: 64차원 숫자 1,797장 → PCA 지도 60% vs t-SNE 98% (5-NN) → 이웃 관계 지키기 (2008) → 식 p (정규분포 · perplexity) · q (t 분포) · KL → 손 계산 거리 1 · 2 · 3 (0.81 · 0.18 · 0.01 vs 0.625 · 0.25 · 0.125), crowding → perplexity 5 · 30 · 100 → 주의: 섬 크기 (10.5배 → 1.2배) · 배치 (시드) → 쓰임 · UMAP + 문제풀이
- 상태: 13쪽. assets/emb.npz (쇼츠 make_emb.py 결과 복사) · exp_tsne.py → assets/extra.npz
- 실습: 유사도 손 계산 · KL · 숫자 PCA vs t-SNE 지도 · perplexity · 크기가 다른 두 무리
