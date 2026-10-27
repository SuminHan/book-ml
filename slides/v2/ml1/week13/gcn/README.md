# GCN 그래프 신경망

- 원본 쇼츠: [`lessons/ml1/gcn/`](../../../../../lessons/ml1/gcn/README.md)
- 구성: 가라테 동아리 34명 · 78개, 이름표 2명 → 그래프 = 노드 + 엣지 → 나와 친구들의 평균 (4명 손 계산 0.5 · 0 → −0.06) → 식 Â = D^-1/2 (A+I) D^-1/2 (Kipf & Welling 2017) → CNN 과 비교 → 실험 31 / 32 (틀린 8번: 친구 5명 중 3명) → 층 수 0층 16 · 2층 31 · 8층 16.6 (과평활) → 쓰임 (Cora 81.5%, DeepFRI, STGCN, PinSage) + 문제풀이
- 상태: 12쪽. assets/gcn.npz (쇼츠 train_gcn.py 결과 복사) · exp_gcn.py → assets/layers.json
- 실습: 4명 평균 · Â · 가라테 2층 GCN (31/32, 가중치 594) · 층 수 · 두 점수 그림
