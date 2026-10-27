"""슬라이드·그림·실습이 공유하는 숫자.
데이터: Zachary (1977) 가라테 클럽 (networkx karate_club_graph): 34명, 친구 관계 78개, 관장 (0번, Mr. Hi) 편 17명 · 회장 (33번, Officer) 편 17명.
모델: Kipf & Welling (2017, ICLR) 2층 GCN — 쇼츠 lessons/ml1/gcn 의 gcn.npz (시드 10개 모두 31/32, 틀린 사람 8번).
  논문 Cora (논문 2,708편 · 인용 5,429개 · 이름표 140개) 정확도 81.5%.
실험: exp_gcn.py → assets/layers.json (층 수 0 ~ 8).
손 계산: 4명 A–B, A–C, C–D, 값 A = +1 (관장 편), D = −1 (회장 편), B = C = 0, 나와 친구들의 평균."""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "assets", "gcn.npz"))
PRED, Y, POS, EDGES = D["pred"], D["y"], D["pos"], D["edges"]
N = len(Y)
L = json.load(open(os.path.join(HERE, "assets", "layers.json")))
LAYERS = [0, 1, 2, 3, 4, 6, 8]
LAYER_MEAN = {k: float(np.mean(L["layers"][str(k)])) for k in LAYERS}
HOPS = {int(k): v for k, v in L["hops"].items()}
REST = [i for i in range(N) if i not in (0, 33)]
WRONG = [i for i in REST if PRED[i] != Y[i]]
NB = {i: sorted({int(b) for a, b in EDGES if a == i} | {int(a) for a, b in EDGES if b == i}) for i in range(N)}

# 손 계산 (4명)
HAND_NAMES = ["A", "B", "C", "D"]
HAND_EDGES = [(0, 1), (0, 2), (2, 3)]
HAND_X = np.array([1.0, 0.0, 0.0, -1.0])
HA = np.eye(4)
for a, b in HAND_EDGES: HA[a, b] = HA[b, a] = 1
HAND_MEAN = HA / HA.sum(1, keepdims=True)          # 나와 친구들의 평균
HAND_H1 = HAND_MEAN @ HAND_X
HAND_H2 = HAND_MEAN @ HAND_H1
HAND_DEG = HA.sum(1)
HAND_SYM = HA / np.sqrt(np.outer(HAND_DEG, HAND_DEG))   # GCN 의 Â

if __name__ == "__main__":
    assert N == 34 and len(EDGES) == 78 and int(Y.sum()) == 17
    assert sum(PRED[i] == Y[i] for i in REST) == 31 and WRONG == [8]
    assert NB[8] == [0, 2, 30, 32, 33] and sum(Y[j] for j in NB[8]) == 3      # 친구 5명 중 3명이 회장 편
    assert LAYER_MEAN[0] == 16 and LAYER_MEAN[2] == 31 and round(LAYER_MEAN[1], 1) == 30.2 and round(LAYER_MEAN[8], 1) == 16.6
    assert sorted(set(HOPS.values())) == [0, 1, 2] and sum(v == 1 for v in HOPS.values()) == 29
    assert [round(v, 2) for v in HAND_H1] == [0.33, 0.5, 0.0, -0.5]
    assert [round(v, 2) for v in HAND_H2] == [0.28, 0.42, -0.06, -0.25]
    assert round(HAND_SYM[0, 1], 2) == 0.41                                    # 1 / √(3 · 2)
    print("data ok")
