"""슬라이드·그림·실습이 공유하는 숫자.
데이터: sklearn digits (손글씨 숫자 1,797장, 8×8 = 64칸). 지도: 쇼츠 lessons/ml1/tsne 의 emb.npz (PCA 2D, t-SNE perplexity 30 · 5)
  + exp_tsne.py → extra.npz (perplexity 100, 크기가 다른 두 무리, 다른 시드).
'지도에서 가까운 5개로 맞히기' = 2차원 좌표에서 5-NN, 5겹 교차검증 정확도.
논문: van der Maaten & Hinton (2008) JMLR 9 'Visualizing Data using t-SNE'. 주의점: Wattenberg, Viégas, Johnson (2016) Distill 'How to Use t-SNE Effectively'.
손 계산: 한 점에서 이웃 셋까지 거리 1 · 2 · 3, 고차원은 정규분포 (σ = 1), 지도는 t 분포 (자유도 1)."""
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
E = np.load(os.path.join(HERE, "assets", "emb.npz")); X2 = np.load(os.path.join(HERE, "assets", "extra.npz"))
Y, PCA2, TS30, TS5, IMG = E["y"], E["pca"], E["ts30"], E["ts5"], E["img0"]
TS100, TWO, TWO_LAB, SEED1, SEED2 = X2["ts100"], X2["two"], X2["two_lab"], X2["seed1"], X2["seed2"]
ORIG_RATIO, MAP_RATIO = float(X2["orig_ratio"]), float(X2["map_ratio"])


def knn_acc(Z, y=Y, k=5):
    return cross_val_score(KNeighborsClassifier(k), Z, y, cv=5).mean()


ACC = {"pca": 60.3, "ts5": 97.4, "ts30": 97.6, "ts100": 96.5, "seed1": 97.8, "seed2": 97.1}   # knn_acc × 100, 실행 결과 고정 (아래 assert 로 확인)
D = np.array([1.0, 2.0, 3.0])
GAUSS = np.exp(-D ** 2 / 2); P = GAUSS / GAUSS.sum()
TDIST = 1 / (1 + D ** 2); Q = TDIST / TDIST.sum()

if __name__ == "__main__":
    got = {k: round(float(knn_acc(Z)) * 100, 1) for k, Z in [("pca", PCA2), ("ts5", TS5), ("ts30", TS30), ("ts100", TS100), ("seed1", SEED1), ("seed2", SEED2)]}
    print(got)
    assert got == ACC, got
    assert [round(v, 3) for v in GAUSS] == [0.607, 0.135, 0.011] and [round(v, 2) for v in P] == [0.81, 0.18, 0.01]
    assert np.allclose(TDIST, [0.5, 0.2, 0.1]) and np.allclose(Q, [0.625, 0.25, 0.125])
    assert round(ORIG_RATIO, 1) == 10.5 and round(MAP_RATIO, 1) == 1.2
    print("data ok")
