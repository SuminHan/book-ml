"""차원의 저주 실험 → assets/dim.json
(1) 거리 비율: [0,1]^d 균등 난수 1,000개와 질의점 1개, 가장 가까운 거리 / 가장 먼 거리. 20번 평균 (시드 0)
(2) kNN(k=7): 두 반달(make_moons, 잡음 0.3) 600개 + 잡음 특징(균등 난수, 같은 범위)을 0·10·100·1,000개 붙여 검증 정확도 (학습 400 · 검증 200), 시드 0~19 평균"""
import json, os, numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
r = np.random.default_rng(0)
DS = [1, 2, 3, 10, 30, 100, 300, 1000]
ratio = {}
for d in DS:
    v = []
    for _ in range(20):
        X = r.random((1000, d)); q = r.random(d); dist = np.linalg.norm(X - q, axis=1); v.append(dist.min() / dist.max())
    ratio[d] = float(np.mean(v))
NOISE = [0, 10, 100, 1000]
acc = {}
for k in NOISE:
    v = []
    for s in range(20):
        X, y = make_moons(600, noise=0.3, random_state=s)
        rr = np.random.default_rng(100 + s)
        X = np.c_[X, rr.uniform(X.min(), X.max(), (600, k))] if k else X
        X = StandardScaler().fit_transform(X)
        v.append(KNeighborsClassifier(7).fit(X[:400], y[:400]).score(X[400:], y[400:]))
    acc[k] = float(np.mean(v))
X2, y2 = make_moons(600, noise=0.3, random_state=0)
json.dump(dict(ratio=ratio, acc=acc, moons=X2[:400].tolist(), moons_y=y2[:400].tolist()),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "dim.json"), "w"))
print({d: round(v, 3) for d, v in ratio.items()}, {k: round(v, 3) for k, v in acc.items()})
