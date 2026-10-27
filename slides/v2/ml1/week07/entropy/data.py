"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/entropy 와 같은 축구 예시 + 결정 트리의 정보 이득).
축구 확률은 예시 (한국 승 10% · 브라질 승 90%, 무승부 없음). 붓꽃: assets/iris.csv (Fisher 1936)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def surprise(p):
    return float(-np.log2(p))


def entropy(ps):
    ps = np.asarray(ps, float); ps = ps[ps > 0]
    return float(-(ps * np.log2(ps)).sum())


def label_entropy(labels):
    _, c = np.unique(labels, return_counts=True); return entropy(c / c.sum())


SURPRISE = {0.9: 0.15, 0.1: 3.32, 0.5: 1.0, 0.01: 6.64, 0.99: 0.01}
H_KB, H_EARLY, H_KJ = 0.47, 0.08, 1.0                         # 한국-브라질 · 조기축구-브라질 (1%) · 한일전 (반반)
H3, H6, H_TREE = 1.585, 2.585, 1.75                           # 3가지 같은 확률 · 주사위 · (1/2, 1/4, 1/8, 1/8)
IRIS = {"root": 1.585, "length_ig": 0.918, "width_ig": 0.686, "second_ig": 0.69}

if __name__ == "__main__":
    assert {p: round(surprise(p), 2) for p in SURPRISE} == SURPRISE
    assert (round(entropy([.1, .9]), 2), round(entropy([.01, .99]), 2), entropy([.5, .5])) == (H_KB, H_EARLY, H_KJ)
    assert (round(entropy([1 / 3] * 3), 3), round(entropy([1 / 6] * 6), 3), entropy([.5, .25, .125, .125])) == (H3, H6, H_TREE)
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "assets", "iris.csv")); y = df["species"].to_numpy(); X = df[["petal_length", "petal_width"]].to_numpy()
    def ig(mask, lab): return label_entropy(lab) - (mask.mean() * label_entropy(lab[mask]) + (~mask).mean() * label_entropy(lab[~mask]))
    R = X[:, 0] >= 2.45
    got = {"root": round(label_entropy(y), 3), "length_ig": round(ig(X[:, 0] < 2.45, y), 3), "width_ig": round(ig(X[:, 1] < 1.75, y), 3),
           "second_ig": round(ig(X[R, 1] < 1.75, y[R]), 2)}
    assert got == IRIS, got
    print("data ok")
