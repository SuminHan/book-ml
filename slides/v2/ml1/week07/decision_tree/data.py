"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/decision_tree 와 같은 붓꽃 예시).
붓꽃 150송이 (assets/iris.csv, Fisher 1936; seaborn-data 사본), 특징은 꽃잎 길이 · 너비 (cm) 두 개. 불순도 = 지니 (Gini).
깊이와 과적합 장면은 섞인 데이터가 더 잘 보여 주는 Pima 당뇨병 768명 (assets/pima.csv) 으로 잰다."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def iris():
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "assets", "iris.csv"))
    return df[["petal_length", "petal_width"]].to_numpy(), df["species"].to_numpy()


def gini(labels):
    _, c = np.unique(labels, return_counts=True); p = c / c.sum()
    return float(1 - (p ** 2).sum())


def split_gini(X, y, f, t, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    L, R = idx[X[idx, f] < t], idx[X[idx, f] >= t]
    return L, R, (len(L) * gini(y[L]) + len(R) * gini(y[R])) / len(idx)


ROOT = 0.667                                         # 3종 50송이씩
CAND = {("length", 2.45): (50, 100, 0.333), ("width", 1.75): (104, 46, 0.391)}
WIDTH_LEFT_GINI = 0.545                              # 뿌리에서 너비 < 1.75 의 왼쪽 104송이 (50 · 49 · 5)   # 뿌리 후보: (왼쪽 수, 오른쪽 수, 가중 지니)
SECOND = {"left": (49, 5, 0.168), "right": (1, 45, 0.043), "weighted": 0.11}     # 남은 100송이를 꽃잎 너비 1.75 로
PIMA_DEPTH = {2: (0.772, 0.754, 4), None: (1.0, 0.686, 130)}                      # 깊이: (학습 정확도, 5겹 CV, 잎 수)
PIMA_SWEEP = {1: 0.727, 2: 0.754, 3: 0.745, 4: 0.736, 5: 0.737, 6: 0.72, 8: 0.699, 10: 0.684, 15: 0.684}

if __name__ == "__main__":
    X, y = iris()
    assert round(gini(y), 3) == ROOT
    for (f, t), (nl, nr, g) in CAND.items():
        L, R, w = split_gini(X, y, 0 if f == "length" else 1, t); assert (len(L), len(R), round(w, 3)) == (nl, nr, g)
    assert round(gini(y[X[:, 1] < 1.75]), 3) == WIDTH_LEFT_GINI
    rest = np.where(X[:, 0] >= 2.45)[0]
    L, R, w = split_gini(X, y, 1, 1.75, rest)
    assert ((y[L] == "versicolor").sum(), (y[L] == "virginica").sum(), round(gini(y[L]), 3)) == SECOND["left"]
    assert ((y[R] == "versicolor").sum(), (y[R] == "virginica").sum(), round(gini(y[R]), 3)) == SECOND["right"] and round(w, 3) == SECOND["weighted"]
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    d = pd.read_csv(os.path.join(HERE, "assets", "pima.csv"), header=None); Xp, yp = d.iloc[:, :8].to_numpy(), d[8].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for dep, (tr, c, nl) in PIMA_DEPTH.items():
        m = DecisionTreeClassifier(max_depth=dep, random_state=0).fit(Xp, yp)
        assert (round(m.score(Xp, yp), 3), round(cross_val_score(DecisionTreeClassifier(max_depth=dep, random_state=0), Xp, yp, cv=cv).mean(), 3), m.get_n_leaves()) == (tr, c, nl)
    assert {k: round(cross_val_score(DecisionTreeClassifier(max_depth=k, random_state=0), Xp, yp, cv=cv).mean(), 3) for k in PIMA_SWEEP} == PIMA_SWEEP
    print("data ok")
