"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/random_forest 와 같은 이야기: 골턴의 황소 · 붓꽃 61 대 39).
골턴: F. Galton, "Vox Populi", Nature 75 (1907) — 표 787장, 가운데 값 1,207 파운드, 실제 1,198 파운드.
붓꽃 (assets/iris.csv) · Pima 당뇨병 768명 (assets/pima.csv). 랜덤 포레스트는 scikit-learn, random_state=0."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GALTON = (787, 1207, 1198)
CARDS = [9, 7, 6, 3, 4, 1, 1, 1, 2, 9]                       # 카드 1~10 에서 10번 복원 추출 (시드 0): 7종류, 5 · 8 · 10 은 안 뽑힘
BOOT_UNIQUE = 0.632                                          # 1 − 1/e
FLOWER, VOTES = (5.0, 1.7), (61, 39)                         # 꽃잎 길이 · 너비 (cm) → versicolor 61 · virginica 39
PIMA = {"tree": 0.686, "tree_d2": 0.754, "forest300": 0.757, "oob": 0.758}
NTREES = {1: 0.685, 3: 0.728, 10: 0.744, 30: 0.758, 100: 0.744, 300: 0.757}
TOP3 = [("glucose", 0.252), ("bmi", 0.168), ("age", 0.139)]
PIMA_COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "pedigree", "age"]


def pima():
    d = pd.read_csv(os.path.join(HERE, "assets", "pima.csv"), header=None)
    return d.iloc[:, :8].to_numpy(), d[8].to_numpy()


def iris():
    d = pd.read_csv(os.path.join(HERE, "assets", "iris.csv"))
    return d[["petal_length", "petal_width"]].to_numpy(), d["species"].to_numpy()


def avg_error_sd(n, rho, sd=1.0):
    """n 명의 답을 평균 냈을 때 오차의 표준편차 (각자 오차 sd, 서로 상관 rho)"""
    return sd * np.sqrt(rho + (1 - rho) / n)


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    assert np.random.default_rng(0).integers(1, 11, 10).tolist() == CARDS and len(set(CARDS)) == 7
    r = np.random.default_rng(0); assert round(np.mean([len(np.unique(r.integers(0, 768, 768))) / 768 for _ in range(200)]), 3) == BOOT_UNIQUE
    Xi, yi = iris(); rf = RandomForestClassifier(100, random_state=0).fit(Xi, yi)
    v = np.array([t.predict([FLOWER])[0] for t in rf.estimators_])
    assert ((v == 1).sum(), (v == 2).sum()) == VOTES
    X, y = pima(); cv = StratifiedKFold(5, shuffle=True, random_state=0)
    got = {"tree": cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=cv).mean(),
           "tree_d2": cross_val_score(DecisionTreeClassifier(max_depth=2, random_state=0), X, y, cv=cv).mean(),
           "forest300": cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, y, cv=cv).mean(),
           "oob": RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1).fit(X, y).oob_score_}
    assert {k: round(v, 3) for k, v in got.items()} == PIMA
    assert {n: round(cross_val_score(RandomForestClassifier(n, random_state=0, n_jobs=-1), X, y, cv=cv).mean(), 3) for n in NTREES} == NTREES
    imp = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X, y).feature_importances_
    assert sorted(zip(PIMA_COLS, imp.round(3)), key=lambda t: -t[1])[:3] == TOP3
    print("data ok")
