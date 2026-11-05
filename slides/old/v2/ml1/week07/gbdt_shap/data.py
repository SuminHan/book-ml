"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/gbdt_shap 과 같은 계산: 타이타닉 GBDT + SHAP).
타이타닉 승객 1,309명 (OpenML titanic v1, 원자료 Vanderbilt titanic3). 특징 5개: 등급 · 여성 · 나이 · 요금 · 가족 수.
GBDT: 나무 200그루, 깊이 3, 학습률 0.05. 섀플리 값은 '성별 · 등급 · 나머지(나이 · 요금 · 가족)' 3명 팀으로 정확히 계산 (확률 %).
부스팅 장난감 예: x 40개 (0~10), y = sin x + 잡음 (시드 1), 깊이 1 나무, 학습률 0.3.
표 데이터에선 왜 나무가 이길까 (Grinsztajn, Oyallon & Varoquaux 2022, NeurIPS D&B, arXiv 2207.08815 의 5장 세 가지 이유):
  배달비 구간 (예시) 0~2km 3,000원 · 2~4km 4,000원 · 4~6km 5,500원, 주문 30건 (0.1, 0.3, ..., 5.9 km).
  나무 = GradientBoostingRegressor (100그루, 깊이 2) · 신경망 = MLPRegressor (32-32 ReLU) · 구간 = 0.3km 칸 20개 one-hot.
  쓸모없는 열: 주문 400건 (학습 300 · 시험 100, 시드 0), 쓸모 있는 열 3개 (거리 · 시각 · 비) + 무작위 열 k 개.
  캐글: Chen & Guestrin 2016 (KDD, arXiv 1603.02754) 1장 — 2015년 캐글 블로그 우승 해법 29개 중 XGBoost 17개, 딥러닝 11개."""
import os
from itertools import combinations, permutations
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "assets", "titanic.csv")


def titanic():
    if not os.path.exists(CACHE):
        from sklearn.datasets import fetch_openml
        f = fetch_openml("titanic", version=1, as_frame=True).frame
        X = pd.DataFrame({"pclass": f.pclass.astype(int), "female": (f.sex == "female").astype(int),
                          "age": f.age.astype(float).fillna(f.age.astype(float).median()),
                          "fare": f.fare.astype(float).fillna(f.fare.astype(float).median()),
                          "family": f.sibsp.astype(int) + f.parch.astype(int), "age_known": f.age.notna().astype(int),
                          "survived": f.survived.astype(int)})
        X.to_csv(CACHE, index=False)
    d = pd.read_csv(CACHE)
    return d[["pclass", "female", "age", "fare", "family"]], d["survived"].to_numpy(), d["age_known"].to_numpy().astype(bool)


def gbdt(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0).fit(X, y)


TEAM = {"sex": ["female"], "class": ["pclass"], "rest": ["age", "fare", "family"]}


def team_value(M, X, i, S):
    Z = X.copy()
    for g in S:
        for c in TEAM[g]: Z[c] = X.iloc[i][c]
    return float(M.predict_proba(Z)[:, 1].mean() * 100)


def shapley(M, X, i):
    key = lambda S: tuple(g for g in TEAM if g in S)
    V = {S: team_value(M, X, i, S) for r in range(4) for S in combinations(TEAM, r)}
    orders = list(permutations(TEAM))
    gain = {o: {g: V[key(o[:k + 1])] - V[key(o[:k])] for k, g in enumerate(o)} for o in orders}
    return V, orders, gain, {g: sum(gain[o][g] for o in orders) / len(orders) for g in TEAM}


def toy():
    r = np.random.default_rng(1); x = np.sort(r.uniform(0, 10, 40)); return x, np.sin(x) + r.normal(0, 0.25, 40)


def fee(km):
    return np.where(km < 2, 3000, np.where(km < 4, 4000, 5500))


DIST = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9,
                 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9])  # km
BIN_W, N_BINS = 0.3, 20


def onehot_bins(km):
    return np.eye(N_BINS)[np.clip((km / BIN_W).astype(int), 0, N_BINS - 1)]


def step_models(km, won):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    tree = GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=0).fit(km[:, None], won)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, learning_rate_init=0.01, random_state=0).fit(km[:, None], won / 1000)
    mlp_bin = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=5000, learning_rate_init=0.01, random_state=0).fit(onehot_bins(km), won / 1000)
    return tree, mlp, mlp_bin


def step_curves(km, won, grid):
    tree, mlp, mlp_bin = step_models(km, won)
    return {"tree": tree.predict(grid[:, None]), "mlp": mlp.predict(grid[:, None]) * 1000, "mlp_bin": mlp_bin.predict(onehot_bins(grid)) * 1000}


def orders(n=400, seed=0):
    r = np.random.default_rng(seed)
    km, hour, rain = r.uniform(0, 6, n), r.uniform(0, 24, n), r.integers(0, 2, n)
    won = fee(km) + 500 * rain + np.where((hour >= 18) & (hour < 21), 1000, 0)  # 비 +500원, 저녁 6~9시 +1,000원
    return np.c_[km, hour, rain], won, r


def noise_errors(X, won, r, ks):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    out = {}
    for k in ks:
        Xk = np.c_[X, r.normal(size=(len(X), k))] if k else X
        Xs = (Xk - Xk.mean(0)) / Xk.std(0)
        tree = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(Xk[:300], won[:300])
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, learning_rate_init=0.003, random_state=0).fit(Xs[:300], won[:300] / 1000)
        out[k] = (round(float(np.abs(tree.predict(Xk[300:]) - won[300:]).mean())), round(float(np.abs(mlp.predict(Xs[300:]) * 1000 - won[300:]).mean())))
    return out


IA, IB = 13, 609                                                      # 둘 다 26살: 1등석 여성 / 3등석 남성
P_A, P_B, BASE = 96.0, 12.7, 38.2
PHI_A = {"sex": 34.0, "class": 22.6, "rest": 1.3}
PHI_B = {"sex": -14.8, "class": -7.6, "rest": -3.2}
V_A = {(): 38.2, ("sex",): 68.8, ("class",): 51.3, ("rest",): 34.3, ("sex", "class"): 87.8, ("sex", "rest"): 62.2, ("class", "rest"): 56.9, ("sex", "class", "rest"): 96.0}
TRAIN_ERR = {1: 0.382, 10: 0.201, 200: 0.139}                          # 나무 수별 학습 데이터 오답률
CV_ACC = 0.807
IMP = {"female": 19, "pclass": 11, "age": 7, "fare": 5, "family": 2}  # 전체 평균 |SHAP| (%p, 반올림)
TOY_MSE = {1: 0.48, 3: 0.39, 10: 0.22, 50: 0.04}                       # 장난감 예: 나무 수별 학습 MSE
KAGGLE_ALL, KAGGLE_XGB, KAGGLE_DNN = 29, 17, 11                       # 2015 캐글 우승 해법
TAB_N = 45                                                             # Grinsztajn 2022 표 데이터 수
STEP_MAE = {"tree": 3, "mlp": 291, "mlp_bin": 67}                     # 배달비 계단: 0~6km 평균 오차 (원)
NOISE_KS = (0, 5, 10, 20, 50)
NOISE_MAE = {0: (25, 207), 5: (31, 480), 10: (31, 480), 20: (47, 682), 50: (42, 747)}  # k: (나무, 신경망) 시험 평균 오차 (원)

if __name__ == "__main__":
    X, y, _ = titanic(); M = gbdt(X, y); P = M.predict_proba(X)[:, 1]
    assert X.iloc[IA].age == X.iloc[IB].age == 26 and (X.iloc[IA].pclass, X.iloc[IA].female, X.iloc[IB].pclass, X.iloc[IB].female) == (1, 1, 3, 0)
    assert (round(P[IA] * 100, 1), round(P[IB] * 100, 1)) == (P_A, P_B)
    VA, _, _, pa = shapley(M, X, IA); _, _, _, pb = shapley(M, X, IB)
    assert {k: round(v, 1) for k, v in pa.items()} == PHI_A and {k: round(v, 1) for k, v in pb.items()} == PHI_B
    assert {k: round(v, 1) for k, v in VA.items()} == V_A and round(VA[()], 1) == BASE
    err = {n: round(float((p.argmax(1) != y).mean()), 3) for n, p in enumerate(M.staged_predict_proba(X), 1) if n in TRAIN_ERR}
    assert err == TRAIN_ERR, err
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import GradientBoostingClassifier
    cv = cross_val_score(GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    assert round(cv, 3) == CV_ACC, cv
    import shap
    ex = shap.TreeExplainer(M, data=shap.sample(X, 200, random_state=0), model_output="probability")
    sv = ex.shap_values(X)
    assert {c: int(round(v)) for c, v in zip(X.columns, np.abs(sv).mean(0) * 100)} == IMP
    from sklearn.ensemble import GradientBoostingRegressor
    x, t = toy()
    g = GradientBoostingRegressor(n_estimators=50, max_depth=1, learning_rate=0.3, random_state=0).fit(x[:, None], t)
    mse = {n: round(float(np.mean((p - t) ** 2)), 2) for n, p in enumerate(g.staged_predict(x[:, None]), 1) if n in TOY_MSE}
    assert mse == TOY_MSE, mse
    grid = np.linspace(0, 6, 601); c = step_curves(DIST, fee(DIST).astype(float), grid)
    assert {k: round(float(np.abs(v - fee(grid)).mean())) for k, v in c.items()} == STEP_MAE
    Xo, won, r = orders(); assert noise_errors(Xo, won, r, NOISE_KS) == NOISE_MAE
    print("data ok")
