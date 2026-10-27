"""슬라이드·그림·실습이 공유하는 숫자 (notebooks/ml1/chapter06_3 + 쇼츠 lessons/ml1/data_leakage 의 동전 실험).
동전 실험: exp_leak.py (assets/leak.json) — 100명 × 의미 없는 난수 5,000개, 정답 = 동전, 숫자 20개 고르기 → 로지스틱 회귀, 5겹, 50번.
나머지는 chapter06_3 과 같은 설정을 여기서 다시 계산한다."""
import os, json
import numpy as np

_J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "leak.json")))
LEAK, OK = _J["leak"], _J["ok"]
COIN = (87, 49, 75, 61)                       # 누수 평균 %, 정상 평균 %, 누수 최저, 정상 최고


def e_max(m, n_sims=40000, seed=1):
    """진짜 성능이 모두 0 인 후보 m 개를 잡음 σ=1 로 재서 가장 좋은 것을 고를 때, 고른 점수의 평균"""
    return float(np.random.default_rng(seed).normal(0, 1, (n_sims, m)).max(axis=1).mean())


EMAX = {1: 0.0, 5: 1.16, 50: 2.25}
STD10 = {"leak": (5.5, 2.872, [0.87, 1.22, 1.57]), "ok": (4.0, 2.0, [2.0, 2.5, 3.0])}   # 1~10 중 test 8·9·10 의 z


def patient_rmse():
    from sklearn.neighbors import KNeighborsRegressor
    P, T = 100, 5
    r = np.random.default_rng(51)
    v = r.normal(0, 2.0, P); c = r.normal(0, 2.0, (P, 2))
    X = (c[:, None, :] + r.normal(0, 0.1, (P, T, 2))).reshape(P * T, 2); y = np.repeat(v, T)

    def rmse(tr, te):
        m, s = X[tr].mean(0), X[tr].std(0)
        p = KNeighborsRegressor(1).fit((X[tr] - m) / s, y[tr]).predict((X[te] - m) / s)
        return float(np.sqrt(np.mean((p - y[te]) ** 2)))
    perm = np.random.default_rng(52).permutation(P * T); cut = int(0.8 * P * T)
    g = np.random.default_rng(53).permutation(P); te_p = np.isin(np.repeat(np.arange(P), T), g[80:])
    return X, y, rmse(perm[:cut], perm[cut:]), rmse(np.where(~te_p)[0], np.where(te_p)[0]), perm[cut:], te_p


def time_rmse():
    from sklearn.neighbors import KNeighborsRegressor
    t = np.arange(365.0); y = 100 + t + np.random.default_rng(71).normal(0, 0.5, 365); X = t[:, None]

    def rmse(tr, te):
        m, s = X[tr].mean(0), X[tr].std(0)
        p = KNeighborsRegressor(5).fit((X[tr] - m) / s, y[tr]).predict((X[te] - m) / s)
        return float(np.sqrt(np.mean((p - y[te]) ** 2))), p
    perm = np.random.default_rng(72).permutation(365); cut = int(365 * 0.8)
    a, pa = rmse(perm[:cut], perm[cut:]); b, pb = rmse(np.arange(cut), np.arange(cut, 365))
    return t, y, a, b, perm[cut:], pa, pb


PATIENT = (1.04, 3.65)
TIME = (0.79, 44.12)
DIAB = {"sizes": (265, 88, 89), "val": {0.1: 58.07, 1.0: 61.75, 10.0: 73.04, 100.0: 77.49, 1000.0: 78.07}, "test_at_val_pick": 53.79}

if __name__ == "__main__":
    assert (round(np.mean(LEAK) * 100), round(np.mean(OK) * 100), round(min(LEAK) * 100), round(max(OK) * 100)) == COIN
    assert {m: round(e_max(m), 2) for m in EMAX} == {1: -0.01, 5: 1.16, 50: 2.25}
    d = np.arange(1, 11.0); tr, te = d[:7], d[7:]
    for k, src in (("leak", d), ("ok", tr)):
        mu, sd = src.mean(), src.std(); assert (round(mu, 3), round(sd, 3), list(np.round((te - mu) / sd, 2))) == STD10[k]
    assert tuple(round(v, 2) for v in patient_rmse()[2:4]) == PATIENT
    assert tuple(round(v, 2) for v in time_rmse()[2:4]) == TIME
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=0)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0)
    assert (len(Xtr), len(Xva), len(Xte)) == DIAB["sizes"]
    val = {l: round(float(np.sqrt(np.mean((Ridge(l).fit(Xtr, ytr).predict(Xva) - yva) ** 2))), 2) for l in DIAB["val"]}
    assert val == DIAB["val"], val
    best = min(val, key=val.get)
    assert round(float(np.sqrt(np.mean((Ridge(best).fit(Xtr, ytr).predict(Xte) - yte) ** 2))), 2) == DIAB["test_at_val_pick"]
    print("data ok")
