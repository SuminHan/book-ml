"""GDA 차시 공유 숫자 (슬라이드 · 그림 · 노트 일치). 실행하면 assert 로 확인한다.
데이터: Fisher (1936) iris 중 versicolor (y = 0) 50송이 · virginica (y = 1) 50송이 (sklearn.datasets.load_iris).
1차원 = 꽃잎 길이, 2차원 = (꽃잎 길이, 꽃잎 너비). 분산 · 공분산은 최대 가능도 (n 으로 나눔), 사전확률 반반.
포아송 GDA: 월드컵 본선 골 수 (../1_glm/data/worldcup_goals.csv, GLM 덱과 같은 파일), y = 강팀 (Elo 앞섬) / 약팀, x = 넣은 골.
Elo 차이가 정확히 0 인 4줄은 뺀다."""
import numpy as np

NAMES = ("versicolor", "virginica")


def iris2():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    m = y > 0
    return X[m][:, 2:4], (y[m] == 2).astype(int)


def goals():
    import os, pandas as pd
    d = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_glm", "data", "worldcup_goals.csv"))
    d = d[d.elo_diff != 0]
    return d.goals.values.astype(int), (d.elo_diff.values > 0).astype(int)


def fit_pois_gda(x, y):
    """포아송 GDA 학습 = 비율과 클래스별 평균 골. 로그 오즈 = w0 + w1 x."""
    phi = y.mean(); lam = np.array([x[y == k].mean() for k in (0, 1)])
    return phi, lam, (-(lam[1] - lam[0]) + np.log(phi / (1 - phi)), np.log(lam[1] / lam[0]))


def gauss(v, mu, sd):
    return np.exp(-(np.asarray(v, float) - mu) ** 2 / (2 * sd * sd)) / (np.sqrt(2 * np.pi) * sd)


def fit_gda(X, y):
    """GDA 학습 = 세기: phi (비율), 클래스별 평균, 공유 공분산 (MLE)."""
    X = np.asarray(X, float).reshape(len(y), -1)
    phi = y.mean(); mu = np.array([X[y == k].mean(0) for k in (0, 1)])
    R = X - mu[y]; S = R.T @ R / len(y)
    return phi, mu, S


def gda_w(phi, mu, S):
    """공유 공분산 GDA 의 사후확률 = sigmoid(w0 + w . x)."""
    Si = np.linalg.inv(S); w = Si @ (mu[1] - mu[0])
    w0 = -0.5 * mu[1] @ Si @ mu[1] + 0.5 * mu[0] @ Si @ mu[0] + np.log(phi / (1 - phi))
    return w0, w


def fit_logistic(X, y, lr=0.5, steps=200000):
    X = np.asarray(X, float).reshape(len(y), -1); m = X.mean(0); A = np.c_[np.ones(len(y)), X - m]; w = np.zeros(A.shape[1])
    for _ in range(steps):
        w -= lr * A.T @ (1 / (1 + np.exp(-A @ w)) - y) / len(y)
    return w[0] - w[1:] @ m, w[1:]


def small_sample(X, y, ns=(4, 6, 10, 20, 40, 70), trials=500, seed=0):
    """학습 꽃 n 송이 (종마다 n/2) 로 GDA vs 로지스틱, 나머지로 시험한 정확도 평균."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed); out = {}
    for n in ns:
        g, l = [], []
        for _ in range(trials):
            tr = np.r_[rng.choice(np.where(y == 0)[0], n // 2, replace=False), rng.choice(np.where(y == 1)[0], n // 2, replace=False)]
            te = np.setdiff1d(np.arange(len(y)), tr)
            g.append(LDA().fit(X[tr], y[tr]).score(X[te], y[te])); l.append(LogisticRegression(C=1e6, max_iter=5000).fit(X[tr], y[tr]).score(X[te], y[te]))
        out[n] = (round(float(np.mean(g)), 3), round(float(np.mean(l)), 3))
    return out


# ---------------------------------------------------------------- 슬라이드 숫자 (아래 assert)
MU1D = (4.26, 5.55)            # 꽃잎 길이 평균 (versicolor, virginica) cm
SD1D = (0.47, 0.55)            # 종마다 표준편차
SD_SHARED = 0.51               # 공유 표준편차
X_Q = 4.9
H49 = (0.355, 0.344)           # 4.9 cm 에서 종의 높이 (versicolor, virginica), 공유 표준편차
POST49_GDA = 0.49              # GDA: virginica 일 확률
POST49_LR = 0.58               # 로지스틱회귀: virginica 일 확률
W_GDA = (-24.6, 5.02)          # GDA 가 정한 sigmoid(w0 + w1 x)
W_LR = (-43.8, 9.0)            # 로지스틱회귀 (경사하강법)
B_GDA, B_LR = 4.91, 4.86       # 경계
SIGMA2D = ((0.257, 0.060), (0.060, 0.056))
MU2D = ((4.26, 1.33), (5.55, 2.03))
ACC2D = {"gda": 0.94, "qda": 0.97, "lr": 0.94}
SMALL = {4: (0.837, 0.867), 6: (0.889, 0.884), 10: (0.921, 0.905), 20: (0.935, 0.919), 40: (0.935, 0.929), 70: (0.939, 0.937)}
POIS_N = 2132                   # 골 수 줄 (강팀 1066 · 약팀 1066)
POIS_LAM = (0.97, 1.87)         # 평균 골 (약팀, 강팀)
POIS_W = (-0.90, 0.66)          # 로그 오즈 = w0 + w1 x
POIS_TAB = {0: (605, 0.29, 0.29), 2: (443, 0.60, 0.60), 4: (89, 0.84, 0.85)}   # 골: 줄 수, 실제 강팀 비율, 모델

if __name__ == "__main__":
    X, y = iris2(); x = X[:, 0]
    mu = [x[y == k].mean() for k in (0, 1)]; sd = [x[y == k].std() for k in (0, 1)]
    phi, M, S = fit_gda(x, y); s = float(np.sqrt(S[0, 0]))
    h = (gauss(X_Q, M[0, 0], s), gauss(X_Q, M[1, 0], s)); post = h[1] / (h[0] + h[1])
    w0, w = gda_w(phi, M, S); wl0, wl = fit_logistic(x, y)
    pl = 1 / (1 + np.exp(-(wl0 + wl[0] * X_Q)))
    print("1d mu", np.round(mu, 2), "sd", np.round(sd, 2), "shared", round(s, 2), "h4.9", np.round(h, 3), "post", round(post, 3))
    print("gda w", round(w0, 2), round(w[0], 3), "b", round(-w0 / w[0], 2), "| lr w", round(wl0, 2), round(wl[0], 3), "b", round(-wl0 / wl[0], 2), "p4.9", round(pl, 3))
    assert tuple(np.round(mu, 2)) == MU1D and tuple(np.round(sd, 2)) == SD1D and round(s, 2) == SD_SHARED
    assert tuple(np.round(h, 3)) == H49 and round(post, 2) == POST49_GDA and round(pl, 2) == POST49_LR
    assert (round(w0, 1), round(w[0], 2)) == W_GDA and (round(wl0, 1), round(wl[0], 2)) == W_LR
    assert (round(-w0 / w[0], 2), round(-wl0 / wl[0], 2)) == (B_GDA, B_LR)
    phi2, M2, S2 = fit_gda(X, y)
    assert np.allclose(S2, SIGMA2D, atol=5e-4) and np.allclose(M2, MU2D, atol=5e-3)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
    from sklearn.linear_model import LogisticRegression
    acc = {"gda": LDA().fit(X, y).score(X, y), "qda": QDA().fit(X, y).score(X, y), "lr": LogisticRegression(C=1e6).fit(X, y).score(X, y)}
    print("2d Sigma", S2.round(3).tolist(), "acc", acc); assert {k: round(v, 2) for k, v in acc.items()} == ACC2D
    sm = small_sample(X, y); print("small", sm); assert sm == SMALL
    x, y = goals(); phi, lam, (p0, p1) = fit_pois_gda(x, y)
    print("pois", len(x), phi, lam.round(3), round(p0, 3), round(p1, 3))
    assert len(x) == POIS_N and phi == 0.5 and tuple(lam.round(2)) == POIS_LAM and (round(p0, 2), round(p1, 2)) == POIS_W
    for k, (n, a, m) in POIS_TAB.items():
        assert ((x == k).sum(), round(y[x == k].mean(), 2), round(1 / (1 + np.exp(-(p0 + p1 * k))), 2)) == (n, a, m)
    print("data ok")
