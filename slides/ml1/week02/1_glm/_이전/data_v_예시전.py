"""GLM 차시 공유 숫자 (슬라이드 · 그림 · 노트 일치). 실행하면 assert 로 확인한다.
데이터
- 점수: 1주차 선형회귀의 학생 30명 (공부 시간 → 점수), ../../week01/2_linear_regression/data.py 와 같은 배열
- 체스: lichess 2013년 1월, 승패가 난 117,003판 (CC0), data/chess_2013_01.csv.gz (로지스틱회귀 덱과 같은 파일)
- 월드컵: 본선 1930~2026 23개 대회 1068경기의 팀별 골 수 2136줄 (martj42/international_results, CC0),
  x = 경기 전 Elo 차이 (내 팀 − 상대, 홈 +100 포함). data/worldcup_goals.csv 는 prep_worldcup.py 가 만든다
- 붓꽃: Fisher (1936) iris 150송이 (sklearn.datasets.load_iris), 꽃잎 길이 하나로 세 종
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

STUDY_HOURS = [6.4, 9.0, 7.9, 2.6, 3.4, 8.8, 0.6, 8.3, 8.1, 4.9, 3.4, 3.1, 2.9, 4.7, 5.3, 5.8, 10.0, 8.0, 6.4, 9.9, 2.5, 2.0, 6.3, 0.9, 0.8, 5.4, 4.9, 9.2, 6.5, 5.4]
STUDY_SCORES = [65, 83, 74, 49, 62, 80, 43, 86, 77, 64, 57, 56, 48, 64, 73, 61, 94, 81, 69, 99, 57, 44, 72, 47, 43, 70, 64, 89, 80, 64]


def study():
    return np.array(STUDY_HOURS), np.array(STUDY_SCORES, dtype=float)


def chess():
    d = pd.read_csv(os.path.join(HERE, "data", "chess_2013_01.csv.gz"))
    return (d.white - d.black).values.astype(float), d.white_win.values.astype(float)


def goals():
    d = pd.read_csv(os.path.join(HERE, "data", "worldcup_goals.csv"))
    return d.elo_diff.values.astype(float), d.goals.values.astype(float)


def iris():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    return X[:, 2], y


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_glm(x, y, mean_fn, lr, steps, scale=1.0, shift=0.0, w0=0.0):
    """GLM 경사하강법: w <- w - lr * mean((yhat - y) x). 분포마다 mean_fn (평균 = g(z)) 만 다르다.
    x 는 (x - shift) / scale 로 넣고 계수를 원래 단위로 되돌린다. w0 = 절편 시작값 (포아송은 log 평균에서 시작)."""
    u = (x - shift) / scale; A = np.c_[np.ones_like(u), u]; w = np.array([w0, 0.0])
    for _ in range(steps):
        w -= lr * A.T @ (mean_fn(A @ w) - y) / len(y)
    w1 = w[1] / scale
    return np.array([w[0] - w1 * shift, w1])


def fit_softmax(x, y, k=3, lr=0.5, steps=40000, shift=4.0):
    """소프트맥스 회귀 (클래스 k 개, 특징 하나), 같은 (yhat - y) x 경사하강법."""
    u = x - shift; A = np.c_[np.ones_like(u), u]; Y = np.eye(k)[y]; W = np.zeros((2, k))
    for _ in range(steps):
        Z = A @ W; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        W -= lr * A.T @ (P - Y) / len(y)
    W[0] -= W[1] * shift
    return W


def softmax_p(W, v):
    z = W[0] + W[1] * np.asarray(v, float)[..., None]; e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- 슬라이드에 적는 숫자 (아래 assert)
STUDY_W = (38.58, 5.25)                 # 선형회귀: 점수 = 38.58 + 5.25 x
CHESS_W = (0.09, 0.0047)                # 로지스틱: P(백 승) = sigma(0.09 + 0.0047 x)
CHESS_Z200, CHESS_P200 = 1.03, 0.74     # +200 점: z = 1.03 -> 74 %
GOALS_N, GOALS_MATCHES, GOALS_CUPS = 2136, 1068, 23
GOALS_MEAN = 1.42                       # 한 팀이 한 경기에 넣는 골 평균
GOALS_XMIN, GOALS_XMAX = -575, 575
GOALS_LINE = (1.418, 0.00276)           # 직선: 골 = 1.418 + 0.00276 x
GOALS_LINE_ZERO, GOALS_LINE_NEG = -513, 5    # 직선이 0 이 되는 Elo 차이, 음수로 예측한 줄 수
GOALS_POIS = (0.289, 0.00195)           # 포아송: 골 = e^(0.289 + 0.00195 x)
GOALS_RATIO = 1.22                      # Elo 100 앞설 때마다 x1.22
GOALS_P = {-400: 0.61, -300: 0.74, 0: 1.34, 300: 2.40}      # 포아송 예측
GOALS_L = {-400: 0.31, -300: 0.59, 0: 1.42, 300: 2.25}      # 직선 예측
GOALS_PMIN, GOALS_LMIN = 0.43, -0.17     # 가장 약한 쪽 (Elo -575): 포아송 / 직선
GOALS_BIN_M450 = (18, 0.83, 0.17)       # Elo 차이 -500 ~ -400: 줄 수, 실제 평균 골, 직선 (가운데 -450)
GOALS_EVEN = (462, 1.30, 1.68)          # Elo 차이 -50 ~ +50: 줄 수, 평균, 분산 (분포 그림)
IRIS_P49 = (0.0, 0.42, 0.58)            # 꽃잎 4.9 cm: setosa / versicolor / virginica
IRIS_LOSS49 = 0.54                      # 4.9 cm 꽃이 virginica 일 때 교차 엔트로피 -ln 0.58
IRIS_CROSS = (2.6, 4.9)                 # 확률이 바뀌는 길이 (setosa|versicolor, versicolor|virginica)

if __name__ == "__main__":
    x, y = study(); w = np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)[0]
    print("study", w.round(2)); assert np.allclose(w, STUDY_W, atol=0.01)

    x, y = chess(); w = fit_glm(x, y, sig, lr=1.0, steps=3000, scale=100)
    print("chess", w.round(5), "z200", round(w[0] + 200 * w[1], 2), "p200", round(sig(w[0] + 200 * w[1]), 3))
    assert np.allclose(w, CHESS_W, atol=5e-3) and abs(w[1] - CHESS_W[1]) < 5e-5
    assert round(w[0] + 200 * w[1], 2) == CHESS_Z200 and round(sig(w[0] + 200 * w[1]), 2) == CHESS_P200

    x, y = goals(); k = np.polyfit(x, y, 1)
    print("goals n", len(x), "mean", y.mean().round(3), "x", x.min(), x.max(), "line", k[::-1].round(5), "zero", round(-k[1] / k[0]), "neg", int((np.polyval(k, x) < 0).sum()))
    assert len(x) == GOALS_N and (x.min(), x.max()) == (GOALS_XMIN, GOALS_XMAX) and round(y.mean(), 2) == GOALS_MEAN
    assert np.allclose(k[::-1], GOALS_LINE, atol=[5e-4, 5e-6]) and round(-k[1] / k[0]) == GOALS_LINE_ZERO and (np.polyval(k, x) < 0).sum() == GOALS_LINE_NEG
    wp = fit_glm(x, y, np.exp, lr=0.05, steps=20000, scale=100, w0=np.log(y.mean()))
    pm = lambda v: np.exp(wp[0] + wp[1] * v)
    print("poisson", wp.round(5), "ratio100", round(np.exp(100 * wp[1]), 3), {v: round(pm(v), 2) for v in GOALS_P})
    from sklearn.linear_model import PoissonRegressor
    sk = PoissonRegressor(alpha=0, max_iter=3000).fit(x[:, None], y)
    assert abs(sk.intercept_ - wp[0]) < 1e-4 and abs(sk.coef_[0] - wp[1]) < 1e-6      # 경사하강법 = sklearn
    assert np.allclose(wp, GOALS_POIS, atol=[5e-4, 5e-6]) and round(np.exp(100 * wp[1]), 2) == GOALS_RATIO
    assert all(round(pm(v), 2) == p for v, p in GOALS_P.items()) and all(round(np.polyval(k, v), 2) == p for v, p in GOALS_L.items())
    assert (round(pm(x.min()), 2), round(np.polyval(k, x.min()), 2)) == (GOALS_PMIN, GOALS_LMIN)
    m = (x >= -500) & (x < -400); assert (m.sum(), round(y[m].mean(), 2), round(np.polyval(k, -450), 2)) == GOALS_BIN_M450
    m = np.abs(x) <= 50; assert (m.sum(), round(y[m].mean(), 2), round(y[m].var(), 2)) == GOALS_EVEN

    x, y = iris(); W = fit_softmax(x, y); p = softmax_p(W, 4.9)
    xs = np.linspace(1, 7, 6001); P = softmax_p(W, xs)
    c01 = xs[np.argmax(P[:, 1] > P[:, 0])]; c12 = xs[np.argmax(P[:, 2] > P[:, 1])]
    print("iris W", W.round(2).tolist(), "p4.9", p.round(3), "cross", round(c01, 2), round(c12, 2), "acc", (softmax_p(W, x).argmax(1) == y).mean())
    assert round(-np.log(p[2]), 2) == IRIS_LOSS49
    assert np.allclose(p, IRIS_P49, atol=0.005) and (round(c01, 1), round(c12, 1)) == IRIS_CROSS
    print("data ok")
