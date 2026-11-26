"""GLM 차시 공유 숫자 (슬라이드 · 그림 · 노트 일치). 실행하면 assert 로 확인한다.
데이터
- 점수: 1주차 선형회귀의 학생 30명 (공부 시간 → 점수), ../../week01/2_linear_regression/data.py 와 같은 배열
- 체스: lichess 2013년 1월, 승패가 난 117,003판 (CC0), data/chess_2013_01.csv.gz (로지스틱회귀 덱과 같은 파일)
- 따릉이: 서울 공공자전거 시간별 대여 수 (UCI Seoul Bike Sharing Demand, 2017-12 ~ 2018-11, CC BY 4.0) 중
  운영한 날의 아침 6시 352개, data/seoul_bike_6am.csv (원본 ~/book-ml/notebooks/datashop/data/seoul_bike.zip)
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


def bike():
    d = pd.read_csv(os.path.join(HERE, "data", "seoul_bike_6am.csv"))
    return d.temp.values.astype(float), d["count"].values.astype(float)


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
BIKE_N = 352
BIKE_LINE = (181.9, 12.31)              # 직선: 대여 = 181.9 + 12.31 t
BIKE_LINE_ZERO, BIKE_LINE_NEG = -14.8, 5   # 직선이 0 이 되는 기온, 음수로 예측한 날 수
BIKE_TMIN, BIKE_TMAX = -17.4, 30.5
BIKE_POIS = (5.15, 0.0447)              # 포아송: 대여 = e^(5.15 + 0.0447 t)
BIKE_RATIO = 1.046                      # 1 도마다 x1.046 (4.6 %)
BIKE_P0, BIKE_P20, BIKE_PMIN = 173, 422, 79   # 포아송 예측: 0 도, 20 도, 가장 추운 날
IRIS_P49 = (0.0, 0.42, 0.58)            # 꽃잎 4.9 cm: setosa / versicolor / virginica
IRIS_CROSS = (2.6, 4.9)                 # 확률이 바뀌는 길이 (setosa|versicolor, versicolor|virginica)

if __name__ == "__main__":
    x, y = study(); w = np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)[0]
    print("study", w.round(2)); assert np.allclose(w, STUDY_W, atol=0.01)

    x, y = chess(); w = fit_glm(x, y, sig, lr=1.0, steps=3000, scale=100)
    print("chess", w.round(5), "z200", round(w[0] + 200 * w[1], 2), "p200", round(sig(w[0] + 200 * w[1]), 3))
    assert np.allclose(w, CHESS_W, atol=5e-3) and abs(w[1] - CHESS_W[1]) < 5e-5
    assert round(w[0] + 200 * w[1], 2) == CHESS_Z200 and round(sig(w[0] + 200 * w[1]), 2) == CHESS_P200

    t, c = bike(); k = np.polyfit(t, c, 1)
    print("bike n", len(t), "t", t.min(), t.max(), "line", k.round(2), "zero", round(-k[1] / k[0], 1), "neg", int((np.polyval(k, t) < 0).sum()))
    assert len(t) == BIKE_N and (t.min(), t.max()) == (BIKE_TMIN, BIKE_TMAX)
    assert np.allclose(k[::-1], BIKE_LINE, atol=0.05) and round(-k[1] / k[0], 1) == BIKE_LINE_ZERO and (np.polyval(k, t) < 0).sum() == BIKE_LINE_NEG
    wp = fit_glm(t, c, np.exp, lr=0.002, steps=20000, scale=10, shift=10, w0=np.log(c.mean()))
    pm = lambda v: np.exp(wp[0] + wp[1] * v)
    print("poisson", wp.round(4), "ratio", round(np.exp(wp[1]), 3), "p0", round(pm(0)), "p20", round(pm(20)), "pmin", round(pm(t.min())))
    from sklearn.linear_model import PoissonRegressor
    sk = PoissonRegressor(alpha=0, max_iter=2000).fit(t[:, None], c)
    assert abs(sk.intercept_ - wp[0]) < 1e-3 and abs(sk.coef_[0] - wp[1]) < 1e-4      # 경사하강법 = sklearn
    assert np.allclose(wp, BIKE_POIS, atol=5e-3) and round(np.exp(wp[1]), 3) == BIKE_RATIO
    assert (round(pm(0)), round(pm(20)), round(pm(t.min()))) == (BIKE_P0, BIKE_P20, BIKE_PMIN)

    x, y = iris(); W = fit_softmax(x, y); p = softmax_p(W, 4.9)
    xs = np.linspace(1, 7, 6001); P = softmax_p(W, xs)
    c01 = xs[np.argmax(P[:, 1] > P[:, 0])]; c12 = xs[np.argmax(P[:, 2] > P[:, 1])]
    print("iris W", W.round(2).tolist(), "p4.9", p.round(3), "cross", round(c01, 2), round(c12, 2), "acc", (softmax_p(W, x).argmax(1) == y).mean())
    assert np.allclose(p, IRIS_P49, atol=0.005) and (round(c01, 1), round(c12, 1)) == IRIS_CROSS
    print("data ok")
