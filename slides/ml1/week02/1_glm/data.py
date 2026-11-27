"""GLM 차시 공유 숫자 (슬라이드 · 그림 · 노트 일치). 실행하면 assert 로 확인한다.
데이터
- 점수: 1주차 선형회귀의 학생 30명 (공부 시간 → 점수), ../../week01/2_linear_regression/data.py 와 같은 배열
- 체스: lichess 2013년 1월, 승패가 난 117,003판 (CC0), data/chess_2013_01.csv.gz (로지스틱회귀 덱과 같은 파일)
- 월드컵: 본선 1930~2026 23개 대회 1068경기의 팀별 골 수 2136줄 (martj42/international_results, CC0),
  x = 경기 전 Elo 차이 (내 팀 − 상대, 홈 +100 포함). data/worldcup_goals.csv 는 prep_worldcup.py 가 만든다
- 말 발길질: Bortkiewicz (1898) 프로이센 기병 10개 부대 x 20년 = 200 부대-년, 한 해 말에 차여 죽은 병사 수
- V-1 폭탄: Clarke (1946) 런던 남부 576 구역 (0.25 km²), 구역마다 떨어진 폭탄 수 (537발)
- 자동차 보험: freMTPL2freq (프랑스 자동차 책임보험, OpenML 41214) 중 1년을 꼬박 채운 169,349 계약.
  y = 1년 사고 청구 횟수, x = 보너스-말러스 등급 (처음 100, 무사고 1년마다 x0.95 · 최저 50, 사고 1번마다 x1.25).
  data/insurance_claims.csv 는 prep_insurance.py 가 만든다
- V-1 실제 낙하 지점: Shaw, L. (2025) "V1 and V2 impact sites in London" (Zenodo 14883986, CC BY 4.0), 런던 시의회 (LCC) 구역 887발,
  LCC Bomb Damage Maps 에서 손으로 옮긴 좌표 (영국 격자 easting/northing, m). data/v1_impacts_lcc.csv
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


def claims():
    d = pd.read_csv(os.path.join(HERE, "data", "insurance_claims.csv"))
    return d.bonus_malus.values.astype(float), d.claims.values.astype(float)


def claim_means(x, y, lo=50, hi=150, step=10):
    """등급 10점 구간마다 [가운데, 실제 평균 청구, 계약 수]."""
    out = []
    for a in range(lo, hi, step):
        m = (x >= a) & (x < a + step)
        if m.sum(): out.append((a + step / 2 if a > lo else lo, y[m].mean(), m.sum()))
    return np.array(out)


def v1():
    d = pd.read_csv(os.path.join(HERE, "data", "v1_impacts_lcc.csv"))
    return d.easting.values, d.northing.values


def v1_lonlat():
    d = pd.read_csv(os.path.join(HERE, "data", "v1_impacts_lcc.csv"))
    return d.lon.values, d.lat.values


V1_CENTER = (-0.114, 51.454)             # 템스강 남쪽 (경도, 위도)
V1_BOX = (12000, 9000)                  # 가로 12 km x 세로 9 km (지도와 나란한 격자)
V1_CELL = 500                           # 0.5 km 칸 (0.25 km², Clarke 와 같은 크기) → 24 x 18 = 432칸


def v1_xy(lon, lat, center=V1_CENTER):
    """경위도 → 가운데 기준 m (동쪽 x, 북쪽 y). 이 넓이에서는 평면으로 봐도 된다."""
    return (lon - center[0]) * 111320 * np.cos(np.radians(center[1])), (lat - center[1]) * 110574


def v1_cells(lon, lat, box=V1_BOX, cell=V1_CELL):
    x, y = v1_xy(lon, lat); w, h = box; m = (np.abs(x) < w / 2) & (np.abs(y) < h / 2)
    return np.histogram2d(x[m], y[m], bins=[w // cell, h // cell], range=[[-w / 2, w / 2], [-h / 2, h / 2]])[0].astype(int), m.sum()


def v1_sim(k, cells, reps=10000, seed=0):
    """k 발을 cells 칸에 제멋대로 (칸마다 같은 확률) 떨어뜨리기를 reps 번: 칸별 개수 표 (reps x 6, 5+ 합침)."""
    rng = np.random.default_rng(seed)
    return np.array([np.bincount(np.minimum(np.bincount(rng.integers(0, cells, k), minlength=cells), 5), minlength=6) for _ in range(reps)])


def cannon(vs=None, reps=1000, n=200, sd=1.0, line=100.0, seed=0, g=9.8):
    """대포 시뮬레이션: 45도로 n 발을 쏜다. 속도는 설정값 v + 흔들림 (표준편차 sd m/s), 사거리 = v^2 / g.
    y = n 발 중 line (m) 선을 넘은 개수. 설정 속도마다 reps 번 되풀이. 돌려주는 것: (x = 설정 속도, y = 개수) 줄들."""
    vs = np.round(np.arange(28.0, 29.51, 0.25), 2) if vs is None else np.asarray(vs)
    rng = np.random.default_rng(seed); X, Y = [], []
    for v in vs:
        V = v + rng.normal(0, sd, (reps, n)); Y.append(((V ** 2 / g) > line).sum(1)); X.append(np.full(reps, v))
    return np.concatenate(X), np.concatenate(Y).astype(float)


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
HORSE_KICKS = [109, 65, 22, 3, 1]       # 사망 0,1,2,3,4 명인 부대-년 수 (200)
HORSE_MEAN = 0.61
V1_BOMBS = [229, 211, 93, 35, 7, 1]     # 폭탄 0,1,2,3,4,5+ 발인 구역 수 (576), 5+ 구역은 실제 7발
V1_N, V1_TOTAL, V1_MEAN = 576, 537, 0.93
POKER = {"원페어": (1098240, 3), "트리플": (54912, 60), "풀하우스": (3744, 880)}   # 5장 패 2,598,960 가지 중 경우 수, 판 수
POKER_LAM = 1.27                        # 세 경우 모두 판 수 x 확률 = 1.27
POKER_P = (42.26, 2.11, 0.144)          # 한 판에 나올 확률 (%)
POKER_GAP = (0.089, 0.004, 0.000)       # 이항 vs 포아송 확률 차이의 최댓값
V1_BINOM = (227, 212, 99, 31, 7)        # 537발이 각자 1/576 확률로 한 구역에: 이항으로 기대한 구역 수 (0~4발)
CLAIMS_N = 169349                      # 1년 꼬박 채운 계약
CLAIMS_MEAN, CLAIMS_VAR = 0.070, 0.072  # 1년 사고 청구: 평균 · 분산 (거의 같다 → 포아송)
CLAIMS_COUNTS = (158087, 10723, 521, 18)   # 0 · 1 · 2 · 3번 이상
CLAIMS_LINE = (-0.097, 0.00306)         # 직선: 청구 = -0.097 + 0.00306 x
CLAIMS_POIS = (-4.137, 0.0260)          # 포아송: 청구 = e^(-4.137 + 0.0260 x)
CLAIMS_RATIO10 = 1.30                   # 등급 10점 오를 때마다 x1.30
CLAIMS_P = {50: 0.059, 100: 0.216, 150: 0.793}   # 포아송 예측
CLAIMS_L = {50: 0.056, 100: 0.209, 150: 0.362}   # 직선 예측
CLAIMS_BIN140 = (50, 0.72)              # 등급 140~149: 계약 수, 실제 평균 청구
CLAIMS_BM50 = (123281, 0.056, 0.058)    # 등급 50 (오래 무사고): 계약 수, 평균, 분산 (분포 그림)
CLAIMS_G1 = -0.035                      # 첫 걸음: 평균((yhat - y) x), x 는 10점 단위
V1_REAL_N = 887                         # LCC 구역 V-1 낙하 지점
V1_REAL_K, V1_REAL_CELLS = 383, 432     # 남쪽 12 x 9 km 안의 폭탄 수, 칸 수
V1_REAL_TAB = (190, 141, 68, 26, 7, 0)  # 0 · 1 · 2 · 3 · 4 · 5+ 발 칸 수
V1_REAL_MV = (0.89, 0.97)               # 칸당 평균, 분산
V1_SIM_ZERO = (166, 190)                # 제멋대로 1만 번: 0발 칸 수의 95% 범위
V1_SIM_ONE, V1_SIM_TWO = (139, 177), (57, 83)   # 1발 · 2발 칸 수의 95% 범위 (실제는 모두 안)
CANNON_MV29 = (2.16, 2.21)              # 설정 속도 29 m/s: 200발 중 100 m 넘은 개수의 평균 · 분산
CANNON_W = (-74.8, 2.60)                # 포아송 회귀: 개수 = e^(w0 + w1 x), x = 설정 속도 (28 ~ 29.5 m/s)
CANNON_RATIO = 1.92                     # 속도 0.25 m/s 늘 때마다 개수 x1.92
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

    x, y = claims(); k = np.polyfit(x, y, 1)
    c = np.bincount(np.minimum(y.astype(int), 3))
    print("claims", len(x), y.mean().round(4), y.var().round(4), c, "line", k[::-1].round(5))
    assert len(x) == CLAIMS_N and (round(y.mean(), 3), round(y.var(), 3)) == (CLAIMS_MEAN, CLAIMS_VAR) and tuple(c) == CLAIMS_COUNTS
    assert np.allclose(k[::-1], CLAIMS_LINE, atol=[5e-4, 5e-6])
    wp = fit_glm(x, y, np.exp, lr=0.5, steps=20000, scale=10, shift=50, w0=np.log(y.mean()))
    pm = lambda v: np.exp(wp[0] + wp[1] * v)
    from sklearn.linear_model import PoissonRegressor
    sk = PoissonRegressor(alpha=0, max_iter=3000).fit(x[:, None], y)
    print("poisson", wp.round(5), "sk", round(sk.intercept_, 5), sk.coef_.round(5), {v: round(pm(v), 3) for v in CLAIMS_P})
    assert abs(sk.intercept_ - wp[0]) < 1e-3 and abs(sk.coef_[0] - wp[1]) < 1e-5
    assert np.allclose(wp, CLAIMS_POIS, atol=[5e-4, 5e-5]) and round(np.exp(10 * wp[1]), 2) == CLAIMS_RATIO10
    assert all(round(pm(v), 3) == p for v, p in CLAIMS_P.items()) and all(round(np.polyval(k, v), 3) == p for v, p in CLAIMS_L.items())
    m = (x >= 140) & (x < 150); assert (m.sum(), round(y[m].mean(), 2)) == CLAIMS_BIN140
    m = x == 50; assert (m.sum(), round(y[m].mean(), 3), round(y[m].var(), 3)) == CLAIMS_BM50
    assert round(np.mean((y.mean() - y) * x / 10), 3) == CLAIMS_G1

    x, y = goals(); k = np.polyfit(x, y, 1)
    assert len(x) == GOALS_N
    m = np.abs(x) <= 50; assert (m.sum(), round(y[m].mean(), 2), round(y[m].var(), 2)) == GOALS_EVEN

    e, n = v1(); vl, va = v1_lonlat(); H, k = v1_cells(vl, va); tab = np.bincount(np.minimum(H.ravel(), 5), minlength=6)
    print("v1 real", len(e), k, H.size, tab, H.mean().round(3), H.var().round(3))
    assert (len(e), k, H.size, tuple(tab)) == (V1_REAL_N, V1_REAL_K, V1_REAL_CELLS, V1_REAL_TAB)
    assert (round(H.mean(), 2), round(H.var(), 2)) == V1_REAL_MV
    S = v1_sim(k, H.size); assert tuple(np.percentile(S[:, 0], [2.5, 97.5]).astype(int)) == V1_SIM_ZERO
    assert tuple(np.percentile(S[:, 1], [2.5, 97.5]).astype(int)) == V1_SIM_ONE and tuple(np.percentile(S[:, 2], [2.5, 97.5]).astype(int)) == V1_SIM_TWO

    x, y = cannon(); m = x == 29.0
    from sklearn.linear_model import PoissonRegressor
    cw = PoissonRegressor(alpha=0, max_iter=1000).fit(x[:, None], y)
    print("cannon", round(y[m].mean(), 2), round(y[m].var(), 2), round(cw.intercept_, 2), cw.coef_.round(3), round(np.exp(0.25 * cw.coef_[0]), 2))
    assert (round(y[m].mean(), 2), round(y[m].var(), 2)) == CANNON_MV29
    assert (round(cw.intercept_, 1), round(cw.coef_[0], 2)) == CANNON_W and round(np.exp(0.25 * cw.coef_[0]), 2) == CANNON_RATIO

    x, y = iris(); W = fit_softmax(x, y); p = softmax_p(W, 4.9)
    xs = np.linspace(1, 7, 6001); P = softmax_p(W, xs)
    c01 = xs[np.argmax(P[:, 1] > P[:, 0])]; c12 = xs[np.argmax(P[:, 2] > P[:, 1])]
    print("iris W", W.round(2).tolist(), "p4.9", p.round(3), "cross", round(c01, 2), round(c12, 2), "acc", (softmax_p(W, x).argmax(1) == y).mean())
    assert round(-np.log(p[2]), 2) == IRIS_LOSS49
    assert np.allclose(p, IRIS_P49, atol=0.005) and (round(c01, 1), round(c12, 1)) == IRIS_CROSS
    print("data ok")
