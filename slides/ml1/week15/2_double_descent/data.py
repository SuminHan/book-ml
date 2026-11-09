"""슬라이드·그림·실습이 공유하는 숫자: 이중 하강 (double descent).
1) 2주차 제동거리 점 10개 (같은 배열) + 다항식 최소 노름 해 (르장드르 기저, 속도를 (v - 55) / 50 으로 [-1, 1] 에).
2) 손글씨 숫자 optdigits (UCI, 8x8 = 64칸, 0~16): 학습 = optdigits.tra 앞 200장, 시험 = optdigits.tes 1797장 전부.
   모델 = 무작위 ReLU 특징 p개 + 선형 (첫 층 a_j 는 무작위로 고정, 마지막 층 w 만 학습). 입력은 학습 200장의 평균을 뺀다
   (안 빼면 모든 그림에서 0 인 '죽은' 특징이 생겨 피크가 p = 200 에서 밀린다). 정답은 원-핫 10칸, 최소제곱. 결과 = seed 5개 평균."""
import os
import numpy as np

# ---------- 1. 제동거리 (2주차와 같은 값) ----------
SPEED = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]          # km/h
DIST = [2, 7, 0, 14, 14, 18, 25, 33, 45, 59]               # m (측정, 잡음 있음)
SPEED_NEW = [15, 35, 45, 65, 85, 95]
DIST_NEW = [3, 9, 6, 19, 45, 54]
MU, G = 0.7, 9.8


def truth(v):
    return (np.asarray(v, float) / 3.6) ** 2 / (2 * MU * G)


def design(v, d):
    from numpy.polynomial import legendre as L
    return L.legvander((np.atleast_1d(np.asarray(v, float)) - 55) / 50, d)


def minnorm(x, y, d):
    """계수가 점보다 많으면 모든 점을 지나는 해 중 계수 제곱합이 가장 작은 것 (lstsq 가 이걸 준다)."""
    c, *_ = np.linalg.lstsq(design(x, d), np.asarray(y, float), rcond=None)
    return c


def mae(c, d, x, y):
    return float(np.abs(design(x, d) @ c - np.asarray(y, float)).mean())


DEGS = list(range(1, 21))
BEYOND = {1: 6.4, 2: 3.1, 3: 2.9, 4: 3.2, 5: 3.2, 6: 3.4, 7: 5.0, 8: 7.3, 9: 9.1, 10: 9.0, 11: 5.9, 12: 5.9, 13: 4.5,
          14: 4.6, 15: 4.7, 16: 3.9, 17: 5.2, 18: 5.1, 19: 5.0, 20: 5.1}               # 새 데이터 MAE (m), 2주차와 같음
NORM = {3: 39.0, 9: 130.0, 16: 38.6}                                                    # 계수 크기 ||w|| (제곱합의 제곱근)
NOISEFREE = {2: 0.0, 9: 0.0, 16: 1.5}            # 측정값 대신 물리 법칙 값으로 학습 → 새 속도에서 법칙과의 MAE (m)
NOISY = {2: 0.9, 9: 7.4, 16: 2.3}                # 같은 기준 (법칙과의 MAE), 측정값으로 학습
GD_DEG, GD_STEPS, GD_START = 16, 1000, 3.0          # 경사하강법: 16차, 1000번, 학습률 1/(가장 큰 특잇값)^2
GD_GAP = {0.0: 0.0, 3.0: 11.6}                   # 출발점 (모든 계수 = 이 값) → 최소 노름 해와의 거리 (둘 다 모든 점을 지난다)
SHOW_DEG = 15                                    # '모든 점을 지나는 곡선이 무수히 많다' 그림의 차수

def gd(X, y, w, steps):
    lr = 1 / np.linalg.norm(X, 2) ** 2
    for _ in range(steps):
        w = w - lr * X.T @ (X @ w - y)
    return w


# ---------- 2. 손글씨 숫자 ----------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
HERE = os.path.dirname(os.path.abspath(__file__))
N = 200                                          # 학습 이미지 수
P_MAX = 4000
SEEDS = range(5)
PS = [10, 20, 50, 100, 150, 180, 200, 220, 250, 300, 400, 600, 1000, 2000, 4000]
LAM = 0.1                                        # 릿지
NS = [50, 100, 150, 180, 200, 220, 250, 300, 400, 600, 1000, 2000]   # 데이터 수 실험 (p = 200 고정)


def load_digits(folder=os.path.join(HERE, "lab", "data")):
    import urllib.request
    os.makedirs(folder, exist_ok=True)
    out = []
    for name in ["optdigits.tra", "optdigits.tes"]:
        f = os.path.join(folder, name)
        if not os.path.exists(f):
            urllib.request.urlretrieve(URL + name, f)
        a = np.loadtxt(f, delimiter=",")
        out += [a[:, :64] / 16, a[:, 64].astype(int)]
    return out                                   # X_train (3823, 64), y_train, X_test (1797, 64), y_test


def relu_features(X, A):
    return np.maximum(X @ A, 0)


def test_error(X, y, Xt, yt, n, p, seed, lam=0.0, noisy=False):
    """학습 n장, 특징 p개, seed 하나 → (시험 오류율 %, ||w||)."""
    A = np.random.default_rng(seed).normal(size=(64, P_MAX))[:, :p] / 8
    mu = X[:n].mean(0)
    F, Ft = relu_features(X[:n] - mu, A), relu_features(Xt - mu, A)
    yy = y[:n].copy()
    if noisy:                                    # 다섯 장마다 하나 (0, 5, 10, ...) 의 라벨을 다음 숫자로 바꾼다 = 20%
        yy[::5] = (yy[::5] + 1) % 10
    Y = np.eye(10)[yy]
    if lam > 0:
        w = F.T @ np.linalg.solve(F @ F.T + lam * np.eye(n), Y)
    else:
        w, *_ = np.linalg.lstsq(F, Y, rcond=None)
    return 100 * float((np.argmax(Ft @ w, 1) != yt).mean()), float(np.linalg.norm(w))


def sweep(data, ps=PS, n=N, **kw):
    return {p: tuple(np.mean([test_error(*data, n, p, s, **kw) for s in SEEDS], 0)) for p in ps}


def sweep_n(data, ns=NS, p=200):
    return {n: float(np.mean([test_error(*data, n, p, s)[0] for s in SEEDS])) for n in ns}


# 슬라이드에 적는 결과 (반올림, 아래 assert 로 확인)
CLEAN = {10: 48, 20: 33, 50: 21, 100: 20, 150: 27, 180: 37, 200: 83, 220: 38, 250: 25, 300: 18, 400: 14, 600: 11,
         1000: 11, 2000: 10, 4000: 10}                                                  # 오류율 %
CLEAN_NORM = {100: 9, 200: 294, 4000: 1}
NOISE = {100: 40, 200: 87, 4000: 36}
RIDGE = {100: 17, 200: 14, 4000: 10}
BY_N = {50: 30, 100: 24, 150: 28, 180: 37, 200: 83, 220: 39, 250: 25, 300: 18, 400: 11, 600: 8, 1000: 6, 2000: 4}                  # p = 200, 학습 n장 → 오류율 %

if __name__ == "__main__":
    for d, e in BEYOND.items():
        assert round(mae(minnorm(SPEED, DIST, d), d, SPEED_NEW, DIST_NEW), 1) == e, d
    for d, v in NORM.items():
        assert round(float(np.linalg.norm(minnorm(SPEED, DIST, d))), 1) == v, d
    for d in NOISEFREE:
        assert round(mae(minnorm(SPEED, truth(SPEED), d), d, SPEED_NEW, truth(SPEED_NEW)), 1) == NOISEFREE[d], d
        assert round(mae(minnorm(SPEED, DIST, d), d, SPEED_NEW, truth(SPEED_NEW)), 1) == NOISY[d], d
    X, y = design(SPEED, GD_DEG), np.asarray(DIST, float)
    for w0, gap in GD_GAP.items():
        w = gd(X, y, np.full(GD_DEG + 1, w0), GD_STEPS)
        assert round(float(np.abs(X @ w - y).max()), 3) == 0 and round(float(np.linalg.norm(w - minnorm(SPEED, DIST, GD_DEG))), 1) == gap
    D = load_digits()
    s = sweep(D)
    assert {p: round(e) for p, (e, _) in s.items()} == CLEAN, {p: round(e) for p, (e, _) in s.items()}
    assert all(round(s[p][1]) == v for p, v in CLEAN_NORM.items())
    assert {p: round(e) for p, (e, _) in sweep(D, list(NOISE), noisy=True).items()} == NOISE
    assert {p: round(e) for p, (e, _) in sweep(D, list(RIDGE), lam=LAM).items()} == RIDGE
    assert {n: round(e) for n, e in sweep_n(D, list(BY_N)).items()} == BY_N
    print("data ok")
