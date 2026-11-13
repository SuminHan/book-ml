"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/svm 과 같은 장면: 빨강/파랑 두 무리, 가장 넓은 길, 원 모양 데이터와 커널).
점 좌표는 0~10 눈금의 예시 값 (손으로 고름). 선형 SVM (C 아주 큼 = 하드 마진) 의 해:
w = (4/9)(-1, 1), b = -1/9  →  경계 x2 - x1 = 0.25, 길 가장자리 x2 - x1 = 2.5 (빨강) / -2 (파랑), 마진 4.5/√2 = 3.18."""
import numpy as np

RED_PTS = [(1.5, 6.0), (2.0, 7.5), (2.5, 5.5), (3.0, 7.0), (1.0, 8.0), (3.5, 6.0), (2.0, 4.5), (3.0, 8.5), (1.2, 5.2), (4.0, 7.8)]
BLUE_PTS = [(6.0, 2.5), (7.0, 4.0), (5.5, 1.5), (8.0, 2.0), (6.5, 3.5), (7.5, 1.0), (5.0, 3.0), (8.5, 3.5), (6.0, 0.8), (7.8, 5.0)]
W, B = (-4 / 9, 4 / 9), -1 / 9
MARGIN = 3.18
SV = [(3.5, 6.0), (2.0, 4.5), (5.0, 3.0)]
FAR_MOVE = ((1.0, 8.0), (0.3, 9.5))                      # 멀리 있는 빨강 점을 옮겨도 경계는 그대로
OUTLIER = (3.0, 5.2)                                     # 소프트 마진 장면: 빨강 무리 안에 들어간 파랑 점
SOFT = {100: (0.21, 3, 0), 0.05: (4.64, 8, 1)}               # C: (마진, 서포트 벡터 수, 틀린 학습 점 수) — 파랑 이상치 포함
SWEEP = {25: 2.00, 45: 3.18, 65: 2.63}                   # 선의 기울기 각도(°): 그 방향으로 낼 수 있는 가장 넓은 길의 폭 (방향 돌리기 장면)


def sweep(deg, R=None, Bl=None):
    """선의 기울기가 deg° 일 때: (폭, 빨강 쪽 가장자리 값, 파랑 쪽 가장자리 값, 닿는 빨강 점, 닿는 파랑 점). 값 = 경계에 수직인 방향으로 잰 위치."""
    R = np.array(RED_PTS if R is None else R, float); Bl = np.array(BLUE_PTS if Bl is None else Bl, float)
    t = np.radians(deg + 90); n = np.array([np.cos(t), np.sin(t)])
    pr, pb = R @ n, Bl @ n
    return pr.min() - pb.max(), pr.min(), pb.max(), R[np.isclose(pr, pr.min())], Bl[np.isclose(pb, pb.max())]


TRAIN_T, TRAIN_LR = 20000, 0.02                         # 경사하강 학습 장면: 걸음 수, 학습률 0.02/√t (C = 1)
TOUCH = {(3.5, 6.0): 176, (5.0, 3.0): 278, (2.0, 4.5): 7567}   # 1걸음 뒤 처음으로 '미는' 걸음 (나머지 17개는 1걸음 뒤로 한 번도 안 민다)
SNAPS = (2, 176, 278, 20000)
STEPS = {1: (0, 0, 0, 20), 2: (-0.88, 0.78, 0, 0), 100: (-0.62, 0.55, 0, 0), 176: (-0.54, 0.48, 0, 1), 278: (-0.46, 0.45, 0, 1), 20000: (-0.45, 0.44, -0.10, 2)}   # 걸음: (w1, w2, b, 미는 점 수) — 표 쪽
SUM_YX = (-44.1, 39.2)                                    # 1걸음: 모든 점이 밀 때 Σ y_i x_i                              # 그림에 쓰는 걸음


def train_svm(X, y, T=TRAIN_T, lr0=TRAIN_LR, C=1.0, snaps=()):
    """여유 m_i = y_i (w·x_i + b) 가 1 보다 작은 점만 경계를 민다 (힌지 손실의 경사하강).
    한 걸음: w ← w − η (w − C Σ_{m<1} y_i x_i),  b ← b + η C Σ_{m<1} y_i,  η = lr0/√t.
    반환: 마지막 (w, b), 걸음마다 민 점 표시 (T × n), snaps 걸음 직전의 (w, b, 미는 점)."""
    w, b = np.zeros(X.shape[1]), 0.0
    pushed, shots = np.zeros((T, len(X)), bool), {}
    for t in range(1, T + 1):
        act = y * (X @ w + b) < 1
        pushed[t - 1] = act
        if t in snaps:
            shots[t] = (w.copy(), b, act.copy())
        eta = lr0 / np.sqrt(t)
        w, b = w - eta * (w - C * (y[act, None] * X[act]).sum(0)), b + eta * C * y[act].sum()
    return w, b, pushed, shots


KX, KY = (1, 2), (3, 1)                                  # 커널 계산 예: (x·x')² = φ(x)·φ(x') = 25


def phi(v):
    return np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])


if __name__ == "__main__":
    from sklearn.svm import SVC
    X = np.array(RED_PTS + BLUE_PTS, float); y = np.r_[np.ones(10), -np.ones(10)]
    m = SVC(kernel="linear", C=1e6).fit(X, y)
    assert np.allclose(m.coef_[0], W, atol=1e-3) and abs(m.intercept_[0] - B) < 1e-2
    assert round(2 / np.linalg.norm(m.coef_[0]), 2) == MARGIN and sorted(map(tuple, X[m.support_])) == sorted(SV)
    X2 = X.copy(); X2[RED_PTS.index(FAR_MOVE[0])] = FAR_MOVE[1]
    m2 = SVC(kernel="linear", C=1e6).fit(X2, y); assert np.allclose(m2.coef_, m.coef_, atol=1e-3)
    Xo = np.r_[X, [OUTLIER]]; yo = np.r_[y, [-1]]
    for C, (mg, nsv, err) in SOFT.items():
        m3 = SVC(kernel="linear", C=C).fit(Xo, yo)
        assert (round(2 / np.linalg.norm(m3.coef_[0]), 2), len(m3.support_), int((m3.predict(Xo) != yo).sum())) == (mg, nsv, err)
    assert np.dot(KX, KY) ** 2 == 25 and round(float(phi(KX) @ phi(KY)), 6) == 25
    for d, wd in SWEEP.items():
        assert round(sweep(d)[0], 2) == wd
    best = max(np.arange(0, 90.01, 0.05), key=lambda d: sweep(d)[0])
    assert abs(best - 45) < 0.06 and len(sweep(45)[3]) + len(sweep(45)[4]) == 3      # 꼭대기 45° 에서 점 3개가 동시에 닿는다
    S = np.array(SV, float); ys = np.array([1, 1, -1])
    m3 = SVC(kernel="linear", C=1e6).fit(S, ys); assert np.allclose(m3.coef_[0], W, atol=1e-3) and abs(m3.intercept_[0] - B) < 1e-2   # 점 3개만으로 풀어도 같은 길
    wt, bt, pushed, _ = train_svm(X, y)
    assert np.allclose(wt, W, atol=3e-3) and abs(bt - B) < 1e-2 and round(2 / np.linalg.norm(wt), 2) == MARGIN
    first = {tuple(X[i]): int(np.argmax(pushed[1:, i])) + 2 for i in range(len(X)) if pushed[1:, i].any()}
    assert first == TOUCH and sorted(TOUCH) == sorted(SV)            # 1걸음 뒤로 민 점 = 서포트 벡터 3개뿐
    _, _, _, sh = train_svm(X, y, snaps=tuple(STEPS))
    for t, (a1, a2, bb, k) in STEPS.items():
        w_, b_, act = sh[t]; assert (round(w_[0], 2), round(w_[1], 2), round(b_, 2), int(act.sum())) == (a1, a2, bb, k), (t, w_, b_)
    assert tuple(np.round((y[:, None] * X).sum(0), 1)) == SUM_YX
    mg = y * (X @ wt + bt); assert sorted(map(tuple, X[mg < 1.02])) == sorted(SV) and mg[mg >= 1.02].min() > 1.2
    print("data ok")
