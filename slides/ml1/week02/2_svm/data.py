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


# 학습 (‖w‖ = 1 기준, 덱 전체가 이 기준): 매 걸음 가장 가까운 점(값 최소)이 민다 → w 길이가 1 넘으면 1 로 나눈다
MM_T, MM_LR, MM_SB = 20000, 0.05, 5                      # 걸음 수, 학습률 0.05/√t, b 는 5배 (b 가 너무 느려서)
MM_FIRST = {(3.5, 6.0): 2, (7.8, 5.0): 3, (5.0, 3.0): 5, (2.0, 4.5): 141}   # 1걸음 뒤 처음 민 걸음 (나머지 16개는 한 번도 안 민다)
MM_STEPS = {1: (0, 0, 0, 0.0, 20), 2: (-0.75, 0.66, 0, 1.37, 1), 3: (-0.58, 0.81, 0.16, 0.28, 1),
            5: (-0.65, 0.76, 0.14, 0.85, 1), 200: (-0.71, 0.71, -0.19, 1.58, 1), 20000: (-0.71, 0.71, -0.18, 1.58, 1)}   # 걸음: (w1, w2, b, m, 미는 점 수), 그 걸음 직전
MM_SUM = 0.05 * 44.1                                      # 1걸음: 0.05 × (−44.1, 39.2) 의 길이 2.95 > 1 → 1 로 나눔


def train_maxmin(X, y, T=MM_T, lr0=MM_LR, sb=MM_SB, w0=(0, 0), b0=0.0, snaps=()):
    """‖w‖ = 1 로 맞춘 채 가장 가까운 점까지 거리 m = min y_i (w·x_i + b) 를 키운다.
    한 걸음: 값이 가장 작은 점(들) k 가 w += η Σ y_k x_k, b += sb·η Σ y_k  → ‖w‖ > 1 이면 w, b 를 ‖w‖ 로 나눈다.
    반환: 마지막 (w, b), 걸음마다 민 점 (T × n), snaps 걸음 직전의 (w, b, 미는 점, m)."""
    w, b = np.array(w0, float), float(b0)
    pushed, shots = np.zeros((T, len(X)), bool), {}
    for t in range(1, T + 1):
        v = y * (X @ w + b); m = v.min(); act = v <= m + 1e-9
        pushed[t - 1] = act
        if t in snaps:
            shots[t] = (w.copy(), b, act.copy(), m)
        eta = lr0 / np.sqrt(t)
        w, b = w + eta * (y[act, None] * X[act]).sum(0), b + sb * eta * y[act].sum()
        n = np.linalg.norm(w)
        if n > 1:
            w, b = w / n, b / n
    return w, b, pushed, shots


# CME 250 Lecture 5 (ISL 9장) 표기: ‖β‖ = 1 로 맞춘 경계 β0 + β·x = 0, 마진 M = 경계에서 가장 가까운 점까지 거리 (길 폭 = 2M)
BETA, BETA0, M_HARD = (-0.71, 0.71), -0.18, 1.59           # β = (−1, 1)/√2, β0 = −1/(4√2), M = 9/(4√2)
ALPHA = {(5.0, 3.0): 16 / 81, (3.5, 6.0): 8 / 81, (2.0, 4.5): 8 / 81}   # w = Σ α_i y_i x_i (서포트 벡터만, 나머지 α = 0)
NEWPT, M_NEW = (2.6, 3.8), 0.46                          # 최대 마진의 약점: 파란 점 하나 추가 → M 1.59 → 0.46, 경계가 기운다
BUDGET = {100: (0.0, 0.11, 3), 1: (1.87, 1.59, 4), 0.05: (2.64, 2.32, 8)}   # sklearn C: (예산 B = Σε, M, 서포트 벡터 수) — 파랑 이상치 포함
HYPER = (1, 2, 3)                                        # 초평면 예: 1 + 2 x1 + 3 x2 = 0 (ISL 그림 9.1 의 식)


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
    nw = np.linalg.norm(m.coef_[0])
    assert tuple(np.round(m.coef_[0] / nw, 2)) == BETA and round(m.intercept_[0] / nw, 2) in (BETA0, -0.17) and round(1 / nw, 2) == M_HARD
    assert abs(9 / (4 * np.sqrt(2)) - 1.591) < 1e-3 and abs(-1 / (4 * np.sqrt(2)) - BETA0) < 0.01
    al = {tuple(X[i]): abs(c) for i, c in zip(m.support_, m.dual_coef_[0])}
    assert all(abs(al[k] - v) < 2e-3 for k, v in ALPHA.items())
    wa = sum(v * (1 if k[1] > k[0] else -1) * np.array(k) for k, v in ALPHA.items()); assert np.allclose(wa, W)
    mn = SVC(kernel="linear", C=1e6).fit(np.r_[X, [NEWPT]], np.r_[y, -1]); assert round(1 / np.linalg.norm(mn.coef_[0]), 2) == M_NEW
    for C, (bud, mg, nsv) in BUDGET.items():
        mb = SVC(kernel="linear", C=C).fit(Xo, yo); e = np.maximum(0, 1 - yo * mb.decision_function(Xo))
        assert (round(e.sum(), 2), round(1 / np.linalg.norm(mb.coef_[0]), 2), len(mb.support_)) == (bud, mg, nsv), C
    wm, bm, pm, _ = train_maxmin(X, y)
    assert tuple(np.round(wm, 2)) == BETA and round(bm, 2) == BETA0 and round((y * (X @ wm + bm)).min(), 2) == 1.58
    firstm = {tuple(X[i]): int(np.argmax(pm[1:, i])) + 2 for i in range(len(X)) if pm[1:, i].any()}
    assert firstm == MM_FIRST
    _, _, _, shm = train_maxmin(X, y, snaps=tuple(MM_STEPS))
    for t, (a1, a2, bb, mm, k) in MM_STEPS.items():
        w_, b_, act, m_ = shm[t]; assert (round(w_[0], 2), round(w_[1], 2), round(b_, 2), round(m_, 2), int(act.sum())) == (a1, a2, bb, mm, k), (t, w_, b_, m_)
    vm = y * (X @ wm + bm); assert sorted(map(tuple, X[vm < 1.6])) == sorted(SV) and vm[vm >= 1.6].min() > 1.9
    print("data ok")
