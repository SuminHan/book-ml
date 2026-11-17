"""슬라이드·그림·실습이 공유하는 숫자. 덱 전체가 z-score 좌표 하나만 쓴다.
원래 점 (RAW, 0~10 눈금 예시 값, 손으로 고름) → z = (x − 평균) / 표준편차, 소수 첫째 자리로 반올림한 값이 덱의 점 20개.
최대 마진 (‖w‖ = 1): w = (−1, 1)/√2, b = 0.05/√2 = 0.04, m = 0.95/√2 = 0.67 (경계 x2 − x1 = −0.05, 가장자리 x2 − x1 = 0.9 / −1.0).
(1걸음 뒤 처음 미는 점이 빨강이 되도록, 옛 점을 (10 − x) 로 뒤집고 색을 바꾼 것. 옛 판 _이전/data_v5_파랑먼저.py)"""
import numpy as np

RAW_RED = [(4.0, 7.5), (3.0, 6.0), (4.5, 8.5), (2.0, 8.0), (3.5, 6.5), (2.5, 9.0), (5.0, 7.0), (1.5, 6.5), (4.0, 9.2), (2.2, 5.0)]
RAW_BLUE = [(8.5, 4.0), (8.0, 2.5), (7.5, 4.5), (7.0, 3.0), (9.0, 2.0), (6.5, 4.0), (8.0, 5.5), (7.0, 1.5), (8.8, 4.8), (6.0, 2.2)]
RAW_MU, RAW_SD = (5.42, 5.36), (2.43, 2.34)               # 원래 점 20개의 평균 · 표준편차 (축마다)

RED_PTS = [(-0.6, 0.9), (-1.0, 0.3), (-0.4, 1.3), (-1.4, 1.1), (-0.8, 0.5), (-1.2, 1.6), (-0.2, 0.7), (-1.6, 0.5), (-0.6, 1.6), (-1.3, -0.2)]
BLUE_PTS = [(1.3, -0.6), (1.1, -1.2), (0.9, -0.4), (0.6, -1.0), (1.5, -1.4), (0.4, -0.6), (1.1, 0.1), (0.6, -1.6), (1.4, -0.2), (0.2, -1.3)]
W_ANS, B_ANS, M_ANS = (-1 / np.sqrt(2), 1 / np.sqrt(2)), 0.05 / np.sqrt(2), 0.95 / np.sqrt(2)   # (−0.71, 0.71), 0.04, 0.67
SV = [(-0.2, 0.7), (0.4, -0.6), (1.1, 0.1)]             # 서포트 벡터: 빨강 하나, 파랑 둘
ALPHA = {(-0.2, 0.7): 0.74, (0.4, -0.6): -0.37, (1.1, 0.1): -0.37}   # ‖w‖ = 1 기준 w = Σ α_i x_i (부호 포함)
CENTROID_D = (0.61, 0.70)                                 # 평균 한가운데 선에서 가장 가까운 빨강 · 파랑까지 거리
FAR_MOVE = ((1.5, -1.4), (1.8, -2.1))                     # 멀리 있는 파랑 점을 옮겨도 경계는 그대로
OUTLIER = (0.6, -0.2)                                     # 소프트 마진: 파랑 무리 안에 들어간 빨강 점
NEWPT, M_NEW = (0.8, 0.4), 0.21                           # 최대 마진의 약점: 빨간 점 하나 추가 → m 0.67 → 0.21
BUDGET = {100: (0.0, 0.07, 3), 3: (1.79, 0.67, 4), 0.3: (2.47, 0.95, 8)}   # sklearn C: (예산 B = Σε, m, 서포트 벡터 수) — 이상치 포함
HYPER = (1, 2, 3)                                         # 초평면 예: 1 + 2 x1 + 3 x2 = 0 (ISL 그림 9.1 의 식)
KX, KY = (1, 2), (3, 1)                                   # 커널 계산 예: (x·x')² = φ(x)·φ(x') = 25

# 학습 (‖w‖ = 1): 매 걸음 값이 가장 작은 점이 민다 → w 길이가 1 넘으면 1 로 나눈다. 학습률 하나 η = 0.05/√걸음
MM_T, MM_LR = 20000, 0.05
SUM_YX = (-18.2, 16.5)                                    # 1걸음: 20개 모두 밀 때 Σ y_i x_i = 빨강 합 − 파랑 합
STEP1_W = (-0.74, 0.67)                                   # 0.05 × (−18.2, 16.5) = (−0.91, 0.83), 길이 1.23 → 1 로
MM_FIRST = {(-0.2, 0.7): 2, (0.4, -0.6): 4, (1.1, 0.1): 10}   # 1걸음 뒤 처음 민 걸음 (나머지 17개는 한 번도 안 민다)
MM_STEPS = {1: (0, 0, 0, 0.0, 20), 2: (-0.74, 0.67, 0.0, 0.62, 1), 3: (-0.73, 0.68, 0.03, 0.66, 1),
            4: (-0.72, 0.69, 0.06, 0.64, 1), 200: (-0.71, 0.71, 0.04, 0.67, 1), 20000: (-0.71, 0.71, 0.04, 0.67, 1)}   # 걸음: (w1, w2, b, m, 미는 점 수), 그 걸음 직전
SHIFT = 100                                               # z-score 안 하면: 원래 점의 x2 에 +100 → 그대로 학습하면 실패


def train_maxmin(X, y, T=MM_T, lr0=MM_LR, w0=(0, 0), b0=0.0, snaps=()):
    """‖w‖ = 1 로 맞춘 채 가장 가까운 점까지 거리 m = min y_i (w·x_i + b) 를 키운다.
    한 걸음: 값이 가장 작은 점(들) k 가 w' = w + η Σ y_k x_k, b' = b + η Σ y_k  → ‖w'‖ > 1 이면 w, b 를 ‖w'‖ 로 나눈다.
    반환: 마지막 (w, b), 걸음마다 민 점 (T × n), snaps 걸음 직전의 (w, b, 미는 점, m)."""
    w, b = np.array(w0, float), float(b0)
    pushed, shots = np.zeros((T, len(X)), bool), {}
    for t in range(1, T + 1):
        v = y * (X @ w + b); m = v.min(); act = v <= m + 1e-9
        pushed[t - 1] = act
        if t in snaps:
            shots[t] = (w.copy(), b, act.copy(), m)
        eta = lr0 / np.sqrt(t)
        w, b = w + eta * (y[act, None] * X[act]).sum(0), b + eta * y[act].sum()
        n = np.linalg.norm(w)
        if n > 1:
            w, b = w / n, b / n
    return w, b, pushed, shots


def phi(v):
    return np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])


def unit(m):
    """sklearn 선형 SVC → ‖w‖ = 1 기준 (w, b, m)"""
    n = np.linalg.norm(m.coef_[0]); return m.coef_[0] / n, m.intercept_[0] / n, 1 / n


if __name__ == "__main__":
    from sklearn.svm import SVC
    R, Bl = np.array(RAW_RED), np.array(RAW_BLUE); A = np.r_[R, Bl]
    assert tuple(np.round(A.mean(0), 2)) == RAW_MU and tuple(np.round(A.std(0), 2)) == RAW_SD
    assert np.allclose(np.round((A - A.mean(0)) / A.std(0), 1), np.array(RED_PTS + BLUE_PTS))
    X = np.array(RED_PTS + BLUE_PTS, float); y = np.r_[np.ones(10), -np.ones(10)]
    assert np.allclose(X.mean(0), 0, atol=0.01) and np.allclose(X.std(0), 1, atol=0.02)
    m = SVC(kernel="linear", C=1e6).fit(X, y); w, b, mm = unit(m)
    assert np.allclose(w, W_ANS, atol=1e-3) and abs(b - B_ANS) < 1e-3 and abs(mm - M_ANS) < 1e-3
    assert sorted(map(tuple, X[m.support_])) == sorted(SV)
    al = {tuple(X[i]): round(c * mm, 2) for i, c in zip(m.support_, m.dual_coef_[0])}; assert al == ALPHA
    assert np.allclose(sum(v * np.array(k) for k, v in ALPHA.items()), W_ANS, atol=0.02)
    mr, mb = X[y > 0].mean(0), X[y < 0].mean(0); u = (mr - mb) / np.linalg.norm(mr - mb); bc = -u @ (mr + mb) / 2
    v = y * (X @ u + bc); assert (round(v[y > 0].min(), 2), round(v[y < 0].min(), 2)) == CENTROID_D
    X2 = X.copy(); X2[10 + BLUE_PTS.index(FAR_MOVE[0])] = FAR_MOVE[1]
    assert np.allclose(SVC(kernel="linear", C=1e6).fit(X2, y).coef_, m.coef_, atol=1e-3)
    assert round(unit(SVC(kernel="linear", C=1e6).fit(np.r_[X, [NEWPT]], np.r_[y, 1]))[2], 2) == M_NEW
    Xo = np.r_[X, [OUTLIER]]; yo = np.r_[y, [1]]
    for C, (bud, mg, nsv) in BUDGET.items():
        mb_ = SVC(kernel="linear", C=C).fit(Xo, yo); e = np.maximum(0, 1 - yo * mb_.decision_function(Xo))
        assert (round(e.sum(), 2), round(unit(mb_)[2], 2), len(mb_.support_)) == (bud, mg, nsv), C
    assert np.dot(KX, KY) ** 2 == 25 and round(float(phi(KX) @ phi(KY)), 6) == 25
    assert tuple(np.round((y[:, None] * X).sum(0), 1)) == SUM_YX
    s1 = MM_LR * np.array(SUM_YX); assert tuple(np.round(s1 / np.linalg.norm(s1), 2)) == STEP1_W
    wm, bm, pm, _ = train_maxmin(X, y)
    assert np.allclose(wm, W_ANS, atol=5e-3) and abs(bm - B_ANS) < 5e-3 and round((y * (X @ wm + bm)).min(), 2) == round(M_ANS, 2)
    firstm = {tuple(X[i]): int(np.argmax(pm[1:, i])) + 2 for i in range(len(X)) if pm[1:, i].any()}
    assert firstm == MM_FIRST and sorted(MM_FIRST) == sorted(SV)
    _, _, _, shm = train_maxmin(X, y, snaps=tuple(MM_STEPS))
    for t, (a1, a2, bb, m_, k) in MM_STEPS.items():
        w_, b_, act, mt = shm[t]; assert (round(w_[0], 2), round(w_[1], 2), round(b_, 2) + 0.0, round(mt, 2), int(act.sum())) == (a1, a2, bb, m_, k), (t, w_, b_, mt)
    print("data ok")
