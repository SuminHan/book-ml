"""슬라이드·그림·실습이 공유하는 숫자. 덱 전체가 z-score 좌표 하나만 쓴다.
원래 점 (RAW, 예시 값: x1 은 3~7, x2 는 0~10 으로 축마다 퍼진 정도가 다르다) → z = (x − 평균) / 표준편차, 소수 첫째 자리로 반올림한 값이 덱의 점 20개.
최대 마진 (‖w‖ = 1): w = (1, 1)/√2, b = 0.05/√2 = 0.04, m = 0.95/√2 = 0.67 (경계 x1 + x2 = −0.05, 가장자리 x1 + x2 = 0.9 / −1.0).
(빨강 = 1사분면, 파랑 = 3사분면. 1걸음 뒤 처음 미는 점은 빨강. 옛 판 _이전/data_v6_빨강왼쪽위.py 를 시계 방향 90° 돌린 것)"""
import numpy as np

RAW_RED = [(5.9, 6.8), (5.3, 8.0), (6.3, 6.2), (6.1, 9.2), (5.5, 7.4), (6.6, 8.6), (5.7, 5.6), (5.5, 9.8), (6.6, 6.8), (4.8, 8.9)]
RAW_BLUE = [(4.4, 1.1), (3.8, 1.7), (4.6, 2.3), (4.0, 3.2), (3.6, 0.5), (4.4, 3.8), (5.1, 1.7), (3.4, 3.2), (4.8, 0.8), (3.7, 4.4)]
RAW_MU, RAW_SD = (5.0, 5.0), (0.99, 3.02)               # 원래 점 20개의 평균 · 표준편차 (축마다)

RED_PTS = [(0.9, 0.6), (0.3, 1.0), (1.3, 0.4), (1.1, 1.4), (0.5, 0.8), (1.6, 1.2), (0.7, 0.2), (0.5, 1.6), (1.6, 0.6), (-0.2, 1.3)]
BLUE_PTS = [(-0.6, -1.3), (-1.2, -1.1), (-0.4, -0.9), (-1.0, -0.6), (-1.4, -1.5), (-0.6, -0.4), (0.1, -1.1), (-1.6, -0.6), (-0.2, -1.4), (-1.3, -0.2)]
W_ANS, B_ANS, M_ANS = (1 / np.sqrt(2), 1 / np.sqrt(2)), 0.05 / np.sqrt(2), 0.95 / np.sqrt(2)   # (0.71, 0.71), 0.04, 0.67
SV = [(0.7, 0.2), (-0.6, -0.4), (0.1, -1.1)]             # 서포트 벡터: 빨강 하나, 파랑 둘
ALPHA = {(0.7, 0.2): 0.74, (-0.6, -0.4): -0.37, (0.1, -1.1): -0.37}   # ‖w‖ = 1 기준 w = Σ α_i x_i (부호 포함)
MEAN_W = (0.83, 0.91)                                    # 가장 쉬운 방법: w = 20개의 y_i x_i 평균 (빨강은 그대로, 파랑은 부호 뒤집어), b = 0
MEAN_D = (0.62, 0.70)                                     # 그 선에서 가장 가까운 빨강 · 파랑까지 거리
FAR_MOVE = ((-1.4, -1.5), (-2.1, -1.8))                     # 멀리 있는 파랑 점을 옮겨도 경계는 그대로
OUTLIER = (-0.2, -0.6)                                     # 소프트 마진: 파랑 무리 안에 들어간 빨강 점
NEWPT, M_NEW = (0.4, -0.8), 0.21                           # 최대 마진의 약점: 빨간 점 하나 추가 → m 0.67 → 0.21
BUDGET = {100: (0.0, 0.07, 3), 3: (1.79, 0.67, 4), 0.3: (2.47, 0.95, 8)}   # sklearn C: (예산 B = Σε, m, 서포트 벡터 수) — 이상치 포함
HYPER = (1, 2, 3)                                         # 초평면 예: 1 + 2 x1 + 3 x2 = 0 (ISL 그림 9.1 의 식)
KX, KY = (1, 2), (3, 1)                                   # 커널 계산 예: (x·x')² = φ(x)·φ(x') = 25

# 학습 (‖w‖ = 1): 매 걸음 값이 가장 작은 점이 민다 → w 길이가 1 넘으면 w 만 길이로 나눈다 (b 는 그대로). 학습률 하나 η = 0.05/√걸음
MM_T, MM_LR = 20000, 0.05
SUM_YX = (16.5, 18.2)                                    # 1걸음: 20개 모두 밀 때 Σ y_i x_i = 빨강 합 − 파랑 합
STEP1_W = (0.67, 0.74)                                    # 0.05 × (16.5, 18.2) = (0.83, 0.91), 길이 1.23 → 1 로
MM_FIRST = {(0.7, 0.2): 2, (-0.6, -0.4): 4, (0.1, -1.1): 11}   # 1걸음 뒤 처음 민 걸음 (나머지 17개는 한 번도 안 민다)
MM_STEPS = {1: (0, 0, 0, 0.0, 20), 2: (0.67, 0.74, 0.0, 0.62, 1), 3: (0.68, 0.73, 0.04, 0.66, 1),
            4: (0.69, 0.72, 0.06, 0.64, 1), 11: (0.71, 0.7, 0.04, 0.67, 1), 20000: (0.71, 0.71, 0.04, 0.67, 1)}   # 걸음: (w1, w2, b, m, 미는 점 수), 그 걸음 직전
SHIFT = 100                                               # z-score 안 하면: 원래 점의 x2 에 +100 → 그대로 학습하면 실패


def train_maxmin(X, y, T=MM_T, lr0=MM_LR, w0=(0, 0), b0=0.0, snaps=()):
    """‖w‖ = 1 로 맞춘 채 가장 가까운 점까지 거리 m = min y_i (w·x_i + b) 를 키운다.
    한 걸음: 값이 가장 작은 점(들) k 가 w' = w + η Σ y_k x_k, b' = b + η Σ y_k  → ‖w'‖ > 1 이면 w 만 ‖w'‖ 로 나눈다 (b 는 그대로).
    (w 와 b 를 둘 다 나누면 b 가 매 걸음 0 쪽으로 끌려가 정답 b 가 클 때 답에 못 간다: 무작위 데이터 200개 중 17개 실패, verify_sklearn.py. 옛 판 _이전/data_v11_wb둘다나눔.py)
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
            w = w / n
    return w, b, pushed, shots


def train_soft(X, y, lam, T=MM_T, lr0=MM_LR):
    """소프트 마진 (‖w‖ = 1): m − λ Σ max(0, m − y_i(w·x_i + b)) 를 최대로. 길 안 (값 < m) 점들 V 가 다 같이 민다.
    w' = w + ηλ Σ_V y_i x_i, b' = b + ηλ Σ_V y_i, m' = m + η(1 − λ|V|)  → ‖w'‖ > 1 이면 w 만 길이로 나눈다.
    (첫 걸음은 하드 마진처럼 값이 가장 작은 점들 = 20개 모두). λ = 1/3 은 sklearn C = 3 과 같은 답 (아래 assert)"""
    w, b, m = np.zeros(X.shape[1]), 0.0, 0.0
    for t in range(1, T + 1):
        v = y * (X @ w + b); V = v <= v.min() + 1e-9 if t == 1 else v < m
        eta = lr0 / np.sqrt(t)
        w = w + eta * lam * (y[V, None] * X[V]).sum(0); b = b + eta * lam * y[V].sum(); m = m + eta * (1 - lam * V.sum())
        n = np.linalg.norm(w)
        if n > 1:
            w = w / n
    return w, b, m


SOFT = {3: (1 / 3, 0.672, 1.79), 0.5: (1 / 5, 0.930, 2.41)}   # sklearn C: (λ, m, 예산 B) — 이상치 포함 21개, 직접 짠 소프트 마진 학습과 같은 선


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
    a = (y[:, None] * X).mean(0); assert np.allclose(a, MEAN_W, atol=0.006); u = a / np.linalg.norm(a)
    v = y * (X @ u); assert (round(v[y > 0].min(), 2), round(v[y < 0].min(), 2)) == MEAN_D
    assert tuple(np.round(u, 2)) == STEP1_W                        # 가장 쉬운 방법의 선 = 학습 1걸음 뒤 첫 선
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
    for C, (lam, mg, bud) in SOFT.items():
        ws, bs, ms = unit(SVC(kernel="linear", C=C).fit(Xo, yo)); wl, bl, ml = train_soft(Xo, yo, lam)
        assert np.allclose(wl, ws, atol=3e-3) and abs(bl - bs) < 3e-3 and abs(ml - ms) < 3e-3 and round(ms, 3) == mg, (C, wl, bl, ml, ws, bs, ms)
        assert round(np.maximum(0, 1 - yo * (Xo @ ws + bs) / ms).sum(), 2) == bud
    print("data ok")
