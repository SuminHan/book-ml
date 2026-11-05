"""슬라이드·그림·실습이 공유하는 숫자 (notebooks/ml1/chapter06_2 를 바탕으로, λ 고르기 예시는 이 덱용으로 새로 만듦).
손 계산: y = 1..10, 평균만 내는 모델, 순서대로 2개씩 5조각.
λ 고르기: 학생 60명 × 특징 30개 (진짜로 쓰이는 건 3개: y = 3·x1 − 2·x2 + 1.5·x3 + 잡음 σ=3), 시드 4. 릿지 (절편 제외).
'실제 오차' = 같은 규칙으로 새로 만든 2만 개에서 잰 MSE (예시라서 잴 수 있다)."""
import numpy as np

Y10 = list(range(1, 11))
HAND_FOLDS = [25.25, 6.5, 0.25, 6.5, 25.25]
HAND_CV, HAND_TRUE = 12.75, 8.25
N, D, K = 60, 30, 5
LAMS = np.logspace(-2, 3, 51)


def make_data(seed, n=N):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (n, D)); y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] + r.normal(0, 3, n)
    return X, y


def fit_ridge(X, y, lam):
    x0, y0 = X.mean(0), y.mean(); Xc, yc = X - x0, y - y0
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc)
    return w, y0 - w @ x0


def mse(wb, X, y):
    w, b = wb; return float(((X @ w + b - y) ** 2).mean())


def cv_errors(X, y, lam, k=K):
    m = len(y); out = []
    for i in range(k):
        va = np.arange(i, m, k); tr = np.setdiff1d(np.arange(m), va)
        out.append(mse(fit_ridge(X[tr], y[tr], lam), X[va], y[va]))
    return out


LAM_BEST, CV_BEST, CV_SD = 12.6, 15.69, 5.54
FOLDS_BEST = [19.1, 8.3, 12.4, 24.3, 14.3]
CV_EDGE = {0.01: 26.73, 1000: 28.25}
TEST = {0.01: 12.78, 12.6: 11.56, 1000: 22.5}
ONE_SE = (79.4, 14.43)                                            # 1SE 규칙이 고르는 λ 와 그 λ 의 실제 오차                     # 실제 오차 (새 데이터 2만 개)
SINGLE = [63.1, 39.81, 7.94, 19.95, 25.12, 7.94, 158.49, 15.85, 19.95, 2.51]   # 48/12 로 한 번만 나눴을 때 고른 λ (10번)
SEEDS = [7.9, 15.8, 50.1, 31.6, 12.6, 6.3, 25.1, 10.0, 20.0, 15.8]            # 데이터를 새로 뽑을 때마다 5겹 CV 가 고른 λ

if __name__ == "__main__":
    folds = []
    for f in range(5):
        val = Y10[2 * f:2 * f + 2]; tr = Y10[:2 * f] + Y10[2 * f + 2:]; mu = sum(tr) / len(tr)
        folds.append(sum((v - mu) ** 2 for v in val) / 2)
    assert folds == HAND_FOLDS and sum(folds) / 5 == HAND_CV and np.mean((np.array(Y10) - 5.5) ** 2) == HAND_TRUE
    X, y = make_data(4)
    E = np.array([cv_errors(X, y, l) for l in LAMS]); mu = E.mean(1); b = int(mu.argmin())
    assert (round(float(LAMS[b]), 1), round(float(mu[b]), 2), round(float(E[b].std()), 2)) == (LAM_BEST, CV_BEST, CV_SD)
    assert [round(v, 1) for v in E[b]] == FOLDS_BEST and (round(float(mu[0]), 2), round(float(mu[-1]), 2)) == tuple(CV_EDGE.values())
    r = np.random.default_rng(4); r.normal(0, 1, (N, D)); r.normal(0, 3, N)
    Xt = r.normal(0, 1, (20000, D)); yt = 3 * Xt[:, 0] - 2 * Xt[:, 1] + 1.5 * Xt[:, 2] + r.normal(0, 3, 20000)
    T = [round(mse(fit_ridge(X, y, l), Xt, yt), 2) for l in (LAMS[0], LAMS[b], LAMS[-1])]
    assert T == list(TEST.values()), T
    ok = int(max(np.where(mu <= mu[b] + E[b].std())[0]))
    assert (round(float(LAMS[ok]), 1), round(mse(fit_ridge(X, y, LAMS[ok]), Xt, yt), 2)) == ONE_SE
    assert round(float(LAMS[np.argmin([mse(fit_ridge(X, y, l), Xt, yt) for l in LAMS])]), 1) == LAM_BEST   # 실제로도 가장 좋은 λ
    single = []
    for s in range(10):
        p = np.random.default_rng(100 + s).permutation(N); tr, va = p[:48], p[48:]
        single.append(round(float(LAMS[int(np.argmin([mse(fit_ridge(X[tr], y[tr], l), X[va], y[va]) for l in LAMS]))]), 2))
    assert single == SINGLE
    seeds = []
    for s in range(1, 11):
        Xs, ys = make_data(1000 + s); seeds.append(round(float(LAMS[int(np.argmin([np.mean(cv_errors(Xs, ys, l)) for l in LAMS]))]), 1))
    assert seeds == SEEDS
    print("data ok")
