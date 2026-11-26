"""슬라이드·그림·실습이 공유하는 숫자 (lessons/ml1/regularization 쇼츠 초안과 같은 예시: 하루 기온 12번).
기온은 예시 (참 곡선 15 + 5·sin(2π·시/24) + 측정 잡음, 소수 첫째 자리). 특징 = 시각(0~24시 → 0~1)의 1~9제곱, 표준화.
λ 는 scikit-learn 의 alpha 그대로. 오차 = 0~24시 200곳에서 참 곡선과의 RMSE (도).
손 계산 예 (notebooks/ml1/chapter06_1 과 같음): X = [1, 2, 3], y = [3, 5, 7], 절편은 벌금 없음."""
import warnings
import numpy as np

HOURS = [0.1, 0.4, 1.0, 6.5, 13.0, 14.6, 15.3, 17.5, 19.5, 19.6, 21.9, 22.4]
TEMPS = [12.2, 15.2, 14.7, 19.0, 13.0, 11.5, 11.7, 11.3, 10.2, 12.1, 11.6, 13.5]
DEG, LAM = 9, 0.005
LAMS = [1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.1, 1, 10, 100]


def truth(t):
    return 15 + 5 * np.sin(2 * np.pi * np.asarray(t, float) / 24)


def make_model(reg, lam=LAM):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    m = {"none": LinearRegression(), "ridge": Ridge(lam), "lasso": Lasso(lam, max_iter=500000)}[reg]
    return make_pipeline(PolynomialFeatures(DEG, include_bias=False), StandardScaler(), m)


def fit(reg, lam=LAM):
    warnings.filterwarnings("ignore")
    return make_model(reg, lam).fit((np.array(HOURS) / 24)[:, None], np.array(TEMPS))


def err(model):
    tt = np.linspace(0, 24, 200)
    return float(np.sqrt(np.mean((model.predict((tt / 24)[:, None]) - truth(tt)) ** 2)))


MAXW = {"none": 33589, "ridge": 19, "lasso": 24}
ERR = {"none": 3.2, "ridge": 1.2, "lasso": 1.2}
TRAIN_ERR_NONE, TRAIN_ERR_RIDGE = 0.57, 0.86
LASSO_X2_ZERO = 0.13                                       # 계수 경로 예 (x1 진짜 · x2 잡음, 점 30개, 시드 0): 라쏘가 x2 를 0 으로 만드는 λ (약 0.13, 0.12 는 아직 양수 · 0.14 는 0)
LASSO_ZEROS = 5
SWEEP = [1.59, 0.99, 0.99, 1.23, 1.43, 2.33, 2.92, 3.39, 3.78]   # λ 별 오차 (도)
HAND = {0: (1, 2), 2: (3, 1), 10: (4.33, 0.33)}                # Ridge 손 계산 (w0, w1)

if __name__ == "__main__":
    M = {k: fit(k) for k in MAXW}
    assert {k: round(float(abs(m[-1].coef_).max())) for k, m in M.items()} == MAXW
    assert {k: round(err(m), 1) for k, m in M.items()} == ERR
    t = np.array(HOURS) / 24
    assert round(float(np.sqrt(np.mean((M["none"].predict(t[:, None]) - TEMPS) ** 2))), 2) == TRAIN_ERR_NONE
    assert round(float(np.sqrt(np.mean((M["ridge"].predict(t[:, None]) - TEMPS) ** 2))), 2) == TRAIN_ERR_RIDGE
    assert int((abs(M["lasso"][-1].coef_) < 1e-6).sum()) == LASSO_ZEROS
    from sklearn.linear_model import Lasso
    r = np.random.default_rng(0); x1 = r.uniform(-1, 1, 30); x2 = r.normal(0, 1, 30); yy = 3 * x1 + r.normal(0, 1, 30)
    assert Lasso(alpha=0.12).fit(np.c_[x1, x2], yy).coef_[1] > 0 and abs(Lasso(alpha=LASSO_X2_ZERO + 0.01).fit(np.c_[x1, x2], yy).coef_[1]) < 1e-9
    assert [round(err(fit("ridge", l)), 2) for l in LAMS] == SWEEP
    X = np.c_[np.ones(3), [1, 2, 3]]; y = np.array([3, 5, 7.]); D = np.diag([0, 1])
    for l, w in HAND.items():
        assert tuple(np.linalg.solve(X.T @ X + l * D, X.T @ y).round(2)) == w
    print("data ok")
