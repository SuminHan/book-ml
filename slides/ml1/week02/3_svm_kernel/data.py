"""커널 SVM 덱의 숫자. 앞 차시 덱(../2_svm/data.py)의 숫자는 그대로 불러 쓰고, 이 덱에서 새로 쓰는 숫자만 여기서 계산 · assert 한다."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2_svm"))
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# 1차원 → 2차원 (x → (x, x²)): 빨강이 가운데, 파랑이 양쪽 (예시)
L1_RED = [-0.5, -0.2, 0.3, 0.6]
L1_BLUE = [-1.7, -1.3, 1.2, 1.6]
L1_H = 0.5                                    # 위에서 긋는 직선: 높이 x² = 0.5
L1_CUT = float(np.sqrt(L1_H))                 # 원래 수직선에서 끊는 곳 ±0.71
assert all(x * x < L1_H for x in L1_RED) and all(x * x > L1_H for x in L1_BLUE)
assert round(L1_CUT, 2) == 0.71


def phi6(x):
    """다항 2차 커널 (x·x' + 1)² 의 특징 6개"""
    r2 = np.sqrt(2)
    return np.array([1, r2 * x[0], r2 * x[1], x[0] ** 2, r2 * x[0] * x[1], x[1] ** 2])


# 1차원 커널 식 풀기: 학습 점 0 (빨강, α = +1), 2 (파랑, α = −0.25), b = 0
P1_X = [0.0, 2.0]; P1_A = [1.0, -0.25]
P1_COEF = [sum(a * c for a, c in zip(P1_A, [1, 1])), sum(a * 2 * x for a, x in zip(P1_A, P1_X)), sum(a * x * x for a, x in zip(P1_A, P1_X))]   # 1, x, x² 앞의 수
assert P1_COEF == [0.75, -1.0, -1.0]
_f1 = lambda x: sum(a * (1 + x * xi) ** 2 for a, xi in zip(P1_A, P1_X))
assert _f1(1) == -1.25 == P1_COEF[0] + P1_COEF[1] + P1_COEF[2]
P1_ROOTS = sorted(np.roots(P1_COEF[::-1]).real.round(6).tolist())
assert P1_ROOTS == [-1.5, 0.5] and _f1(0) > 0 and _f1(-2) < 0 and _f1(1) < 0

import data as _d2                                   # ../2_svm/data.py

# 서포트 벡터 2개 예 (../2_svm 'sv_two_three' 의 작은 예, 점 8개, y D = 1 약속): 서포트 벡터 빨강 (0.5, 0.5) · 파랑 (−0.5, −0.5), w* = (1, 1), b* = 0
TOY2_RED = [(0.5, 0.5), (1.5, 1.5), (0.5, 1.6), (1.6, 0.6)]
TOY2_BLUE = [(-0.5, -0.5), (-1.5, -1.2), (-0.8, -1.6), (-1.6, -0.3)]
_Xt = np.array(TOY2_RED + TOY2_BLUE, float); _yt = np.r_[np.ones(4), -np.ones(4)]
_mt = SVC(kernel="linear", C=1e6).fit(_Xt, _yt)
TOY2_W, TOY2_B = _mt.coef_[0].round(3), round(float(_mt.intercept_[0]), 3)
TOY2_SV = [tuple(_Xt[i]) for i in _mt.support_]
assert len(TOY2_SV) == 2 and np.allclose(TOY2_W, [1, 1], atol=2e-3) and abs(TOY2_B) < 2e-3

# 서포트 벡터 몇 개? 겹치지 않는 2차원 점 20개 (두 무리 N((±1.2, ±1.2), 0.6²)) 를 2,000 번 만들어 하드 마진 SVM: 2개 · 3개 · 4개 횟수
# (python data.py 로 다시 센다. 2개 = 빨강 1 · 파랑 1, 3개 = 2 + 1 또는 1 + 2, 4개 이상 = 점이 한 직선에 놓인 우연)
SV_COUNT = {2: 922, 3: 1073, 4: 4}
def sv_count_sim(n_trial=2000, seed=0):
    rng = np.random.default_rng(seed); c = {}
    for _ in range(n_trial):
        X = np.r_[rng.normal([1.2, 1.2], 0.6, (10, 2)), rng.normal([-1.2, -1.2], 0.6, (10, 2))]; y = np.r_[np.ones(10), -np.ones(10)]
        m = SVC(kernel="linear", C=1e6).fit(X, y)
        if (y * m.decision_function(X)).min() < 0.999:
            continue
        k = int((np.abs(m.dual_coef_[0]) > 1e-6).sum()); c[k] = c.get(k, 0) + 1
    return c

# 라그랑주 승수법 예 (등식 조건, 언덕에서 조건 위 가장 높은 점): max f = 4 − ½[(x − 2)² + (y − 2)²]  s.t.  g = x + y − 2 = 0,  L = f − α g
# ∂L/∂x = −(x − 2) − α = 0, ∂L/∂y = −(y − 2) − α = 0, ∂L/∂α = −g = 0  →  x = y = 2 − α, 4 − 2α = 2  →  α = 1, (x, y) = (1, 1), f = 3 (꼭대기 (2, 2) 는 4)
LG_A, LG_X, LG_Y, LG_C, LG_TOP = 1.0, 1.0, 1.0, (2.0, 2.0), 4.0
_lgf = lambda x, y: LG_TOP - 0.5 * ((x - LG_C[0]) ** 2 + (y - LG_C[1]) ** 2)
assert -(LG_X - 2) - LG_A == 0 and -(LG_Y - 2) - LG_A == 0 and LG_X + LG_Y - 2 == 0
_lgx = np.linspace(-5, 5, 100001); assert abs(_lgx[np.argmax(_lgf(_lgx, 2 - _lgx))] - LG_X) < 1e-3   # 직선 위에서 f 가 가장 큰 곳
LG_F = _lgf(LG_X, LG_Y)                                                       # 3
assert LG_F == 3

# 1/19 다음 쪽: 지난 시간 학습 (train_maxmin) 을 그대로 돌리면서, 걸음마다 민 점(argmin)에 η 를 적어 둔다 → w = Σ α_k y_k x_k 가 늘 성립
def train_maxmin_alpha(X, y, T=_d2.MM_T, lr0=_d2.MM_LR, snaps=()):
    """train_maxmin 과 같은 걸음. α_k += η (k = 민 점), w 를 ‖w‖ 로 나눌 때 α 도 같이 나눈다. 반환: w, b, α, snaps 걸음 뒤의 α."""
    a, b, shots = np.zeros(len(X)), 0.0, {}
    w = np.zeros(X.shape[1])
    for t in range(1, T + 1):
        v = y * (X @ w + b); act = v <= v.min() + 1e-9
        eta = lr0 / np.sqrt(t)
        a[act] += eta; b += eta * y[act].sum()
        w = (a * y) @ X; n = np.linalg.norm(w)
        if n > 0:
            a, w = a / n, w / n
        if t in snaps:
            shots[t] = a.copy()
    return w, b, a, shots

_XA = np.array(_d2.RED_PTS + _d2.BLUE_PTS, float); _YA = np.r_[np.ones(10), -np.ones(10)]
_wa, _ba, _aa, _sa = train_maxmin_alpha(_XA, _YA, snaps=(1, 11, 100, 1000, _d2.MM_T))
_ia = [(_d2.RED_PTS + _d2.BLUE_PTS).index(p) for p in _d2.SV]
ALPHA_TRACE = {t: tuple(s[_ia].round(2).tolist()) + (round(float(np.delete(s, _ia).sum()), 2),) for t, s in _sa.items()}   # 걸음 뒤: (α_r, α_b1, α_b2, 나머지 17개 합)
assert ALPHA_TRACE == {1: (0.04, 0.04, 0.04, 0.69), 11: (0.14, 0.11, 0.05, 0.6), 100: (0.36, 0.2, 0.16, 0.39),
                       1000: (0.66, 0.33, 0.32, 0.09), 20000: (0.74, 0.37, 0.37, 0.0)}, ALPHA_TRACE
assert np.allclose(_wa, _d2.W_ANS, atol=1e-3) and tuple(_aa[_ia].round(2)) == tuple(_d2.ALPHA_UNIT.values())
# 커널 쪽 그림: (x·x')² = φ(x)·φ(x'), φ(x) = (x1², √2 x1x2, x2²)
K_X, K_XP = np.array([1.0, 2.0]), np.array([3.0, 1.0])
phi3 = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
K_PX, K_PXP = phi3(K_X).round(2), phi3(K_XP).round(2)            # (1, 2.83, 4), (9, 4.24, 1)
K_PROD = (phi3(K_X) * phi3(K_XP)).round(6)                       # 9, 12, 4
K_DOT = float(K_X @ K_XP)                                         # 5
assert K_PX.tolist() == [1, 2.83, 4] and K_PXP.tolist() == [9, 4.24, 1] and K_PROD.tolist() == [9, 12, 4]
assert K_DOT == 5 and abs(phi3(K_X) @ phi3(K_XP) - K_DOT ** 2) < 1e-9

# α 3개 → 계수 6개 (AL_A = α_k y_k: α = 0.5 · 0.2 · 0.3, y = +1 · −1 · +1)
AL_PTS = np.array([[1, 2], [3, 1], [0, 1]], float)
AL_A = np.array([0.5, -0.2, 0.3])
_W = sum(a * phi6(p) for a, p in zip(AL_A, AL_PTS))
AL_COEF = (_W * np.array([1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), 1])).round(3)   # 1, x1, x2, x1², x1x2, x2² 앞의 수
AL_X = np.array([1.0, 1.0])
AL_FK = float(sum(a * (1 + AL_X @ p) ** 2 for a, p in zip(AL_A, AL_PTS)))
AL_F6 = float(_W @ phi6(AL_X))
assert AL_COEF.tolist() == [0.6, -0.2, 2.2, -1.3, 0.8, 2.1]
assert round(AL_FK, 6) == round(AL_F6, 6) == 4.2

# 방법 A (특징 6개 직접 → w) vs 방법 B (커널 → α): 특징 늘리기 그림과 같은 원 데이터
Xc, yc = make_circles(120, factor=0.4, noise=0.06, random_state=1); yc = 2 * yc - 1
Fc = np.array([phi6(x) for x in Xc])
_A = SVC(kernel="linear", C=1).fit(Fc, yc)
_B = SVC(kernel="poly", degree=2, coef0=1, gamma=1, C=1).fit(Xc, yc)
_wB = _B.dual_coef_[0] @ Fc[_B.support_]
AB_N = len(Xc)                                # α 후보 수 = 학습 점 수
AB_NSV = len(_B.support_)                     # 0 이 아닌 α
AB_W = _A.coef_[0].round(2)
assert np.allclose(_A.coef_[0], _wB, atol=1e-3) and abs(_A.intercept_[0] - _B.intercept_[0]) < 1e-3
assert (_A.predict(Fc) == _B.predict(Xc)).all()
AB_ACC = _B.score(Xc, yc)
QL_ACC = SVC(kernel="linear", C=1).fit(Xc, yc).score(Xc, yc)   # 질문 그림: 직선 SVM 정확도


import importlib.util as _u                            # 알파 학습 (train_dual) 은 ../9_svm_extra/data.py
_sp = _u.spec_from_file_location("sdata", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "9_svm_extra", "data.py"))
_sd = _u.module_from_spec(_sp); _sp.loader.exec_module(_sd); train_dual = _sd.train_dual

# 같은 학습에서 점곱 자리에 다항 커널 (x·x' + 1)² → 원 데이터 120개, sklearn (하드 마진) 과 비교
_G = (1 + Xc @ Xc.T) ** 2
AK, BK, _ = train_dual(_G, yc.astype(float), T=200000, eta=0.002, etab=0.1)
_ay = AK * yc
AK_M = float(1 / np.sqrt(_ay @ _G @ _ay))
_sk = SVC(kernel="poly", degree=2, coef0=1, gamma=1, C=1e6).fit(Xc, yc)
_ds = _sk.dual_coef_[0]; _Gs = (1 + Xc[_sk.support_] @ Xc[_sk.support_].T) ** 2
AK_M_SK = float(1 / np.sqrt(_ds @ _Gs @ _ds))
AK_NSV, AK_NSV_SK = int((AK > 0.01).sum()), len(_sk.support_)
_g = np.random.RandomState(0).uniform(-1.3, 1.3, (5000, 2))
AK_AGREE = float((np.sign(((1 + _g @ Xc.T) ** 2) @ _ay + BK) == np.sign(_sk.decision_function(_g))).mean())
assert round(AK_M, 2) == round(AK_M_SK, 2) == 0.20 and AK_AGREE > 0.998

if __name__ == "__main__":
    print("cut", round(L1_CUT, 2), "coef", AL_COEF, AL_FK)
    print("A=B w", AB_W, "b", round(_A.intercept_[0], 2), "n", AB_N, "nsv", AB_NSV, "acc", AB_ACC, "linear acc", round(QL_ACC, 2))
    print("poly", round(AK_M, 4), round(AK_M_SK, 4), AK_NSV, AK_NSV_SK, AK_AGREE)
    assert sv_count_sim() == SV_COUNT, sv_count_sim()
    print("data ok")
