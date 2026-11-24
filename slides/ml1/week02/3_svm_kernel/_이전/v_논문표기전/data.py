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
    """다항 2차 커널 (1 + x·x')² 의 특징 6개"""
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

# 점곱 꼴로 새 점 하나 분류 (지난 시간 점 20개, 서포트 벡터 3개)
import data as _d2                                   # ../2_svm/data.py
NEW_PT = (0.5, -0.3)
NEW_TERMS = [round(a * float(np.dot(NEW_PT, s)), 2) for s, a in _d2.ALPHA.items()]      # 0.21, 0.07, −0.14
NEW_F = _d2.B_ANS + sum(a * float(np.dot(NEW_PT, s)) for s, a in _d2.ALPHA.items())
assert NEW_TERMS == [0.21, 0.07, -0.14] and round(_d2.B_ANS, 2) == 0.04 and round(NEW_F, 2) == 0.18
assert abs(NEW_F - (np.dot(_d2.W_ANS, NEW_PT) + _d2.B_ANS)) < 0.01      # w·x + b 와 같다
# 커널 쪽 그림: (x·x')² = φ(x)·φ(x'), φ(x) = (x1², √2 x1x2, x2²)
K_X, K_XP = np.array([1.0, 2.0]), np.array([3.0, 1.0])
phi3 = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
K_PX, K_PXP = phi3(K_X).round(2), phi3(K_XP).round(2)            # (1, 2.83, 4), (9, 4.24, 1)
K_PROD = (phi3(K_X) * phi3(K_XP)).round(6)                       # 9, 12, 4
K_DOT = float(K_X @ K_XP)                                         # 5
assert K_PX.tolist() == [1, 2.83, 4] and K_PXP.tolist() == [9, 4.24, 1] and K_PROD.tolist() == [9, 12, 4]
assert K_DOT == 5 and abs(phi3(K_X) @ phi3(K_XP) - K_DOT ** 2) < 1e-9

# α 3개 → 계수 6개 (α 는 덱 약속대로 부호 포함)
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


# ---- α 학습: 1차시 학습 한 걸음을 w 대신 α 로 (G = 점끼리 점곱 또는 K 표)
def train_alpha(G, y, T=20000, lr0=0.05, snaps=()):
    """매 걸음: 값 = y_i(Σ_j α_j G_ji + b), 가장 작은 점(동점 모두) α_k += η y_k, b += η y_k, α 를 ‖w′‖ = √(αᵀGα) 로 나눈다"""
    a, b, hist = np.zeros(len(y)), 0.0, {}
    for t in range(1, T + 1):
        v = y * (G @ a + b); k = v <= v.min() + 1e-9; eta = lr0 / np.sqrt(t)
        a = a + eta * k * y; b += eta * y[k].sum()
        n = np.sqrt(a @ G @ a)
        if n > 0:
            a = a / n
        if t in snaps:
            hist[t] = a.copy()
    return a, b, hist


AL_STEPS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
_X20 = np.array(_d2.RED_PTS + _d2.BLUE_PTS, float); _y20 = np.r_[np.ones(10), -np.ones(10)]
AL20, BL20, AL_HIST = train_alpha(_X20 @ _X20.T, _y20, snaps=AL_STEPS)
AL_ROWS = {t: [round(float(AL_HIST[t][_X20.tolist().index(list(s))]), 2) for s in _d2.SV] for t in (1, 10, 100, 1000, 20000)}
AL_OTHER = {t: round(float(np.sort(np.abs(AL_HIST[t]))[-4]), 3) for t in (1, 10, 100, 1000, 20000)}   # 서포트 벡터 3개 빼고 가장 큰 |α|
assert AL_ROWS[20000] == [0.74, -0.37, -0.37] and AL_OTHER[20000] == 0.0 and AL_ROWS[1] == [0.04, -0.04, -0.04]
assert np.allclose(AL20 @ _X20, _d2.W_ANS, atol=1e-3) and abs(BL20 - _d2.B_ANS) < 2e-3

# 같은 학습에서 점곱 자리에 다항 커널 (1 + x·x')² → 원 데이터 120개, sklearn (하드 마진) 과 비교
_G = (1 + Xc @ Xc.T) ** 2
AK, BK, _ = train_alpha(_G, yc.astype(float))
AK_M = float((yc * (_G @ AK + BK)).min())
_sk = SVC(kernel="poly", degree=2, coef0=1, gamma=1, C=1e6).fit(Xc, yc)
_ds = _sk.dual_coef_[0]; _Gs = (1 + Xc[_sk.support_] @ Xc[_sk.support_].T) ** 2
AK_M_SK = float(1 / np.sqrt(_ds @ _Gs @ _ds))
AK_NSV, AK_NSV_SK = int((np.abs(AK) > 0.01).sum()), len(_sk.support_)
_g = np.random.RandomState(0).uniform(-1.3, 1.3, (5000, 2))
AK_AGREE = float((np.sign(((1 + _g @ Xc.T) ** 2) @ AK + BK) == np.sign(_sk.decision_function(_g))).mean())
assert round(AK_M, 2) == round(AK_M_SK, 2) == 0.20 and AK_NSV == AK_NSV_SK == 7 and AK_AGREE > 0.999

if __name__ == "__main__":
    print("cut", round(L1_CUT, 2), "coef", AL_COEF, AL_FK)
    print("A=B w", AB_W, "b", round(_A.intercept_[0], 2), "n", AB_N, "nsv", AB_NSV, "acc", AB_ACC, "linear acc", round(QL_ACC, 2))
    print("alpha rows", AL_ROWS, AL_OTHER)
    print("poly", round(AK_M, 4), round(AK_M_SK, 4), AK_NSV, AK_NSV_SK, AK_AGREE)
    print("data ok")
