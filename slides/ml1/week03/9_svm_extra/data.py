"""SVM 2 (소프트 마진) 덱의 숫자. 선형 SVM 덱(../1_svm/data.py)의 숫자는 그대로 불러 쓰고, 이 덱에서 새로 쓰는 숫자만 계산 · assert 한다."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_svm"))
from sklearn.svm import SVC

# 점곱 꼴로 새 점 하나 분류 (지난 시간 점 20개, 서포트 벡터 3개)
import data as _d2                                   # ../1_svm/data.py
NEW_PT = (0.5, -0.3)
_YK = {s: (1 if s in _d2.RED_PTS else -1) for s in _d2.SV}
_AX = {(0.7, 0.2): 1.1076, (-0.6, -0.4): 0.5538, (0.1, -1.1): 0.5538}                  # sklearn α* (식 (13) 기준, 덱 표시는 1.1 · 0.55)
assert all(abs(_AX[s] - _d2.ALPHA[s]) < 0.01 for s in _AX)
NEW_TERMS = [round(_AX[s] * _YK[s] * float(np.dot(NEW_PT, s)), 2) for s in _d2.SV]      # α_k y_k (x·x_k): 0.32, 0.10, −0.21
NEW_WX = sum(NEW_TERMS)
NEW_F = _d2.B_STAR + sum(_AX[s] * _YK[s] * float(np.dot(NEW_PT, s)) for s in _d2.SV)    # D(x) = 0.26
assert NEW_TERMS == [0.32, 0.1, -0.21] and round(_d2.B_STAR, 2) == 0.05 and round(NEW_F, 2) == 0.26
assert abs(NEW_F - (np.dot(_d2.W_STAR, NEW_PT) + _d2.B_STAR)) < 0.01      # w*·x + b* 와 같다
assert round(NEW_F * _d2.M_ANS, 2) == 0.18                              # 거리 = D × M
# ---- α 학습: 논문 식 (17) J(α, b) = Σ α_k(1 − b y_k) − ½ α·H·α 를 조금씩 키운다 (α ≥ 0).
def train_dual(G, y, T=20000, eta=0.05, etab=0.05, snaps=(), a0=None):
    """한 걸음: 값 = y_k D(x_k), D(x_k) = Σ_l α_l y_l G_lk + b.  α_k ← max(0, α_k + η(1 − 값))  (∂J/∂α_k = 1 − y_k D(x_k)),
    b ← b + η_b Σ α_k y_k  (J 의 b 방향은 안장점: Σ α y = 0 이 되면 멈춘다). J 는 논문 식, 이렇게 조금씩 올라가는 방법은 수업용 선택 (논문은 Powell, sklearn 은 SMO).
    hist[t] = t 걸음 뒤의 (α, b)"""
    a, b, hist = np.zeros(len(y)) if a0 is None else np.array(a0, float), 0.0, {}
    if 0 in snaps:
        hist[0] = (a.copy(), b)
    for t in range(1, T + 1):
        v = y * (G @ (a * y) + b)
        a = np.maximum(0.0, a + eta * (1 - v))
        b = b + etab * (a @ y)
        if t in snaps:
            hist[t] = (a.copy(), float(b))
    return a, float(b), hist


AL_STEPS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
_X20 = np.array(_d2.RED_PTS + _d2.BLUE_PTS, float); _y20 = np.r_[np.ones(10), -np.ones(10)]
A0 = np.where(_y20 > 0, 1 / (_y20 > 0).sum(), 1 / (_y20 < 0).sum())   # 시작: 같은 색 점 수 분의 1 → w = 빨강 평균 − 파랑 평균, Σ α y = 0
AL20, BL20, AL_HIST = train_dual(_X20 @ _X20.T, _y20, snaps=AL_STEPS, a0=A0)
_isv = [_X20.tolist().index(list(s)) for s in _d2.SV]
AL_TABLE = {}                                                    # 걸음: ([α r, b1, b2], α > 0 인 점 수, (w1, w2), b)
for t in (0, 1, 10, 50, 100, 500, 20000):
    al, bb = AL_HIST[t]; w = (al * _y20) @ _X20
    AL_TABLE[t] = ([round(float(al[i]), 2) for i in _isv], int((al > 1e-6).sum()), (round(float(w[0]), 2), round(float(w[1]), 2)), round(bb, 2) + 0.0)
assert [round(float(AL20[i]), 3) for i in _isv] == [1.108, 0.554, 0.554] and int((AL20 > 1e-6).sum()) == 3   # 논문 α* 와 sklearn 과 같다
assert np.allclose((AL20 * _y20) @ _X20, _d2.W_STAR, atol=2e-3) and abs(BL20 - _d2.B_STAR) < 2e-3
_w0 = (A0 * _y20) @ _X20
assert np.allclose(_w0, _X20[_y20 > 0].mean(0) - _X20[_y20 < 0].mean(0)) and np.allclose(_w0 / 2, _d2.MEAN_W, atol=0.006) and abs(A0 @ _y20) < 1e-12   # 0걸음 = 평균 선

# 소프트 마진 (Cortes · Vapnik 1995): 같은 학습에서 α 를 C 에서 자른다 → sklearn SVC(C) 와 같은 선
_Xo = np.r_[_X20, [_d2.OUTLIER]]; _yo = np.r_[_y20, [1]]; _Go = _Xo @ _Xo.T
for _C in (3, 0.5):
    _a, _b = np.zeros(21), 0.0
    for _t in range(20000):
        _a = np.clip(_a + 0.02 * (1 - _yo * (_Go @ (_a * _yo) + _b)), 0, _C); _b += 0.02 * (_a @ _yo)
    _s = SVC(kernel="linear", C=_C, tol=1e-8).fit(_Xo, _yo)
    assert np.allclose((_a * _yo) @ _Xo, _s.coef_[0], atol=2e-3) and abs(_b - _s.intercept_[0]) < 2e-3

if __name__ == "__main__":
    print("alpha table", AL_TABLE, "max alpha", round(max(float(h[0].max()) for h in AL_HIST.values()), 3))
    print("data ok")
