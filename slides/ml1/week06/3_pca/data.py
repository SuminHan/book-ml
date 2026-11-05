"""슬라이드·그림·실습이 공유하는 숫자.
학생 20명 (수학, 물리) 점수는 설명용 예시 (쇼츠 lessons/ml1/pca 와 같은 값). 분산 비율은 실제 계산.
붓꽃 (sklearn iris, 150송이 · 4개 측정값): 표준화 없이 92.5 · 5.3 · 1.7 · 0.5 %, 표준화하면 73.0 · 22.9 · 3.7 · 0.5 %.
손글씨 숫자 (sklearn digits, 1,797장 · 8×8 = 64칸): 90% 를 남기려면 성분 21개, 2개면 28.5%.
고유얼굴: Turk & Pentland (1991) 'Eigenfaces for Recognition'."""
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCORES = np.array([[95, 92], [35, 47], [73, 76], [61, 59], [62, 62], [65, 68], [42, 55], [65, 64], [57, 57], [99, 99],
                   [71, 66], [63, 62], [64, 70], [59, 63], [54, 56], [63, 67], [74, 58], [65, 71], [80, 74], [65, 56]], float)
MU = SCORES.mean(0)
S = np.cov((SCORES - MU).T)                              # 공분산 행렬 (n − 1 로 나눔)
EV, EVEC = np.linalg.eigh(S)
L1, L2 = EV[1], EV[0]
PC1 = EVEC[:, 1] * np.sign(EVEC[0, 1]); PC2 = EVEC[:, 0] * np.sign(EVEC[0, 0])
ANG1 = np.degrees(np.arctan2(PC1[1], PC1[0]))


def frac(deg, S=S):
    """deg 도 방향으로 그림자를 냈을 때 남는 퍼짐 (분산) 비율."""
    u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))]); return u @ S @ u / np.trace(S)


IRIS = np.round(PCA().fit(load_iris().data).explained_variance_ratio_ * 100, 1)
IRIS_STD = np.round(PCA().fit(StandardScaler().fit_transform(load_iris().data)).explained_variance_ratio_ * 100, 1)
DIG = load_digits().data
DIG_CUM = np.cumsum(PCA().fit(DIG).explained_variance_ratio_)
DIG_K90 = int(np.searchsorted(DIG_CUM, 0.9) + 1)

if __name__ == "__main__":
    assert [round(v, 1) for v in MU] == [65.6, 66.1]
    assert np.round(S, 1).tolist() == [[217.8, 162.8], [162.8, 152.4]]
    assert round(L1, 1) == 351.2 and round(L2, 1) == 19.0 and round(L1 + L2, 1) == round(np.trace(S), 1) == 370.2
    tr, det = np.trace(S), np.linalg.det(S)                     # 손 계산: λ² − (tr) λ + det = 0
    assert np.allclose(sorted(np.roots([1, -tr, det])), [L2, L1])
    assert round(frac(0) * 100) == 59 and round(L1 / (L1 + L2) * 100) == 95 and round(ANG1) == 39
    assert max(frac(t) for t in np.linspace(0, 180, 1801)) <= L1 / (L1 + L2) + 1e-9
    assert IRIS.tolist() == [92.5, 5.3, 1.7, 0.5] and IRIS_STD.tolist() == [73.0, 22.9, 3.7, 0.5]
    assert DIG_K90 == 21 and round(DIG_CUM[1] * 100, 1) == 28.5
    print("data ok", np.round(PC1, 2), np.round(PC2, 2), round(det, 1))
