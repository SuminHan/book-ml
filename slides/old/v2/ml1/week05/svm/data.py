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
    print("data ok")
