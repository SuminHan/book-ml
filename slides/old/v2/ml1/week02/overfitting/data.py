"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/overfitting 과 같은 제동거리 예시 데이터).
진짜 규칙 = 물리 법칙 d = v²/(2μg), 마른 아스팔트 μ = 0.7 가정 → 속도 2배면 제동거리 4배. 측정 잡음을 더한 예시 값.
모델 = 다항식 회귀(직선 1차 · 2차 곡선 · 구불구불 9차). 오차 = 평균 절대 오차 MAE (m). 교차 검증 = 5겹, 조각 k = k번째 점과 k+5번째 점."""
import numpy as np

SPEED = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]          # 측정한 속도 (km/h)
DIST = [2, 7, 0, 14, 14, 18, 25, 33, 45, 59]               # 측정한 제동거리 (m)
SPEED_NEW = [15, 35, 45, 65, 85, 95]                       # 새로 잰 속도
DIST_NEW = [3, 9, 6, 19, 45, 54]                           # 새로 잰 제동거리
MU, G = 0.7, 9.8
DEGS = {"직선": 1, "2차 곡선": 2, "구불구불 곡선": 9}
K = 5


def truth(v):
    return (np.asarray(v, float) / 3.6) ** 2 / (2 * MU * G)


def fit(x, y, d):
    """d차 다항식 최소제곱. 속도를 (v - 55) / 27 로 줄여 계산이 안정적이게 한다 (결과 곡선은 같다)."""
    V = np.vander((np.asarray(x, float) - 55) / 27, d + 1)
    c, *_ = np.linalg.lstsq(V, np.asarray(y, float), rcond=None)
    return lambda t: np.vander((np.atleast_1d(np.asarray(t, float)) - 55) / 27, d + 1) @ c


def mae(m, x, y):
    return float(np.abs(m(x) - np.asarray(y, float)).mean())


def folds(n=10, k=K):
    return [np.arange(n)[i::k] for i in range(k)]


def cv(x, y, d, k=K):
    x, y = np.asarray(x, float), np.asarray(y, float); es = []
    for te in folds(len(x), k):
        tr = np.setdiff1d(np.arange(len(x)), te)
        es.append(mae(fit(x[tr], y[tr], d), x[te], y[te]))
    return float(np.mean(es))


# 슬라이드에 적는 결과 (아래 assert 로 확인)
ERR_TR = {"직선": 5.5, "2차 곡선": 2.3, "구불구불 곡선": 0.0}
ERR_TE = {"직선": 6.4, "2차 곡선": 3.1, "구불구불 곡선": 9.1}
ERR_CV = {"직선": 6.8, "2차 곡선": 3.3, "구불구불 곡선": 121.5}
LINE_AT_10 = -5.1                                            # 직선이 10 km/h 에서 예측한 제동거리
WIGGLE_AT_15 = 32.9                                          # 9차 곡선이 새 데이터 15 km/h(실제 3 m) 에서 예측한 값
SWEEP = {1: (5.5, 6.4, 6.8), 2: (2.3, 3.1, 3.3), 3: (1.6, 2.9, 2.5), 4: (1.7, 3.2, 4.6), 5: (1.7, 3.2, 10.5),
         6: (1.8, 3.4, 28.6), 7: (1.4, 5.0, 60.3), 8: (0.5, 7.3, 49.5), 9: (0.0, 9.1, 121.5)}   # 차수: (측정, 새 데이터, 교차검증) MAE

if __name__ == "__main__":
    M = {k: fit(SPEED, DIST, d) for k, d in DEGS.items()}
    assert {k: round(mae(m, SPEED, DIST), 1) for k, m in M.items()} == ERR_TR
    assert {k: round(mae(m, SPEED_NEW, DIST_NEW), 1) for k, m in M.items()} == ERR_TE
    assert {k: round(cv(SPEED, DIST, d), 1) for k, d in DEGS.items()} == ERR_CV
    assert round(float(M["직선"](10)[0]), 1) == LINE_AT_10 and round(float(M["구불구불 곡선"](15)[0]), 1) == WIGGLE_AT_15
    for d, (a, b, c) in SWEEP.items():
        m = fit(SPEED, DIST, d)
        assert (round(mae(m, SPEED, DIST), 1), round(mae(m, SPEED_NEW, DIST_NEW), 1), round(cv(SPEED, DIST, d), 1)) == (a, b, c), d
    assert round(float(truth(100) / truth(50)), 6) == 4 and (round(float(truth(50))), round(float(truth(100)))) == (14, 56)
    print("data ok")
