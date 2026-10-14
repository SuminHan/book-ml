"""수업·슬라이드·영상·실습이 공유하는 데이터 (숫자가 서로 일치하도록)"""
import numpy as np

STUDY_HOURS = [6.4, 9.0, 7.9, 2.6, 3.4, 8.8, 0.6, 8.3, 8.1, 4.9, 3.4, 3.1, 2.9, 4.7, 5.3, 5.8, 10.0, 8.0, 6.4, 9.9, 2.5, 2.0, 6.3, 0.9, 0.8, 5.4, 4.9, 9.2, 6.5, 5.4]
STUDY_SCORES = [65, 83, 74, 49, 62, 80, 43, 86, 77, 64, 57, 56, 48, 64, 73, 61, 94, 81, 69, 99, 57, 44, 72, 47, 43, 70, 64, 89, 80, 64]

def study():      # 공부시간(h) → 점수 (실습 노트북에 그대로 적힌 30명)
    return np.array(STUDY_HOURS), np.array(STUDY_SCORES, dtype=float)

def braking():    # 속도(km/h) → 제동거리(m): 곡선
    r = np.random.default_rng(3); v = r.uniform(20, 120, 40)
    return v, 0.0065 * v ** 2 + 0.18 * v + r.normal(0, 4, 40)

def adspend():    # 광고비(만원) → 방문자(명): 점점 덜 늘어남
    r = np.random.default_rng(5); x = r.uniform(5, 300, 40)
    return x, 120 * np.log(x) - 80 + r.normal(0, 25, 40)

# 슬라이드에 적는 결과 (아래 assert 로 위 데이터와 맞는지 확인)
W0, W1, PRED7, RMSE, R2 = 38.58, 5.25, 75.3, 4.36, 0.916
BRAKE_RMSE = {1: 6.7, 2: 4.0}
AD_R2 = {"x": 0.78, "log": 0.96, "sqrt": 0.90}
DEGREES = [1, 2, 4, 8, 12, 15]
OVERFIT = {1: (6.0, 7.4), 2: (3.5, 4.5), 4: (2.7, 7.0), 8: (2.1, 26.5), 12: (1.8, 187.3), 15: (1.7, 1428.9)}  # (학습, 시험) RMSE

if __name__ == "__main__":
    x, y = study(); A = np.c_[np.ones_like(x), x]; w = np.linalg.lstsq(A, y, rcond=None)[0]; p = A @ w
    assert np.allclose(w, [W0, W1], atol=0.01) and round(w[0] + 7 * w[1], 1) == PRED7
    assert round(np.sqrt(np.mean((p - y) ** 2)), 2) == RMSE and round(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), 3) == R2
    v, d = braking()
    for k, r in BRAKE_RMSE.items():
        P = np.vander(v, k + 1); c = np.linalg.lstsq(P, d, rcond=None)[0]; assert round(np.sqrt(np.mean((P @ c - d) ** 2)), 1) == r
    print("data ok")
