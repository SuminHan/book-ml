"""로지스틱회귀 차시 공유 숫자 (슬라이드 · 그림 · 실습 일치). 모두 설명용 예시 데이터."""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

PASS_HOURS = [1.7, 5.2, 6.2, 0.8, 1.9, 9.3, 1.2, 1.7, 9.5, 6.4, 4.0, 5.4, 6.8, 3.1, 1.8, 8.0, 6.9, 5.4, 8.3, 5.7, 9.8, 2.4, 5.8, 5.1, 3.9, 6.1, 2.7, 8.1, 8.7, 1.7, 4.9, 3.1, 1.3, 9.0, 4.6, 1.9, 6.9, 2.4, 9.1, 2.6]
PASSED = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]

def study_pass():
    """학생 40명: 공부 시간(시간) → 합격(1) / 불합격(0) (실습 노트북에 그대로 적힌 값)"""
    return np.array(PASS_HOURS), np.array(PASSED)

def study_sleep():
    """학생 60명: (공부 시간, 수면 시간) → 합격"""
    r = np.random.default_rng(5); h = r.uniform(0.5, 10, 60); s = r.uniform(4, 9, 60)
    y = (r.uniform(size=60) < sigmoid(-11 + 1.0 * h + 1.0 * s)).astype(int)
    return h, s, y

def fit(X, y, alpha=0.1, steps=20000):
    """경사하강법 (교차 엔트로피). X 에 1 열 포함"""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= alpha * X.T @ (sigmoid(X @ w) - y) / len(y)
    return w

# 슬라이드에 적는 결과 (아래 확인)
W0, W1, BOUNDARY, P7, ODDS_X = -4.24, 0.96, 4.42, 0.92, 2.61

if __name__ == "__main__":
    x, y = study_pass(); w = fit(np.c_[np.ones_like(x), x], y)
    print("1D w", w.round(2), "경계 x =", round(-w[0] / w[1], 2), "7시간 p =", round(sigmoid(w[0] + 7 * w[1]), 2),
          "odds 배수 e^w1 =", round(np.exp(w[1]), 2), "합격", y.sum(), "/", len(y))
    h, s, y2 = study_sleep(); w2 = fit(np.c_[np.ones_like(h), h, s], y2)
    print("2D w", w2.round(2), "합격", y2.sum())
    assert np.allclose(w, [W0, W1], atol=0.01) and round(-w[0] / w[1], 2) == BOUNDARY
    assert round(sigmoid(w[0] + 7 * w[1]), 2) == P7 and round(np.exp(w[1]), 2) == ODDS_X
    print("data ok")
