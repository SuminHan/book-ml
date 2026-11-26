"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/gaussian 과 같은 골턴 보드 · 시험 예시).
골턴 보드: 못 10줄, 공 1,000개 → 칸별 기대 개수 1000·C(10,k)/2^10 을 반올림 (합 1,000).
시험: 평균 70, 표준편차 10. 키: 예시 (남 평균 174 · 표준편차 6, 여 평균 161 · 표준편차 5)."""
import numpy as np
from math import comb, erf, sqrt, pi, exp

ROWS, BALLS = 10, 1000
BINS = [1, 10, 44, 117, 205, 246, 205, 117, 44, 10, 1]
EXAM_MU, EXAM_SD, EXAM_CUT = 70, 10, 90
SAMPLE = [168, 172, 174, 176, 180]                              # 키 5명 (cm): 평균 174, 표준편차 4
MALE, FEMALE = (174, 6), (161, 5)                                # (평균, 표준편차) cm, 예시
X_Q = 168                                                        # 이 키는 남? 여?
POST_MALE_168 = 57                                               # % (사전확률 반반)
BOUNDARY = 167.3                                                 # 두 종의 높이가 같은 곳 (사전확률 반반)
PRIOR_MALE, X_Q2, POST_MALE_165 = 0.8, 165, 60                   # 남 80% 인 교실에서 165 cm → 남일 확률 60%
POST_MALE_165_EVEN = 27                                          # 같은 165 cm, 반반 교실이면 27%


def pdf(x, mu, sd):
    return exp(-(x - mu) ** 2 / (2 * sd ** 2)) / (sqrt(2 * pi) * sd)


def cdf(x, mu, sd):
    return 0.5 * (1 + erf((x - mu) / (sd * sqrt(2))))


def posterior(x, prior=0.5):
    a, b = pdf(x, *MALE) * prior, pdf(x, *FEMALE) * (1 - prior)
    return a / (a + b)


if __name__ == "__main__":
    assert BINS == [round(BALLS * comb(ROWS, k) / 2 ** ROWS) for k in range(ROWS + 1)] and sum(BINS) == BALLS
    assert round((1 - cdf(EXAM_CUT, EXAM_MU, EXAM_SD)) * 100, 1) == 2.3
    assert round((cdf(1, 0, 1) - cdf(-1, 0, 1)) * 100) == 68 and round((cdf(2, 0, 1) - cdf(-2, 0, 1)) * 100) == 95
    m = np.mean(SAMPLE); s = np.sqrt(np.mean((np.array(SAMPLE) - m) ** 2)); assert (m, s) == (174, 4)
    assert round(posterior(X_Q) * 100) == POST_MALE_168
    xs = np.linspace(150, 185, 35001); d = [pdf(x, *MALE) - pdf(x, *FEMALE) for x in xs]
    assert round(float(xs[np.where(np.diff(np.sign(d)))[0][0]]), 1) == BOUNDARY
    assert round(posterior(X_Q2, PRIOR_MALE) * 100) == POST_MALE_165 and round(posterior(X_Q2) * 100) == POST_MALE_165_EVEN
    print("data ok", round(pdf(X_Q, *MALE), 4), round(pdf(X_Q, *FEMALE), 4), round(pdf(X_Q2, *MALE), 4), round(pdf(X_Q2, *FEMALE), 4))
