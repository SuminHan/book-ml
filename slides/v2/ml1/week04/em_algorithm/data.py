"""슬라이드·그림·실습이 공유하는 숫자 (쇼츠 lessons/ml1/em_algorithm 과 같은 예시: 학년이 지워진 키 명단 24명).
키는 손으로 만든 예시 (cm, 실제 통계 아님). 1차원 가우시안 혼합 (K=2) 을 EM 으로 학습.
시작: 평균 150 · 155, 표준편차 6 · 6, 비율 0.5 · 0.5. 40번 반복."""
import numpy as np

A = [142, 145, 147, 148, 149, 150, 151, 152, 153, 155, 157, 159]      # 초6 (예시)
B = [158, 161, 163, 164, 165, 166, 167, 168, 169, 171, 173, 176]      # 중3 (예시)
MU0, SD0, PI0 = (150.0, 155.0), (6.0, 6.0), (0.5, 0.5)
ITERS = 40


def pdf(x, m, s):
    return np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))


def e_step(x, mu, sd, pi):
    """학생마다 [초6일 확률, 중3일 확률] (책임값 γ) 과 로그 우도"""
    p = np.array(pi) * pdf(x[:, None], np.array(mu), np.array(sd))
    return p / p.sum(1, keepdims=True), float(np.log(p.sum(1)).sum())


def m_step(x, g):
    """γ 를 무게로 평균 · 표준편차 · 비율 다시"""
    N = g.sum(0); mu = (g * x[:, None]).sum(0) / N
    return mu, np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / N), N / len(x)


def run_em(x, mu, sd, pi, iters):
    P = [(np.array(mu), np.array(sd), np.array(pi))]; G, LL = [], []
    for _ in range(iters):
        g, ll = e_step(x, *P[-1]); G.append(g); LL.append(ll); P.append(m_step(x, g))
    g, ll = e_step(x, *P[-1]); G.append(g); LL.append(ll)
    return P, G, LL


X = np.array(A + B, float)
TRUE = np.r_[np.zeros(12), np.ones(12)]
PDF153 = (0.0587, 0.0629)                  # 153 cm 에서 두 종의 높이 (시작값)
G153 = 52                                  # 153 cm → 중3 52%
MU1 = (152.3, 162.1); SD1 = (7.4, 8.5); PI1 = (0.35, 0.65)
MU_FIN = (149.7, 165.4); SD_FIN = (4.3, 5.8); PI_FIN = (0.43, 0.57)
LL0, LL1, LL_FIN = -103.08, -87.45, -85.78
CORRECT, WRONG = 22, (157, 159)
G157 = 60

if __name__ == "__main__":
    P, G, LL = run_em(X, MU0, SD0, PI0, ITERS)
    assert tuple(round(v, 4) for v in (pdf(153, 150, 6), pdf(153, 155, 6))) == PDF153
    assert round(G[0][A.index(153), 1] * 100) == G153
    r1 = lambda a: tuple(round(float(v), 1) for v in a); r2 = lambda a: tuple(round(float(v), 2) for v in a)
    assert (r1(P[1][0]), r1(P[1][1]), r2(P[1][2])) == (MU1, SD1, PI1)
    assert (r1(P[-1][0]), r1(P[-1][1]), r2(P[-1][2])) == (MU_FIN, SD_FIN, PI_FIN)
    assert (round(LL[0], 2), round(LL[1], 2), round(LL[-1], 2)) == (LL0, LL1, LL_FIN) and np.all(np.diff(LL) > -1e-9)
    pred = G[-1][:, 1] > 0.5
    assert int((pred == TRUE).sum()) == CORRECT and tuple(int(v) for v in X[pred != TRUE]) == WRONG
    assert round(G[-1][A.index(157), 1] * 100) == G157
    print("data ok")
