"""KL 발산 실험 → assets/kl.json
1) 뽑기 흉내: 실제 당첨 4% 로 10만 명이 당첨될 때까지 뽑기 (평균 · 중앙값 · 50번 넘게 뽑은 비율). 공지 20% 와 비교.
2) 봉우리 둘인 분포 P (−2 · +2 에 0.5 씩, 표준편차 0.5) 를 정규분포 Q 하나로 맞추기:
   KL(P‖Q) 를 줄이면 (앞 방향) vs KL(Q‖P) 를 줄이면 (뒤 방향). 격자 −8 ~ 8, 평균 · 표준편차 격자 탐색."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
draws = rng.geometric(0.04, 100_000); draws_q = rng.geometric(0.20, 100_000)
x = np.linspace(-8, 8, 4001); dx = x[1] - x[0]
npdf = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
P = 0.5 * npdf(-2, 0.5) + 0.5 * npdf(2, 0.5)
def kl(a, b):
    a = a + 1e-300; b = b + 1e-300; return float(np.sum(a * np.log(a / b)) * dx)
best = {"fwd": (9e9, None), "rev": (9e9, None)}
for m in np.linspace(-3, 3, 121):
    for s in np.linspace(0.2, 3.5, 166):
        Q = npdf(m, s); f, r = kl(P, Q), kl(Q, P)
        if f < best["fwd"][0]: best["fwd"] = (f, (float(m), float(s)))
        if r < best["rev"][0]: best["rev"] = (r, (float(m), float(s)))
out = dict(draw_mean=float(draws.mean()), draw_median=float(np.median(draws)), over50=float((draws > 50).mean()), over100=float((draws > 100).mean()),
           draw_q_mean=float(draws_q.mean()), fwd=best["fwd"], rev=best["rev"])
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "assets", "kl.json"), "w"), indent=1)
