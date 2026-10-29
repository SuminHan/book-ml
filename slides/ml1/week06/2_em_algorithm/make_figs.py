"""이 덱에만 있는 그림 원본 (푸아송 혼합). 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py
나머지 그림은 원본 덱 slides/v2/ml1/week04/em_algorithm/figs/ 를 그대로 쓴다.
데이터: data/taxi_gramercy_3am.csv (prep_taxi.py). EM 은 count 만 보고, weekend 는 확인에만 쓴다. 슬라이드 숫자는 print 로 확인."""
import os, math, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GRAY = "#2E3192", "#d62728", "#888888"


def poisson(k, lam):
    k = np.asarray(k, dtype=float)
    return np.exp(-lam + k * np.log(lam) - np.array([math.lgamma(v + 1) for v in np.atleast_1d(k)]))


def em_poisson(x, lam_quiet, lam_busy, pi_busy, steps=200):
    """푸아송 두 개 섞기: E = 칸마다 '붐비는 무리' 확률 g, M = g 를 무게로 평균 (λ) · 비율 (π)."""
    for _ in range(steps):
        a = pi_busy * poisson(x, lam_busy); b = (1 - pi_busy) * poisson(x, lam_quiet); g = a / (a + b)
        lam_busy = (g * x).sum() / g.sum(); lam_quiet = ((1 - g) * x).sum() / (1 - g).sum(); pi_busy = g.mean()
    return lam_quiet, lam_busy, pi_busy, g


d = pd.read_csv("data/taxi_gramercy_3am.csv")
x, we = d["count"].values, d["weekend"].values.astype(bool)
lq, lb, pb, g = em_poisson(x, 1.0, 5.0, 0.5)
ks = np.arange(0, x.max() + 1)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.bar(ks, np.bincount(x, minlength=len(ks)), color="#dddddd", label="10분 칸 수 (실제)")
ax.plot(ks, len(x) * (1 - pb) * poisson(ks, lq), "o-", color=BLUE, lw=2.5, label=f"조용한 무리  λ = {lq:.2f}")
ax.plot(ks, len(x) * pb * poisson(ks, lb), "o-", color=RED, lw=2.5, label=f"붐비는 무리  λ = {lb:.1f}")
ax.set_xlabel("새벽 3시대 10분 동안 태운 손님 수"); ax.set_ylabel("칸 수"); ax.legend(frameon=False, fontsize=13)
fig.tight_layout(); fig.savefig("figs/em_poisson.png"); plt.close(fig)
a = pb * poisson(3, lb)[0]; b = (1 - pb) * poisson(3, lq)[0]
print(len(x), "windows; EM lam quiet", round(lq, 2), "busy", round(lb, 2), "pi busy", round(pb, 2))
print("true weekday mean", round(x[~we].mean(), 2), "weekend mean", round(x[we].mean(), 2), "weekend share", round(we.mean(), 2), f"({we.sum()}/{len(x)})")
print("agree", round(((g > 0.5) == we).mean(), 3), int(((g > 0.5) == we).sum()), "/", len(x))
print("x=3: P(quiet)", round(poisson(3, lq)[0], 4), "P(busy)", round(poisson(3, lb)[0], 4), "gamma busy", round(a / (a + b), 2))
