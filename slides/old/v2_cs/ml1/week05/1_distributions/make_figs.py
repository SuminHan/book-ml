"""이 덱(v2_cs)에만 있는 스튜던트 t · 푸아송 분포 그림. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/ga_t_*.png, ga_poisson*.png).
나머지 그림은 원본 덱 slides/v2/ml1/week03/gaussian/make_figs.py 가 만든다."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#bbbbbb", "#2a9d8f"
O = "figs/"
HEIGHTS = np.array([168, 172, 174, 176, 180])  # 데이터에서 μ, σ 구하기 쪽의 키 5명


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 표본 3개로 표준화: 진짜 σ 로 나누기 vs 표본 s 로 나누기
def sim_small_sample(m, reps, seed):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 1, (reps, m))
    se_true = 1 / np.sqrt(m)
    se_hat = s.std(1, ddof=1) / np.sqrt(m)
    return s.mean(1) / se_true, s.mean(1) / se_hat


z, t = sim_small_sample(3, 20000, 0)
x = np.linspace(-6, 6, 400)
bins = np.linspace(-6, 6, 61)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
for ax, v, title, p in ((axs[0], z, "진짜 σ 로 나누면", 2 * stats.norm.sf(3)),
                        (axs[1], t, "표본에서 잰 s 로 나누면", 2 * stats.t.sf(3, 2))):
    ax.hist(v, bins=bins, weights=np.full(len(v), 1 / (len(v) * (bins[1] - bins[0]))), color=BLUE, alpha=0.35)
    ax.plot(x, stats.norm.pdf(x), color=BLUE, lw=2.5, label="정규분포")
    ax.axvspan(3, 6, color=RED, alpha=0.08); ax.axvspan(-6, -3, color=RED, alpha=0.08)
    ax.text(4.5, 0.25, f"±3 밖\n{p * 100:.2g}%", ha="center", color=RED, fontsize=15, fontweight="bold")
    ax.set_title(f"표본 3개 · {title}", fontsize=15); ax.set_yticks([]); ax.set_xticks([-6, -3, 0, 3, 6])
axs[1].plot(x, stats.t.pdf(x, 2), color=RED, lw=2.5, label="t 분포 (자유도 2)")
axs[1].legend(frameon=False, loc="upper left", fontsize=12); axs[0].legend(frameon=False, loc="upper left", fontsize=12)
save(fig, "ga_t_sim.png")

# 2. t 분포 모양: 자유도가 작을수록 꼬리가 두껍다
x = np.linspace(-5, 5, 500)
curves = ((stats.norm.pdf, "정규분포", BLUE), (lambda u: stats.t.pdf(u, 9), "t (표본 10개, 자유도 9)", GOLD),
          (lambda u: stats.t.pdf(u, 2), "t (표본 3개, 자유도 2)", RED))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
for f, lab, c in curves:
    axs[0].plot(x, f(x), color=c, lw=3, label=lab)
    m = x >= 2.5; axs[1].plot(x[m], f(x[m]), color=c, lw=3)
axs[0].set_title("가운데는 낮게", fontsize=15); axs[0].legend(frameon=False, fontsize=12, loc="upper left")
axs[1].set_title("꼬리는 두껍게 (x ≥ 2.5 확대)", fontsize=15)
axs[0].set_yticks([]); axs[1].set_yticks([]); axs[1].set_xticks([3, 4, 5])
save(fig, "ga_t_shape.png")

# 3. 95% 구간 폭: 표본 수가 늘면 1.96 으로
ms = np.arange(3, 31)
crit = stats.t.ppf(0.975, ms - 1)
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(ms, crit, "o-", color=RED, lw=2.5, ms=6, label="t 분포")
ax.axhline(stats.norm.ppf(0.975), color=BLUE, lw=2.5, ls="--", label="정규분포 1.96")
for mm in (3, 5, 10):
    v = stats.t.ppf(0.975, mm - 1)
    ax.annotate(f"{mm}개: {v:.2f}", (mm, v), (mm + 1.2, v + 0.25), fontsize=15, color=RED, fontweight="bold")
ax.set_xlabel("표본 수 m"); ax.set_ylabel("95% 구간 = 표준오차 × ?"); ax.set_ylim(1.5, 4.9)
ax.legend(frameon=False, loc="upper right")
save(fig, "ga_t_crit.png")

# 4. 푸아송 분포: 1분에 평균 3통 오는 전화, 1000분 동안 k통 온 분(minute)의 수
def poisson_minutes(lam, kmax, total):
    k = np.arange(kmax + 1)
    return k, np.round(total * stats.poisson.pmf(k, lam)).astype(int)


k, cnt = poisson_minutes(3, 10, 1000)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.bar(k, cnt, color=[RED if v in (0, 3) else BLUE for v in k], alpha=0.85, width=0.75)
for v, c in zip(k, cnt):
    ax.text(v, c + 4, str(c), ha="center", va="bottom", fontsize=14, fontweight="bold")
ax.set_xticks(k); ax.set_xlabel("1분 동안 온 전화 k 통"); ax.set_ylabel("1000분 중 몇 분")
ax.set_ylim(0, 265); ax.set_title("λ = 3 (1분에 평균 3통)", fontsize=15)
save(fig, "ga_poisson.png")

# 5. λ 가 커지면 종 모양: 푸아송 막대 위에 N(λ, λ) 곡선
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, lam in zip(axs, (1, 4, 15)):
    kk = np.arange(0, int(lam + 4.5 * np.sqrt(lam)) + 3)
    ax.bar(kk, stats.poisson.pmf(kk, lam), color=BLUE, alpha=0.6, width=0.8, label="푸아송")
    xx = np.linspace(kk[0] - 0.5, kk[-1] + 0.5, 300)
    ax.plot(xx, stats.norm.pdf(xx, lam, np.sqrt(lam)), color=RED, lw=2.5, label="N(λ, λ)")
    ax.set_title(f"λ = {lam}", fontsize=15); ax.set_yticks([]); ax.set_xlabel("k")
axs[0].legend(frameon=False, fontsize=12, loc="upper right")
save(fig, "ga_poisson_lam.png")

# 확인용 숫자 (슬라이드에 적은 값)
assert list(cnt[:4]) == [50, 149, 224, 224] and abs(np.exp(-3) - 0.050) < 1e-3
assert abs(stats.poisson.pmf(0, 2) - 0.135) < 1e-3 and abs(stats.poisson.pmf(2, 2) - 0.271) < 1e-3  # 문제풀이 6번
print("poisson λ=3 minutes:", cnt.tolist(), "sum", cnt.sum())
s = HEIGHTS.std(ddof=1); se = s / np.sqrt(len(HEIGHTS))
assert abs(2 * stats.norm.sf(3) - 0.0027) < 1e-4 and abs(2 * stats.t.sf(3, 2) - 0.095) < 1e-3
assert abs(s ** 2 - 20) < 1e-9 and abs(se - 2.0) < 1e-9
print(f"t(4)={stats.t.ppf(0.975, 4):.2f}  t-interval ±{stats.t.ppf(0.975, 4) * se:.1f}  normal ±{1.96 * se:.1f}")
print("ok")
