"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#999999", "#2a9d8f"
O = "figs/"


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 당첨까지 뽑은 횟수 (실제 4%)
rng = np.random.default_rng(0); d = rng.geometric(0.04, 100_000)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.hist(d, bins=np.arange(1, 151, 3), color=BLUE, alpha=0.8)
for v, c, t in [(5, GOLD, "공지대로면 평균 5번"), (J["draw_mean"], RED, f"실제 평균 {J['draw_mean']:.0f}번")]:
    ax.axvline(v, color=c, lw=2.5, ls="--"); ax.text(v + 2, ax.get_ylim()[1] * 0.85, t, color=c, fontsize=14)
ax.set_xlabel("draws until the first win"); ax.set_ylabel("players (of 100,000)")
save(fig, "kl_gacha.png")

# 2. 놀라움 곡선
fig, ax = plt.subplots(figsize=(6.6, 4.0))
p = np.linspace(0.01, 1, 300); ax.plot(p, -np.log2(p), color=GRAY, lw=2)
for v, c in [(0.04, RED), (0.2, GOLD), (0.8, GOLD), (0.96, RED)]:
    ax.plot(v, -log2(v), "o", color=c, ms=9); ax.annotate(f"{v:.0%} → {-log2(v):.2f}", (v, -log2(v)), (8, 4), textcoords="offset points", color=c, fontsize=13)
ax.set_xlabel("probability p"); ax.set_ylabel("surprise  $-\\log_2 p$  (bits)")
save(fig, "kl_surprise.png")

# 3. H + KL = CE 막대
fig, ax = plt.subplots(figsize=(6.2, 2.4))
ax.barh([0], [H_P], color=BLUE); ax.barh([0], [KL_PQ], left=[H_P], color=RED)
ax.text(H_P / 2, 0, f"H(P) = {H_P:.2f}", ha="center", va="center", color="white", fontsize=14)
ax.text(H_P + KL_PQ / 2, 0, f"KL = {KL_PQ:.2f}", ha="center", va="center", color="white", fontsize=14)
ax.annotate("", (CE_PQ, 0.55), (0, 0.55), arrowprops=dict(arrowstyle="<->", color=GOLD, lw=2)); ax.text(CE_PQ / 2, 0.68, f"H(P, Q) = {CE_PQ:.2f} bits", ha="center", color="#7a6a40", fontsize=14)
ax.set_ylim(-0.6, 0.95); ax.set_xlim(0, 0.45); ax.set_yticks([]); ax.set_xlabel("bits per draw")
save(fig, "kl_bar.png")

# 4. 앞 방향 vs 뒤 방향
x = np.linspace(-6, 6, 1201)
npdf = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
Pd = 0.5 * npdf(-2, 0.5) + 0.5 * npdf(2, 0.5)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, key, t, c in [(axs[0], "fwd", "KL(P‖Q) 최소: 둘 다 덮는다", BLUE), (axs[1], "rev", "KL(Q‖P) 최소: 하나만 고른다", RED)]:
    m, s = J[key][1]; ax.fill_between(x, Pd, color=GRAY, alpha=0.35, label="P (two peaks)"); ax.plot(x, npdf(m, s), color=c, lw=2.5, label=f"Q = N({m:.0f}, {s:.2f}²)")
    ax.set_title(t, fontsize=15, color=c); ax.legend(frameon=False, fontsize=11); ax.set_xlabel("x")
save(fig, "kl_modes.png")
print("figs ok")
