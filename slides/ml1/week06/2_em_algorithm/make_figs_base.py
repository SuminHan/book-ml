"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY = "#2E3192", "#d62728", "#A9976B", "#999999"
CM = LinearSegmentedColormap.from_list("br", [BLUE, "#bbbbbb", RED])
O = "figs/"
P, G, LL = run_em(X, MU0, SD0, PI0, ITERS)
xs = np.linspace(135, 185, 400)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


def strip(ax, g=None, y=-0.004):
    c = [GRAY] * len(X) if g is None else [CM(v) for v in g[:, 1]]
    ax.scatter(X, np.full(len(X), y), s=90, c=c, edgecolor="black", lw=0.6, zorder=4, clip_on=False)


def curves(ax, mu, sd, pi):
    for j, c in enumerate((BLUE, RED)):
        ax.plot(xs, pi[j] * pdf(xs, mu[j], sd[j]), color=c, lw=3)
    ax.set_yticks([]); ax.set_xlabel("키 (cm)"); ax.set_xlim(135, 185); ax.set_ylim(-0.008, 0.05)


# 1. 질문
fig, ax = plt.subplots(figsize=(9, 2.6)); strip(ax, y=0)
ax.set_yticks([]); ax.set_xlabel("키 (cm)"); ax.set_xlim(135, 185); ax.set_ylim(-0.01, 0.01); ax.spines["left"].set_visible(False)
save(fig, "em_hook.png")

# 2. 시작 + E 단계
fig, ax = plt.subplots(figsize=(9, 3.8)); curves(ax, *P[0]); strip(ax, G[0])
ax.axvline(153, color=GOLD, ls="--", lw=2); ax.text(153.4, 0.044, "153 cm: 중3 52%", color="#7a6a3f", fontsize=15, fontweight="bold")
ax.text(137, 0.044, "파랑 = 초6 쪽, 빨강 = 중3 쪽", fontsize=13, color="#555555")
save(fig, "em_estep.png")

# 3. M 단계
fig, ax = plt.subplots(figsize=(9, 3.8)); curves(ax, *P[1]); strip(ax, G[0])
for j, c in enumerate((BLUE, RED)):
    ax.annotate("", (P[1][0][j], 0.046), (P[0][0][j], 0.046), arrowprops=dict(arrowstyle="->", lw=2.5, color=c))
    ax.text(P[1][0][j], 0.048, f"{P[1][0][j]:.0f}", color=c, fontsize=15, fontweight="bold", ha="center")
save(fig, "em_mstep.png")

# 4. 평균의 이동
fig, ax = plt.subplots(figsize=(8, 4.2))
mus = np.array([p[0] for p in P])
ax.plot(mus[:, 0], color=BLUE, lw=3, label="초6 쪽 평균"); ax.plot(mus[:, 1], color=RED, lw=3, label="중3 쪽 평균")
ax.axhline(np.mean(A), color=BLUE, ls=":", lw=1.5); ax.axhline(np.mean(B), color=RED, ls=":", lw=1.5)
ax.text(41, np.mean(A), "진짜 151", color=BLUE, va="center", fontsize=12); ax.text(41, np.mean(B), "진짜 167", color=RED, va="center", fontsize=12)
ax.set_xlabel("반복 횟수"); ax.set_ylabel("평균 (cm)"); ax.legend(frameon=False, loc="center right"); ax.set_xlim(0, 46)
save(fig, "em_traj.png")

# 5. 로그 우도
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.plot(LL, "o-", color=BLUE, lw=2, ms=4); ax.set_xlabel("반복 횟수"); ax.set_ylabel("로그 우도")
ax.annotate(f"{LL[0]:.1f}", (0, LL[0]), (3, LL[0] + 1), fontsize=13); ax.annotate(f"{LL[-1]:.2f}", (len(LL) - 1, LL[-1]), (30, LL[-1] - 4), fontsize=13)
save(fig, "em_ll.png")

# 6. 결과
fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), gridspec_kw={"height_ratios": [1.3, 1]}, sharex=True)
curves(axs[0], *P[-1]); strip(axs[0], G[-1]); axs[0].set_xlabel("")
g = G[-1][:, 1]
axs[1].bar(X, g, width=0.8, color=[CM(v) for v in g])
for i, x in enumerate(X):
    if (g[i] > 0.5) != TRUE[i]: axs[1].text(x, g[i] + 0.05, f"{int(x)}\n{g[i] * 100:.0f}%", ha="center", fontsize=11, color=RED)
axs[1].axhline(0.5, color=GRAY, ls="--"); axs[1].set_ylabel("중3일 확률"); axs[1].set_ylim(0, 1.3); axs[1].set_xlabel("키 (cm)")
save(fig, "em_final.png")
print("ok")
