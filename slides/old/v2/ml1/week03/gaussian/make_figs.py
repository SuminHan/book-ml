"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다). 숫자는 data.py."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from math import comb
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#bbbbbb", "#2a9d8f"
O = "figs/"
vpdf = np.vectorize(pdf)


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name); plt.close(fig)


# 1. 골턴 보드 (못 + 아래 칸)
fig, ax = plt.subplots(figsize=(7, 6))
for r in range(ROWS):
    for j in range(r + 1):
        ax.add_patch(Circle((j - r / 2, -r), 0.12, color="#555555"))
ax.add_patch(Circle((0, 1.0), 0.28, color=GOLD))
ax.annotate("", (0, 0.35), (0, 0.75), arrowprops=dict(arrowstyle="->", lw=2, color=GOLD))
xs = np.arange(ROWS + 1) - ROWS / 2
ax.bar(xs, np.array(BINS) / 40, bottom=-ROWS - 6.5, width=0.8, color=BLUE, alpha=0.25)
ax.set_xlim(-6.3, 6.3); ax.set_ylim(-ROWS - 7, 1.6); ax.set_aspect("equal"); ax.axis("off")
save(fig, "ga_board.png")

# 2. 결과 막대
fig, ax = plt.subplots(figsize=(8.5, 4.4))
k = np.arange(ROWS + 1)
b = ax.bar(k, BINS, color=BLUE, width=0.8)
for kk, v in zip(k, BINS):
    ax.text(kk, v + 5, str(v), ha="center", fontsize=13)
ax.set_xticks(k); ax.set_xlabel("오른쪽으로 튄 횟수 (10번 중)"); ax.set_ylabel("공 개수"); ax.set_ylim(0, 280)
save(fig, "ga_bins.png")

# 3. 작은 효과의 합
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
for ax, n in zip(axs, (1, 2, 10, 50)):
    sums = rng.integers(0, 2, (20000, n)).sum(1)
    ax.hist(sums, bins=np.arange(n + 2) - 0.5, color=BLUE, alpha=0.8); ax.set_yticks([])
    ax.set_title(f"동전 {n}개의 합", fontsize=15)
save(fig, "ga_sum.png")

# 4. 평균 · 표준편차 바꾸기
x = np.linspace(-8, 8, 400)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for mu, c in ((-2, GRAY), (0, BLUE), (3, RED)):
    axs[0].plot(x, vpdf(x, mu, 1), color=c, lw=3, label=f"μ = {mu}")
for sd, c in ((0.6, RED), (1, BLUE), (2, GRAY)):
    axs[1].plot(x, vpdf(x, 0, sd), color=c, lw=3, label=f"σ = {sd}")
axs[0].set_title("평균 μ: 가운데 위치", fontsize=16); axs[1].set_title("표준편차 σ: 퍼진 정도", fontsize=16)
for ax in axs: ax.legend(frameon=False); ax.set_yticks([])
save(fig, "ga_musd.png")

# 5. 68 · 95
x = np.linspace(-4, 4, 400)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(x, vpdf(x, 0, 1), color=BLUE, lw=3)
for lo, c, a in ((2, BLUE, 0.15), (1, BLUE, 0.3)):
    m = (x >= -lo) & (x <= lo); ax.fill_between(x[m], vpdf(x[m], 0, 1), color=c, alpha=a)
ax.annotate("", (-1, 0.27), (1, 0.27), arrowprops=dict(arrowstyle="<->", lw=2)); ax.text(0, 0.285, "68%", ha="center", fontsize=18, fontweight="bold")
ax.annotate("", (-2, 0.06), (2, 0.06), arrowprops=dict(arrowstyle="<->", lw=2)); ax.text(0, 0.075, "95%", ha="center", fontsize=18, fontweight="bold")
ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_xticklabels(["μ−2σ", "μ−σ", "μ", "μ+σ", "μ+2σ"]); ax.set_yticks([])
save(fig, "ga_6895.png")

# 6. 시험 90점 이상
x = np.linspace(30, 110, 400)
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(x, vpdf(x, EXAM_MU, EXAM_SD), color=BLUE, lw=3)
m = x >= EXAM_CUT; ax.fill_between(x[m], vpdf(x[m], EXAM_MU, EXAM_SD), color=RED, alpha=0.5)
ax.annotate("90점 이상 2.3%", (95, 0.003), (93, 0.02), fontsize=17, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xticks([50, 60, 70, 80, 90]); ax.set_xlabel("시험 점수"); ax.set_yticks([])
save(fig, "ga_exam.png")

# 7. 키 두 종
x = np.linspace(140, 195, 500)
def heights(ax, mark=True, prior=0.5):
    ax.plot(x, vpdf(x, *MALE) * prior * 2, color=BLUE, lw=3, label=f"남 (평균 {MALE[0]}, σ {MALE[1]})")
    ax.plot(x, vpdf(x, *FEMALE) * (1 - prior) * 2, color=RED, lw=3, label=f"여 (평균 {FEMALE[0]}, σ {FEMALE[1]})")
    ax.set_xlabel("키 (cm)"); ax.set_yticks([]); ax.legend(frameon=False, loc="upper left")
fig, ax = plt.subplots(figsize=(8.5, 4.2)); heights(ax)
ax.axvline(X_Q, color=GOLD, lw=2.5, ls="--")
ax.scatter([X_Q, X_Q], [pdf(X_Q, *MALE), pdf(X_Q, *FEMALE)], color=[BLUE, RED], s=90, zorder=4)
ax.text(X_Q + 0.8, 0.075, f"{X_Q} cm", color="#7a6a3f", fontsize=16, fontweight="bold")
ax.text(X_Q + 0.8, pdf(X_Q, *MALE), f"{pdf(X_Q, *MALE):.4f}", color=BLUE, fontsize=13, va="bottom")
ax.text(X_Q - 0.8, pdf(X_Q, *FEMALE), f"{pdf(X_Q, *FEMALE):.4f}", color=RED, fontsize=13, ha="right", va="top")
save(fig, "ga_heights.png")

# 8. 경계
fig, ax = plt.subplots(figsize=(8.5, 4.2)); heights(ax)
ax.axvline(BOUNDARY, color="black", lw=2.5)
ax.axvspan(140, BOUNDARY, color=RED, alpha=0.07); ax.axvspan(BOUNDARY, 195, color=BLUE, alpha=0.07)
ax.text(BOUNDARY + 0.5, 0.075, f"경계 {BOUNDARY} cm", fontsize=16, fontweight="bold")
save(fig, "ga_boundary.png")

# 9. 남 80% 교실
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
for ax, pr, t in ((axs[0], 0.5, "남 50% 교실"), (axs[1], PRIOR_MALE, "남 80% 교실")):
    heights(ax, prior=pr); ax.axvline(X_Q2, color=GOLD, lw=2.5, ls="--")
    ax.set_title(f"{t}: {X_Q2} cm → 남일 확률 {round(posterior(X_Q2, pr) * 100)}%", fontsize=15)
    if ax is axs[1]: ax.get_legend().remove()
save(fig, "ga_prior.png")
print("ok")
