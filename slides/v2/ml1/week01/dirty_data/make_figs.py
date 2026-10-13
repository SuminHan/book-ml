"""이 덱의 그림 원본. 실행: ~/Documents/manim_shorts/_env/v/bin/python make_figs.py  (figs/*.png 를 다시 만든다)
숫자는 data.py (쇼츠와 같은 반 10명 키 예시). 마지막 그림은 scikit-learn 내장 iris."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from data import *
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
BLUE, RED, GOLD, GRAY, GREEN = "#2E3192", "#d62728", "#A9976B", "#888888", "#2a9d8f"
O = "figs/"


def bars(ax, rows, mean_v, colors=None, ylim=(0, 270), mean_label=None):
    names = [n for n, _ in rows]; vals = [v for _, v in rows]
    cs = colors or [BLUE] * len(rows)
    ax.bar(range(len(rows)), vals, color=cs, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, f"{v}", ha="center", fontsize=13, color=cs[i] if cs[i] != BLUE else "black")
    ax.set_xticks(range(len(rows)), names, fontsize=13); ax.set_ylim(*ylim); ax.set_ylabel("키 (cm)")
    ax.axhline(mean_v, color=RED, ls="--", lw=2.5)
    ax.set_xlim(-0.6, len(rows) + 1.3)
    ax.text(len(rows) + 1.25, mean_v, mean_label or f"평균\n{mean_v:g}cm", ha="right", va="center", color=RED, fontsize=16,
            fontweight="bold", bbox=dict(fc="white", ec="none", pad=1))


# 1. 질문: 평균 131.2
fig, ax = plt.subplots(figsize=(10, 4.4))
bars(ax, CLASS, round(MEAN_DIRTY, 1), [GOLD if v == 0 else BLUE for _, v in CLASS], ylim=(0, 200))
fig.tight_layout(); fig.savefig(O + "dd_hook.png"); plt.close(fig)

# 2. 결측치 0 을 빼면 164
fig, ax = plt.subplots(figsize=(10, 4.4))
bars(ax, REAL, MEAN_CLEAN, ylim=(0, 200))
fig.tight_layout(); fig.savefig(O + "dd_clean.png"); plt.close(fig)

# 3. 중복: 민수 ×3 → 166
rows = REAL + [(DUP, DUP_V)] * 2
fig, ax = plt.subplots(figsize=(10.5, 4.4))
bars(ax, rows, MEAN_DUP, [RED if n == DUP else BLUE for n, _ in rows], ylim=(0, 200))
fig.tight_layout(); fig.savefig(O + "dd_dup.png"); plt.close(fig)

# 4. 이상치: 오타 250 vs 농구 선수 205
rows = REAL + [(TYPO, TYPO_V), (ATH, ATH_V)]
cs = [RED if n == TYPO else (GOLD if n == ATH else BLUE) for n, _ in rows]
fig, ax = plt.subplots(figsize=(10.5, 4.4))
bars(ax, rows, round(mean([v for _, v in rows]), 1), cs, ylim=(0, 290))
ax.text(len(rows) - 2, TYPO_V + 18, "오타?", ha="center", color=RED, fontsize=15, fontweight="bold")
ax.text(len(rows) - 1, ATH_V + 18, "농구 선수", ha="center", color=GOLD, fontsize=15, fontweight="bold")
fig.tight_layout(); fig.savefig(O + "dd_outlier.png"); plt.close(fig)

# 5. 그림으로 찾기: 상자 그림 (세로) — 왼쪽: 읽는 법, 오른쪽: 우리 반 키
vals = np.array([v for _, v in REAL] + [TYPO_V, ATH_V])
q1, med, q3 = np.percentile(vals, [25, 50, 75]); iqr = q3 - q1; fence = q3 + 1.5 * iqr
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
# 읽는 법 (모양만 보여 주는 그림)
from matplotlib.patches import Rectangle
a1.add_patch(Rectangle((0.3, 3), 0.8, 3, fc="#eef0fa", ec="black", lw=1.6))
a1.plot([0.3, 1.1], [4.3, 4.3], color=RED, lw=3)
a1.plot([0.7, 0.7], [6, 7.6], color="black", lw=1.4); a1.plot([0.55, 0.85], [7.6, 7.6], color="black", lw=1.4)
a1.plot([0.7, 0.7], [3, 1.6], color="black", lw=1.4); a1.plot([0.55, 0.85], [1.6, 1.6], color="black", lw=1.4)
a1.scatter([0.7], [9.3], s=110, color=RED)
for y, t, c in ((9.3, "이상치 후보: 수염보다 바깥", RED), (7.6, "수염 끝: 상자에서 상자 길이 × 1.5 안의 가장 먼 값", GRAY),
                (6, "75% (Q3): 이보다 큰 값이 1/4", "black"), (4.3, "중앙값 (50%): 한가운데", RED),
                (3, "25% (Q1): 이보다 작은 값이 1/4", "black"), (1.6, "수염 끝", GRAY)):
    a1.text(1.3, y, t, va="center", fontsize=13.5, color=c)
a1.annotate("", (0.2, 6), (0.2, 3), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
a1.text(0.12, 4.5, "상자 = 가운데 절반", rotation=90, va="center", ha="right", fontsize=13, color=BLUE)
a1.set_xlim(-0.3, 6.2); a1.set_ylim(0.8, 10.2); a1.axis("off"); a1.set_title("상자 그림 읽는 법", fontsize=16, fontweight="bold")
# 우리 반 키
a2.boxplot(vals, positions=[0], widths=0.45, medianprops=dict(color=RED, lw=2.5),
           flierprops=dict(marker="o", markerfacecolor=RED, markersize=11, markeredgecolor=RED))
a2.axhline(fence, color=GRAY, ls=":", lw=1.5); a2.text(0.32, fence + 2, f"Q3 + 1.5 × 상자 길이 = {fence:g}", color=GRAY, fontsize=12)
for v in (TYPO_V, ATH_V):
    a2.text(0.32, v, f"{v} cm", color=RED, fontsize=14, va="center", fontweight="bold")
a2.text(0.32, med, f"중앙값 {med:g}", color=RED, fontsize=12.5, va="center")
a2.set_xlim(-0.6, 1.6); a2.set_xticks([]); a2.set_ylabel("키 (cm)"); a2.spines["bottom"].set_visible(False)
a2.set_title("우리 반 키 (10명)", fontsize=16, fontweight="bold")
fig.tight_layout(); fig.savefig(O + "dd_box.png"); plt.close(fig)

# 6. 모델도 속는다: iris 꽃잎 길이 → 너비, '비었음'을 0으로 적은 꽃 6송이
from sklearn.datasets import load_iris
X = load_iris().data; pl, pw = X[:, 2], X[:, 3]
rng = np.random.default_rng(1)
bad = rng.choice(np.where(pl > 4.5)[0], 6, replace=False)
pw_dirty = pw.copy(); pw_dirty[bad] = 0
xs = np.linspace(0.8, 7.2, 50)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.scatter(pl, pw_dirty, s=22, color=BLUE, alpha=0.6)
ax.scatter(pl[bad], pw_dirty[bad], s=90, color=RED, zorder=3, label="'비었음'을 0으로 적은 6송이")
for y_, c_, lab in ((pw, GREEN, "깨끗한 데이터의 직선"), (pw_dirty, RED, "0 이 섞인 데이터의 직선")):
    k = np.polyfit(pl, y_, 1); ax.plot(xs, np.polyval(k, xs), color=c_, lw=2.8, label=f"{lab} (기울기 {k[0]:.2f})")
ax.set_xlabel("꽃잎 길이 (cm)"); ax.set_ylabel("꽃잎 너비 (cm)"); ax.legend(frameon=False, fontsize=12.5, loc="upper left")
fig.tight_layout(); fig.savefig(O + "dd_iris.png"); plt.close(fig)
print("slopes", [round(np.polyfit(pl, y_, 1)[0], 3) for y_ in (pw, pw_dirty)])
