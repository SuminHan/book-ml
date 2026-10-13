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

# 5. 그림으로 찾기: 상자 그림
vals = [v for _, v in REAL] + [TYPO_V, ATH_V]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.boxplot(vals, orientation="vertical", widths=0.5, whis=1.5, flierprops=dict(marker="o", markerfacecolor=RED, markersize=11, markeredgecolor=RED))
ax.scatter(np.ones(len(vals)) + np.random.default_rng(0).uniform(-0.08, 0.08, len(vals)), vals, color=BLUE, s=30, zorder=3)
q1, med, q3 = np.percentile(vals, [25, 50, 75]); iqr = q3 - q1
hi = max(v for v in vals if v <= q3 + 1.5 * iqr)
for y, t in [(q3, f"상자 윗변 = 75% 지점 ({q3:g})"), (med, f"가운데 선 = 50% 지점, 중앙값 ({med:g})"), (q1, f"상자 아랫변 = 25% 지점 ({q1:g})")]:
    ax.annotate(t, (1.25, y), (1.42, y), fontsize=13, va="center", arrowprops=dict(arrowstyle="-", color=GRAY))
ax.annotate(f"수염 끝 ({hi:g}): 상자 높이의 1.5배 안\n에서 가장 먼 값", (1.0, hi), (1.42, hi + 9), fontsize=12, va="center", color=GRAY, arrowprops=dict(arrowstyle="-", color=GRAY))
ax.annotate(f"{TYPO_V}", (1.0, TYPO_V), (0.8, TYPO_V - 3), fontsize=15, color=RED)
ax.annotate(f"{ATH_V}", (1.0, ATH_V), (0.8, ATH_V - 3), fontsize=15, color=RED)
ax.set_xlim(0.6, 2.4); ax.set_xticks([]); ax.set_ylabel("키 (cm)"); ax.spines["bottom"].set_visible(False)
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
