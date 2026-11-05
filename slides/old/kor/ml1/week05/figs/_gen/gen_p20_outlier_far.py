"""p20g1: 서포트 벡터가 아닌 점 (4,3)을 (100,100)으로 옮겨도 경계 불변."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
RED = "#C0392B"
TAN = "#D6CBB1"

pos = np.array([[3, 3], [4, 3], [3, 4]])
neg = np.array([[-3, -3], [-4, -3], [-3, -4]])
w = np.array([1/6, 1/6]); b = 0.0

fig, ax = plt.subplots(figsize=(8.4, 6.6))
x1 = np.linspace(-6, 6, 100)
f = lambda c: (-w[0] * x1 - b + c) / w[1]
ax.fill_between(x1, f(1), f(-1), color=TAN, alpha=0.4, zorder=0)
ax.plot(x1, f(0), color="black", lw=2.4, zorder=1, label="결정 경계 (변화 없음)")
ax.plot(x1, f(1), color=NAVY, lw=1.2, ls="--", zorder=1)
ax.plot(x1, f(-1), color=NAVY, lw=1.2, ls="--", zorder=1)

ax.scatter(pos[:, 0], pos[:, 1], c=NAVY, s=95, zorder=3)
ax.scatter(neg[:, 0], neg[:, 1], c=RED, s=95, marker="s", zorder=3)
ax.scatter([3], [3], s=230, facecolors="none", edgecolors="#E67E22", lw=2.5, zorder=4)
ax.scatter([-3], [-3], s=230, facecolors="none", edgecolors="#E67E22", lw=2.5, zorder=4)

for p in np.vstack([pos, neg]):
    ax.annotate(f"({p[0]:g},{p[1]:g})", p, textcoords="offset points",
                xytext=(8, 6), fontsize=10)

# annotate (4,3) moving far away, off-canvas
ax.annotate("", xy=(5.7, 5.7), xytext=(4, 3),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.0, ls=(0, (4, 2))), zorder=3)
ax.text(4.9, 5.55, "(100,100)으로\n이동해도\n경계 불변", fontsize=10.8, color="#333333",
        ha="left", va="bottom")

ax.set_xlim(-6, 6.4); ax.set_ylim(-6, 6.4); ax.set_aspect("equal")
ax.set_xlabel("$x_1$", fontsize=12); ax.set_ylabel("$x_2$", fontsize=12)
ax.set_title("서포트 벡터가 아닌 점은 멀리 옮겨도 경계가 그대로", fontsize=13.5, pad=10)
ax.tick_params(labelsize=10)

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week05/slides/kor/ml1/week05/figs/ch05_1_outlier_far.png", dpi=200, facecolor="white")
print("done")
