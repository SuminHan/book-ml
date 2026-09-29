"""p08g1: w가 경계에 수직, 1만큼 이동 = w방향으로 1/||w|| 이동."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
TAN = "#D6CBB1"
RED = "#C0392B"

# boundary: x1 + x2 = 0  (w = (1,1)/sqrt2 direction, unit-normalized for clean picture)
w_dir = np.array([1.0, 1.0]) / np.sqrt(2)   # unit vector along w
fig, ax = plt.subplots(figsize=(7.4, 6.4))

t = np.linspace(-4, 4, 50)
perp = np.array([-1.0, 1.0]) / np.sqrt(2)   # direction along the boundary line

# decision boundary f=0 through origin
line0 = np.outer(t, perp)
ax.plot(line0[:, 0], line0[:, 1], color="black", lw=2.4, zorder=2)
ax.text(2.9, -3.5, "$w^\\top x + b = 0$", fontsize=13, ha="left")

# margin boundary f=1, shifted along w_dir by distance 1/||w|| -- pick ||w||=0.5 for a clean 1/||w||=2 picture
w_norm = 0.5
dist = 1.0 / w_norm
shift = w_dir * dist
line1 = shift + np.outer(t, perp)
ax.plot(line1[:, 0], line1[:, 1], color=NAVY, lw=2.2, ls="--", zorder=2)
ax.text(shift[0] + 2.6, shift[1] - 3.1, "$w^\\top x + b = 1$", fontsize=13, color=NAVY, ha="left")

# points x0 (on boundary) and x_+ (on margin boundary), connected along w direction
x0 = np.array([0.0, 0.0])
xplus = x0 + shift
ax.scatter(*x0, color="black", s=70, zorder=5)
ax.scatter(*xplus, color="#E67E22", s=110, zorder=5, edgecolors="black", linewidths=1.2)
ax.annotate("", xy=xplus, xytext=x0,
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.6, shrinkA=0, shrinkB=0), zorder=4)
mid = (x0 + xplus) / 2
ax.text(mid[0] + 0.35, mid[1] + 0.15, r"$\frac{1}{\|w\|}$", fontsize=19, color=RED)

ax.text(x0[0] - 0.55, x0[1] - 0.55, "$x_0$", fontsize=13)
ax.text(xplus[0] + 0.25, xplus[1] + 0.35, "$x_+$", fontsize=13, color="#E67E22")

# w vector arrow (direction), drawn separately near origin, small inset arrow
w_arrow_start = np.array([-3.0, 3.0])
w_arrow_end = w_arrow_start + w_dir * 1.6
ax.annotate("", xy=w_arrow_end, xytext=w_arrow_start,
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.4), zorder=4)
ax.text(w_arrow_end[0] + 0.15, w_arrow_end[1] + 0.05, "$w$ (경계에 수직)", fontsize=12.5, color=NAVY)

ax.set_xlim(-5, 6)
ax.set_ylim(-5.5, 5)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("$w$ 방향으로 $1/\\|w\\|$ 이동하면 $w^\\top x+b$ 가 정확히 1만큼 변한다", fontsize=13.5, pad=12)

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week05/slides/kor/ml1/week05/figs/ch05_1_geometry.png", dpi=200, facecolor="white", bbox_inches="tight")
print("done")
