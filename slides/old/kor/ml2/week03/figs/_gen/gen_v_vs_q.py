import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
TAN = "#D6CBB1"
RED = "#C0392B"

fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.4)
ax.axis("off")

ax.text(2.4, 5.05, "$V^\\pi$: 상태별 가격표", ha="center", fontsize=14, color=NAVY, weight="bold")
ax.text(7.6, 5.05, "$Q^\\pi$: 상태-행동별 가격표", ha="center", fontsize=14, color=RED, weight="bold")

# --- V side: 3 states as circles with a single price each ---
states = [(1.1, 3.6), (2.4, 3.6), (3.7, 3.6)]
vvals = ["V=2.4", "V=5.1", "V=1.8"]
for (x, y), v in zip(states, vvals):
    ax.add_patch(Circle((x, y), 0.55, facecolor=NAVY, edgecolor=NAVY, zorder=2))
    ax.text(x, y, "s", ha="center", va="center", fontsize=13, color="white", weight="bold", zorder=3)
    ax.text(x, y - 0.95, v, ha="center", fontsize=12, color=NAVY, weight="bold")
ax.text(2.4, 2.15, "상태 하나 = 숫자 하나", ha="center", fontsize=11.5, color="#333333")

# --- Q side: 1 state with 2 actions, each its own price ---
sx, sy = 7.6, 3.6
ax.add_patch(Circle((sx, sy), 0.5, facecolor=RED, edgecolor=RED, zorder=2))
ax.text(sx, sy, "s", ha="center", va="center", fontsize=13, color="white", weight="bold", zorder=3)

for dx, label, q in [(-1.3, "$a_0$", "Q=1+0.9V(S1)"), (1.3, "$a_1$", "Q=3+0.9V(S1)")]:
    ax.add_patch(FancyArrowPatch((sx, sy), (sx + dx, sy - 1.1), arrowstyle="-|>",
                                  mutation_scale=16, linewidth=2, color=RED, zorder=1))
    ax.text(sx + dx, sy - 1.35, label + "\n" + q, ha="center", fontsize=11, color=RED, weight="bold")
ax.text(7.6, 2.15, "상태-행동 쌍마다 숫자 하나", ha="center", fontsize=11.5, color="#333333")

# divider
ax.plot([5.0, 5.0], [0.3, 5.3], color="#cccccc", lw=1.5, ls="--")

ax.text(5.0, 0.7,
        "둘 다 “앞으로 평균적으로 얼마를 버는가”의 기댓값 —\n"
        "$V^\\pi$는 상태 단위, $Q^\\pi$는 (상태,행동) 단위로 쪼갠 것",
        ha="center", va="center", fontsize=12, color=NAVY,
        weight="bold")

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week03/slides/kor/ml2/week03/figs/ch03_v_vs_q.png"
plt.savefig(out, dpi=200, facecolor="white")
print("saved", out)
