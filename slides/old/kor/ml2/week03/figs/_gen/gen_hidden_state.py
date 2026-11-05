import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
TAN = "#D6CBB1"
RED = "#C0392B"

fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.6)
ax.axis("off")

def box(x, y, w, h, text, fc, ec=NAVY, fontsize=12.5, fontcolor="white", weight="bold"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color=fontcolor, weight=weight, zorder=3)

def arrow(x0, y0, x1, y1, color=NAVY, lw=2.2, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                         linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)

# center: single observation (camera icon as simple box)
box(3.3, 4.2, 3.4, 0.9, "같은 관측 $o_t$\n(카메라 한 장, 자동차 화면)", NAVY, fontsize=12.5)

arrow(4.3, 4.2, 1.9, 3.15)
arrow(5.7, 4.2, 8.1, 3.15)

# left: fast
box(0.2, 2.25, 3.4, 0.85, "실제 상태: 빠르게 이동 중\n(속도 정보는 화면 밖)", TAN, ec=NAVY,
    fontcolor=NAVY, fontsize=12)
arrow(1.9, 2.25, 1.9, 1.35, color=RED)
box(0.2, 0.5, 3.4, 0.8, "미래: 다음 프레임에서\n크게 이동", RED, ec=RED, fontsize=12)

# right: slow
box(6.4, 2.25, 3.4, 0.85, "실제 상태: 느리게 이동 중\n(속도 정보는 화면 밖)", TAN, ec=NAVY,
    fontcolor=NAVY, fontsize=12)
arrow(8.1, 2.25, 8.1, 1.35, color=RED)
box(6.4, 0.5, 3.4, 0.8, "미래: 다음 프레임에서\n조금 이동", RED, ec=RED, fontsize=12)

ax.text(5.0, 5.15, "같은 관측, 다른 미래 분포  →  마르코프 성질 깨짐",
        ha="center", va="center", fontsize=13, color=RED, weight="bold")

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week03/slides/kor/ml2/week03/figs/ch03_hidden_state_speed.png"
plt.savefig(out, dpi=200, facecolor="white")
print("saved", out)
