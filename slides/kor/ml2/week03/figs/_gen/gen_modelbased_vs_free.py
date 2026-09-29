import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
TAN = "#D6CBB1"
RED = "#C0392B"

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

def box(x, y, w, h, text, fc, ec=NAVY, fontsize=13, fontcolor="white", weight="bold"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color=fontcolor, weight=weight, zorder=3)

def arrow(x0, y0, x1, y1, color=NAVY, lw=2.2):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                         linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)

# top: environment
box(3.3, 5.0, 3.4, 0.85, "환경 (Environment)\n같은 P, 같은 샘플", NAVY, fontsize=13)

# arrows down to two branches
arrow(4.3, 5.0, 1.9, 3.9)
arrow(5.7, 5.0, 8.1, 3.9)

# left branch: model-based
box(0.2, 3.05, 3.4, 0.85, "P 를 (추정해서)\n명시적으로 쓴다", TAN, ec=NAVY, fontcolor=NAVY, fontsize=12.5)
arrow(1.9, 3.05, 1.9, 2.15)
box(0.2, 1.3, 3.4, 0.85, "모델 기반\n(Week 4–5, 동적계획법)", NAVY, fontsize=12.5)

# right branch: model-free
box(6.4, 3.05, 3.4, 0.85, "P 는 무시하고\n경험(샘플)만 쓴다", TAN, ec=NAVY, fontcolor=NAVY, fontsize=12.5)
arrow(8.1, 3.05, 8.1, 2.15)
box(6.4, 1.3, 3.4, 0.85, "모델 없는\n(Week 6–11, Q-러닝·DQN)", NAVY, fontsize=12.5)

ax.text(5.0, 0.55, "이 “$P$ 의 사용 방식”이 알고리즘의 가족을 갈라놓는다",
        ha="center", va="center", fontsize=12.5, color=RED, weight="bold")

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week03/slides/kor/ml2/week03/figs/ch03_modelbased_vs_free.png"
plt.savefig(out, dpi=200, facecolor="white")
print("saved", out)
