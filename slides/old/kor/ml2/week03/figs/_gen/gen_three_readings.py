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

def box(x, y, w, h, text, fc, ec=NAVY, fontsize=12, fontcolor="white", weight="bold"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.09,rounding_size=0.14",
                        linewidth=2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, color=fontcolor, weight=weight, zorder=3, linespacing=1.35)

# center: the equation itself
box(2.6, 4.55, 4.8, 0.95, "$V^\\pi(s)=\\mathbb{E}_\\pi[R_t+\\gamma V^\\pi(s_{t+1})]$",
    NAVY, fontsize=13.5)

arrow_specs = [
    (3.0, 4.55, 1.5, 3.35),
    (5.0, 4.55, 5.0, 3.35),
    (7.0, 4.55, 8.5, 3.35),
]
for x0, y0, x1, y1 in arrow_specs:
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                         linewidth=2.2, color=NAVY, zorder=1)
    ax.add_patch(a)

box(0.2, 2.45, 2.9, 0.9, "정의(definition)\n$V^\\pi$ 를 정의하는 식", TAN, ec=NAVY,
    fontcolor=NAVY, fontsize=11.5)
box(3.55, 2.45, 2.9, 0.9, "계산 절차(procedure)\n값이 안 바뀔 때까지\n반복 대입", TAN, ec=NAVY,
    fontcolor=NAVY, fontsize=11.5)
box(6.9, 2.45, 2.9, 0.9, "학습 목표(target)\n신경망이 좌변=우변\n되도록 학습", TAN, ec=NAVY,
    fontcolor=NAVY, fontsize=11.5)

arrow2 = [
    (1.65, 2.45, 1.65, 1.55, "이 절"),
    (5.0, 2.45, 5.0, 1.55, "Week 4\n동적계획법"),
    (8.35, 2.45, 8.35, 1.55, "Week 9\nDQN 손실"),
]
for x0, y0, x1, y1, label in arrow2:
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                         linewidth=2.0, color=RED, zorder=1)
    ax.add_patch(a)
    ax.text(x0, y1 - 0.15, label, ha="center", va="top", fontsize=10.5, color=RED, weight="bold")

ax.text(5.0, 0.35, "같은 재귀식을 어떤 형태로 계산·학습하는가 — 이 학기 알고리즘 지도",
        ha="center", va="center", fontsize=12, color=NAVY, weight="bold")

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week03/slides/kor/ml2/week03/figs/ch03_three_readings.png"
plt.savefig(out, dpi=200, facecolor="white")
print("saved", out)
