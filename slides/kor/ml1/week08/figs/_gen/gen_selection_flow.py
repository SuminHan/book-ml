import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2E3192"
GOLD = "#D6CBB1"
RED = "#C0392B"
GRAY = "#5B5B5B"

fig, ax = plt.subplots(figsize=(12.5, 5.1), dpi=200)
ax.set_xlim(0, 12.5)
ax.set_ylim(-0.4, 4.6)
ax.axis("off")

def box(cx, cy, w, h, text, fc="white", ec=BLUE, fontsize=13, textcolor="black", lw=1.6):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle="round,pad=0.06,rounding_size=0.12",
                        linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, linespacing=1.35)

def arrow(x0, y0, x1, y1, color=GRAY, style="-", lw=2.2):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                          linewidth=lw, color=color, linestyle=style)
    ax.add_patch(a)

# stage boxes
box(1.55, 3.15, 2.7, 1.5, "후보 모델 여럿\n(로지스틱·kNN·트리\nRF·GBDT $\\cdots$)", fc="#F4F3EF", ec=GRAY)
box(4.65, 3.15, 2.7, 1.5, "각 후보를\n$\\bf{train}$으로만 fit", fc="white", ec=BLUE)
box(7.75, 3.15, 2.9, 1.5, "$\\bf{val}$ 성능으로 순위\n(교차검증 포함)", fc="white", ec=BLUE)
box(11.0, 3.15, 2.6, 1.5, "val 1등\n선택", fc=GOLD, ec="#A9976B")

arrow(2.9, 3.15, 3.3, 3.15)
arrow(6.0, 3.15, 6.4, 3.15)
arrow(9.2, 3.15, 9.7, 3.15)

box(11.0, 0.95, 2.6, 1.4, "그 모델만\ntest 적용", fc="white", ec=RED, lw=2.0)
box(7.6, 0.95, 3.6, 1.4, "최종 숫자 1회 확정\n(accuracy·F1·PR-AUC)", fc="#FBEAE7", ec=RED, lw=2.0)

arrow(11.0, 2.4, 11.0, 1.65, color=RED, lw=2.4)
arrow(9.7, 0.95, 9.4, 0.95, color=RED, lw=2.4)

ax.text(11.0, -0.05, "test는 이 화살표를 딱 한 번만 지나간다", fontsize=10.5,
         color=RED, ha="center", fontweight="bold")
ax.text(1.55, 1.3, "다른 후보들은\ntest를 보지 않는다", fontsize=11.5, color=GRAY,
         ha="center", linespacing=1.4)
arrow(1.7, 2.35, 1.7, 1.75, color=GRAY, style=(0, (4, 3)), lw=1.6)

fig.suptitle("모델 비교의 기준: val로 고르고, test는 한 번만", fontsize=17, y=1.02)
fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_1_selection_flow.png",
            bbox_inches="tight", facecolor="white")
print("saved")
