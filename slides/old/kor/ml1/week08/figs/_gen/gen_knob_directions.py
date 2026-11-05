import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2E3192"
RED = "#C0392B"
GRAY = "#5B5B5B"

fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.2)
ax.axis("off")

def slider(y, color, label, left_txt, right_txt, arrow_dir):
    ax.add_line(plt.Line2D([1, 9], [y, y], color=color, linewidth=4, solid_capstyle="round"))
    ax.text(0.5, y, label, fontsize=15, color=color, ha="right", va="center", fontweight="bold")
    ax.text(1, y+0.55, "0", fontsize=12.5, ha="center", color=GRAY)
    ax.text(9, y+0.55, "$\\infty$", fontsize=13.5, ha="center", color=GRAY)
    if arrow_dir == "right":
        a = FancyArrowPatch((1.3, y-0.55), (8.7, y-0.55), arrowstyle="-|>",
                             mutation_scale=20, color=color, linewidth=2.4)
        ax.add_patch(a)
        ax.text(1.3, y-0.95, left_txt, fontsize=12.5, ha="left", color=color)
        ax.text(8.7, y-0.95, right_txt, fontsize=12.5, ha="right", color=color)
    else:
        a = FancyArrowPatch((8.7, y-0.55), (1.3, y-0.55), arrowstyle="-|>",
                             mutation_scale=20, color=color, linewidth=2.4)
        ax.add_patch(a)
        ax.text(1.3, y-0.95, right_txt, fontsize=12.5, ha="left", color=color)
        ax.text(8.7, y-0.95, left_txt, fontsize=12.5, ha="right", color=color)

slider(3.9, BLUE, "Ridge\n$\\lambda$", "적합만(과적합)", "단순성만($w{\\to}0$)", "right")
slider(1.6, RED, "SVM\n$C$", "단순성만($w{=}0$)", "적합만(하드마진)", "right")

ax.text(5.0, 5.0, "$\\lambda \\uparrow$: 단순해진다   vs   $C \\uparrow$: 적합해진다 --- 정반대 방향",
         fontsize=13.5, ha="center", color="black", fontweight="bold")

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_3_knob_directions.png",
            bbox_inches="tight", facecolor="white")
print("saved")
