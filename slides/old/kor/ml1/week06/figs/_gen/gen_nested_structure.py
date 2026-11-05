import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
SAND = "#D6CBB1"
RED = "#C0392B"
GREEN = "#2E8B57"

fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.2)
ax.axis("off")

# outer fold blocks: 5 cells along top, one highlighted as outer-validation
n = 5
cell_w = 1.6
x0 = 0.6
y_top = 4.1
for i in range(n):
    x = x0 + i * cell_w
    color = GREEN if i == 2 else NAVY
    alpha = 0.85 if i == 2 else 0.55
    ax.add_patch(patches.Rectangle((x, y_top), cell_w - 0.12, 0.7,
                                    facecolor=color, alpha=alpha, edgecolor="white"))
    label = "outer\nval" if i == 2 else "fold"
    ax.text(x + (cell_w - 0.12) / 2, y_top + 0.35, label, ha="center", va="center",
            fontsize=10.5, color="white", fontweight="bold")

ax.text(x0 - 0.3, y_top + 0.35, "외층\n(outer)", ha="right", va="center", fontsize=11.5, color=NAVY)
ax.text(x0 + n * cell_w / 2 - 0.5, y_top + 1.05, "outer fold: 검증용 점 1개(녹색)만 떼어둔다",
        ha="center", fontsize=11, color="#333333")

# arrow down to inner box
ax.annotate("", xy=(5.0, 3.55), xytext=(5.0, 3.95),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8))
ax.text(5.35, 3.7, "나머지 4조각", fontsize=10, color="#555555")

# inner box: k-fold over remaining data to pick lambda
inner_box = patches.FancyBboxPatch((1.0, 1.5), 8.0, 1.9, boxstyle="round,pad=0.05",
                                    linewidth=1.6, edgecolor=SAND, facecolor="#FBF6EE")
ax.add_patch(inner_box)
ax.text(5.0, 3.15, "내층 (inner): 남은 데이터로 k-fold CV → $\\lambda$ 선택",
        ha="center", fontsize=11.5, color="#8a6d3b", fontweight="bold")

m = 4
cell_w2 = 1.7
x1 = 1.5
y_inner = 1.9
for i in range(m):
    x = x1 + i * cell_w2
    color = RED if i == 1 else NAVY
    alpha = 0.55 if i != 1 else 0.75
    ax.add_patch(patches.Rectangle((x, y_inner), cell_w2 - 0.15, 0.6,
                                    facecolor=color, alpha=alpha, edgecolor="white"))
    label = "inner\nval" if i == 1 else "train"
    ax.text(x + (cell_w2 - 0.15) / 2, y_inner + 0.3, label, ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")

ax.text(5.0, 1.65, "선택은 여기에서만", ha="center", fontsize=10.5, color="#8a6d3b", style="italic")

# arrow from inner box to final model
ax.annotate("", xy=(5.0, 1.15), xytext=(5.0, 1.45),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
ax.text(5.0, 0.85, "선택된 $\\lambda$로 학습한 모델 → outer val(녹색)에서만 채점",
        ha="center", fontsize=11, color=GREEN, fontweight="bold")

ax.text(0.4, 0.15, "정보 흐름: 내층 → 외층. 외층 검증 데이터는 이 그림 어디의 선택에도 쓰이지 않는다.",
        fontsize=10, color="#555555")

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week06/slides/kor/ml1/week06/figs/ch06_2_nested_structure.png",
            facecolor="white")
