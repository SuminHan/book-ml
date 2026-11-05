import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = "#2E3192"
KSA_TAN = "#D6CBB1"
RED = "#C0392B"
GRAY = "#8a8a8a"

OUT = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week17/slides/kor/ml1/week17/figs"

# ---------------------------------------------------------------
# D4 (p18g1): layer-freeze diagram -- front layers frozen, back layers fine-tuned
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=200)
n_layers = 8
n_frozen = 5
box_w, box_h, gap = 0.9, 1.6, 0.15
x0 = 0.3

for i in range(n_layers):
    x = x0 + i * (box_w + gap)
    frozen = i < n_frozen
    color = "#d9d9d9" if frozen else KSA_BLUE
    edge = GRAY if frozen else "#1b1e63"
    rect = FancyBboxPatch((x, 0), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.05",
                           linewidth=1.8, edgecolor=edge, facecolor=color, alpha=0.95)
    ax.add_patch(rect)
    txt_color = "#555555" if frozen else "white"
    ax.text(x + box_w / 2, box_h / 2, f"층 {i+1}", ha="center", va="center",
            fontsize=12, color=txt_color, fontweight="bold")
    if frozen:
        ax.text(x + box_w / 2, box_h + 0.25, "고정", ha="center", va="center",
                fontsize=11, color=GRAY, fontweight="bold")

x_end = x0 + n_layers * (box_w + gap) - gap

# bottom arrow: input -> frozen layers
ax.annotate("", xy=(x0 + n_frozen * (box_w + gap) - gap / 2, -0.35),
            xytext=(x0, -0.35),
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=2))
ax.text(x0 + (n_frozen * (box_w + gap) - gap) / 2, -0.62,
        "문법·의미 패턴 (저수준) --- 고정", ha="center", fontsize=12.5, color="#555555")

ax.annotate("", xy=(x_end, -0.35),
            xytext=(x0 + n_frozen * (box_w + gap), -0.35),
            arrowprops=dict(arrowstyle="-", color=KSA_BLUE, lw=2))
ax.text(x0 + n_frozen * (box_w + gap) + (n_layers - n_frozen) * (box_w + gap) / 2 - gap / 2, -0.62,
        "행동/형식 매핑 (고수준) --- 파인튜닝", ha="center", fontsize=12.5,
        color=KSA_BLUE, fontweight="bold")

ax.annotate("입력 토큰", xy=(x0 + box_w / 2, 0), xytext=(x0 + box_w / 2, -1.2),
            ha="center", fontsize=13,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.annotate("출력 (답변 형식)", xy=(x_end - box_w / 2, box_h), xytext=(x_end - box_w / 2, box_h + 0.9),
            ha="center", fontsize=13,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

ax.set_xlim(-0.2, x_end + 0.5)
ax.set_ylim(-1.6, box_h + 1.3)
ax.axis("off")
ax.set_title("왜 앞쪽은 고정한 채 뒤쪽만 다듬나 --- 층별 역할 분화", fontsize=15, fontweight="bold", pad=8)
fig.tight_layout()
fig.savefig(f"{OUT}/layer_freeze_finetune.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------
# D5 (p27g1): CoT probability chain -- bar comparison + geometric gap over steps
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), dpi=200)

ax = axes[0]
labels = ["직접 한 방에\n(0.5$^3$)", "CoT로 단계별\n(0.9$^3$)"]
vals = [0.125, 0.729]
bars = ax.bar(labels, vals, color=[RED, KSA_BLUE], edgecolor="black", linewidth=1.2, width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=15, fontweight="bold")
ax.set_ylim(0, 0.85)
ax.set_ylabel("3단계 추론 전체 성공 확률", fontsize=13)
ax.set_title("47$\\times$83: 단계를 쓰면 0.125 $\\to$ 0.729", fontsize=13.5, fontweight="bold")
ax.tick_params(labelsize=12)
ax.grid(True, axis="y", alpha=0.25)

ax = axes[1]
steps = np.array([2, 5, 10])
direct = 0.5 ** steps
cot = 0.9 ** steps
width = 0.35
x = np.arange(len(steps))
ax.bar(x - width/2, direct, width, label="직접(0.5 per step)", color=RED, edgecolor="black")
ax.bar(x + width/2, cot, width, label="CoT(0.9 per step)", color=KSA_BLUE, edgecolor="black")
for xi, d, c in zip(x, direct, cot):
    ax.text(xi - width/2, d + 0.02, f"{d:.2f}", ha="center", fontsize=10.5)
    ax.text(xi + width/2, c + 0.02, f"{c:.2f}", ha="center", fontsize=10.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{s}단계" for s in steps], fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_ylabel("전체 성공 확률", fontsize=13)
ax.set_title("단계가 늘수록 격차는 기하급수적", fontsize=13.5, fontweight="bold")
ax.legend(fontsize=10.5, loc="upper right")
ax.grid(True, axis="y", alpha=0.25)

fig.suptitle("CoT의 메커니즘: 중간 결과가 조건이 되면 정확도가 기하급수로 벌어진다",
             fontsize=14.5, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/cot_probability_chain.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

print("done: layer_freeze_finetune.png, cot_probability_chain.png")
