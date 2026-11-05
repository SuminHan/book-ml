import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2E3192"
GOLD = "#D6CBB1"
RED = "#C0392B"

labels = [
    "로지스틱회귀\ntest AUC",
    "GDA/LDA\ntest AUC",
    "출력 확률\n상관계수",
    "test 라벨\n일치율",
    "경계 벡터\n$\\cos(w_{logit}, w_{GDA})$",
]
values = [0.9934, 0.9855, 0.900, 0.936, 0.203]
colors = [BLUE, BLUE, "#2E8B57", "#2E8B57", RED]

y = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9.3, 5.3), dpi=200)
bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.8, height=0.6)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12.5)
ax.invert_yaxis()
ax.set_xlim(0, 1.12)
for b, v in zip(bars, values):
    ax.text(v + 0.02, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=13, fontweight="bold")

ax.axvline(0.9, color="black", linewidth=0.6, linestyle=":")
ax.set_title("breast_cancer 실측: 경계·라벨은 같고, $w$ 방향은 다르다", fontsize=14.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='x', labelsize=11.5)

ax.annotate("형태(경계·확률)는 거의 같음", xy=(0.97, 3.3), xytext=(0.6, 3.8),
             fontsize=11.5, color="#2E8B57", ha="center",
             arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.3))
ax.annotate("계수 방향은 다름\n(정규분포 전제 약하게 깨짐)", xy=(0.10, 3.78), xytext=(0.55, 4.6),
             fontsize=11.5, color=RED, ha="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
ax.set_ylim(4.9, -0.6)

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_3_gda_logistic_bars.png",
            bbox_inches="tight", facecolor="white")
print("saved")
