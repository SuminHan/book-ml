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

metrics = ["정확도", "Precision", "Recall", "F1"]
all_negative = [0.998, 1.0, 0.0, 0.0]      # 모두 "정상" 예측
all_positive = [0.002, 0.002, 1.0, 0.004]  # 모두 "부정" 예측

x = np.arange(len(metrics))
w = 0.35

fig, ax = plt.subplots(figsize=(9.5, 5.3), dpi=200)
b1 = ax.bar(x - w/2, all_negative, width=w, color=BLUE, edgecolor="black", linewidth=0.8, label="모두 '정상' 예측")
b2 = ax.bar(x + w/2, all_positive, width=w, color=RED, edgecolor="black", linewidth=0.8, label="모두 '부정' 예측")

for b, v in zip(b1, all_negative):
    ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}" if v < 1 else "1.000",
             ha="center", va="bottom", fontsize=12)
for b, v in zip(b2, all_positive):
    ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}",
             ha="center", va="bottom", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=14)
ax.set_ylim(0, 1.18)
ax.set_ylabel("값", fontsize=13)
ax.set_title("사기 탐지(99.8:0.2)의 두 극단 분류기", fontsize=16, pad=52)
ax.legend(fontsize=12, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.22))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='y', labelsize=12)

fig.tight_layout()
fig.text(0.5, -0.04,
          "정확도·Precision은 두 극단 모두 높아 보이지만, F1은 0과 0.004로 둘 다 가혹하게 낮게 잡아낸다",
          fontsize=12.5, ha="center", color="#333333")
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_1_pr_extremes.png",
            bbox_inches="tight", facecolor="white")
print("saved")
