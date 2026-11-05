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

m = np.array([1, 5, 50, 100])
inflation = np.array([0.0, 0.023, 0.045, 0.050])

fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
ax.plot(m, inflation, marker="o", markersize=9, color=BLUE, linewidth=2.4,
        markerfacecolor=RED, markeredgecolor="black", markeredgewidth=1.0)
ax.set_xscale("log")
ax.set_xticks(m)
ax.set_xticklabels([str(v) for v in m], fontsize=13)
ax.set_xlabel("후보 수 $m$ (모델 $\\times$ 하이퍼파라미터)", fontsize=13)
ax.set_ylabel("test 성능 부풀림 $\\mathbb{E}[\\max]$", fontsize=13)
ax.set_title("``살짝 훔쳐보기''가 부풀리는 크기", fontsize=16)

for xi, yi in zip(m, inflation):
    ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=12.5, fontweight="bold", color=BLUE)

ax.axvspan(40, 60, color=GOLD, alpha=0.4, zorder=0)

ax.set_ylim(-0.005, 0.062)
ax.tick_params(axis='y', labelsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.tight_layout()
fig.text(0.5, -0.03, "음영: 본 프로젝트 규모(5모델 $\\times$ 10후보 $\\approx$ 50) $\\to$ 부풀림 약 0.045",
          fontsize=11.5, ha="center", color="#A9976B", fontweight="bold")
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_2_inflation_curve.png",
            bbox_inches="tight", facecolor="white")
print("saved")
