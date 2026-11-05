import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = "#2E3192"
GOLD = "#A9976B"
RED = "#C0392B"

c = np.linspace(0, 10, 300)
bias2 = 4.2 * np.exp(-0.55 * c) + 0.15
variance = 0.05 * np.exp(0.45 * c) + 0.05 * c
noise = 0.35
total = bias2 + variance + noise

i_min = np.argmin(total)

fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
ax.plot(c, bias2, color=BLUE, linewidth=2.4, label="편향$^2$ (Ridge $\\lambda\\!\\uparrow$, kNN $k\\!\\uparrow$ 쪽)")
ax.plot(c, variance, color=RED, linewidth=2.4, label="분산 (트리 깊이$\\uparrow$, kNN $k\\!\\downarrow$ 쪽)")
ax.plot(c, total, color="black", linewidth=3.0, label="총 오차 (편향$^2$+분산+잡음)")
ax.axhline(noise, color=GOLD, linewidth=1.6, linestyle=":", label="환원 불가능한 잡음")

ax.axvline(c[i_min], color="#2E8B57", linewidth=1.6, linestyle="--")
ax.scatter([c[i_min]], [total[i_min]], color="#2E8B57", zorder=5, s=70, edgecolor="black")
ax.annotate("최적 지점\n(편향-분산 균형)", xy=(c[i_min], total[i_min]),
             xytext=(c[i_min]+1.3, total[i_min]+1.4), fontsize=12.5, color="#2E8B57",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.4))

ax.text(0.3, bias2[3]+0.35, "왼쪽: 단순한 모델\n(Ridge $\\lambda$ 큼, kNN $k$ 큼)", fontsize=11, color=BLUE)
ax.text(6.5, 0.85, "오른쪽: 유연한 모델\n(무제한 트리, kNN $k{=}1$)", fontsize=11, color=RED,
         ha="left", va="bottom")

ax.set_xlabel("모델 유연성(자유도) $\\longrightarrow$", fontsize=13)
ax.set_ylabel("오차", fontsize=13)
ax.set_title("편향-분산 스펙트럼: 총 오차의 U자 곡선", fontsize=16)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)
ax.legend(fontsize=10.8, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.14))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='both', labelsize=11.5)

fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_3_bias_variance_curve.png",
            bbox_inches="tight", facecolor="white")
print("saved")
