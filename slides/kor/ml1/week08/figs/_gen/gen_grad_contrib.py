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

x = np.array([1, 2, 3])
y = np.array([0, 1, 1])
phat = np.array([0.524979, 0.549834, 0.574443])
diff = phat - y                    # p̂ - y
contrib = diff * x                 # (p̂-y) x_i
avg = contrib.mean()               # -0.550675

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)

labels = [f"샘플{i+1}\n$x={xi}$" for i, xi in enumerate(x)]

ax = axes[0]
colors = [RED if d > 0 else BLUE for d in diff]
bars = ax.bar(labels, diff, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
ax.axhline(0, color="black", linewidth=1)
for b, v in zip(bars, diff):
    ax.text(b.get_x()+b.get_width()/2, v + (0.02 if v > 0 else -0.05),
             f"{v:+.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=13)
ax.set_title("$\\hat{p}_i - y_i$ (오차 크기)", fontsize=15)
ax.set_ylim(-0.6, 0.65)
ax.tick_params(axis='both', labelsize=12)

ax = axes[1]
colors2 = [RED if c > 0 else BLUE for c in contrib]
bars2 = ax.bar(labels, contrib, color=colors2, width=0.55, edgecolor="black", linewidth=0.8)
ax.axhline(0, color="black", linewidth=1)
ax.axhline(avg, color=BLUE, linewidth=1.6, linestyle="--")
ax.text(-0.42, avg + 0.10, f"평균 = {avg:+.4f}", color=BLUE, fontsize=13, ha="left", va="bottom", fontweight="bold")
for b, v in zip(bars2, contrib):
    ax.text(b.get_x()+b.get_width()/2, v + (0.04 if v > 0 else -0.07),
             f"{v:+.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=13)
ax.set_title("경사 기여 $(\\hat{p}_i-y_i)\\,x_i$", fontsize=15)
ax.set_ylim(-1.5, 0.7)
ax.tick_params(axis='both', labelsize=12)

for ax in axes:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

fig.suptitle("샘플 3($x{=}3$)의 기여가 샘플 1($x{=}1$)보다 $2.4$배 큰 이유", fontsize=16, y=1.03)
fig.tight_layout()
fig.savefig("/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week08/slides/kor/ml1/week08/figs/ch08_3_grad_contrib.png",
            bbox_inches="tight", facecolor="white")
print("saved")
