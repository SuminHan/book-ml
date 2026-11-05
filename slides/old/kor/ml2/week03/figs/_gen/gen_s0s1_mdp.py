import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
TAN = "#D6CBB1"
RED = "#C0392B"

fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.2)
ax.axis("off")

s0 = (2.4, 2.7)
s1 = (7.6, 2.7)

for (x, y), name in [(s0, "$S_0$"), (s1, "$S_1$")]:
    ax.add_patch(Circle((x, y), 0.75, facecolor=NAVY, edgecolor=NAVY, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=17, color="white", weight="bold", zorder=4)

# S0 -> S1 via a0 (upper arc)
a = FancyArrowPatch(s0, s1, connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                     mutation_scale=24, linewidth=2.4, color=NAVY, zorder=1,
                     shrinkA=46, shrinkB=46)
ax.add_patch(a)
ax.text(5.0, 4.35, "$a_0$  (p=1/2),  R=1", ha="center", fontsize=12.5, color=NAVY, weight="bold")

# S0 -> S1 via a1 (lower arc)
b = FancyArrowPatch(s0, s1, connectionstyle="arc3,rad=0.05", arrowstyle="-|>",
                     mutation_scale=24, linewidth=2.4, color=NAVY, zorder=1,
                     shrinkA=46, shrinkB=46)
ax.add_patch(b)
ax.text(5.0, 2.95, "$a_1$  (p=1/2),  R=3", ha="center", fontsize=12.5, color=NAVY, weight="bold")

# S1 -> S0 via a0 (bottom arc, returning)
c = FancyArrowPatch(s1, s0, connectionstyle="arc3,rad=-0.55", arrowstyle="-|>",
                     mutation_scale=24, linewidth=2.4, color=RED, zorder=1,
                     shrinkA=46, shrinkB=46)
ax.add_patch(c)
ax.text(5.0, 0.55, "$a_0$  (p=1, 항상),  R=2", ha="center", fontsize=12.5, color=RED, weight="bold")

ax.text(2.4, 1.2, "정책: $a_0$ 1/2, $a_1$ 1/2", ha="center", fontsize=11.5, color="#333333")
ax.text(7.6, 1.2, "정책: 항상 $a_0$", ha="center", fontsize=11.5, color="#333333")

ax.text(5.0, 5.0, "확률적 MDP: $\\gamma=0.9$", ha="center", fontsize=13.5, color=NAVY, weight="bold")

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week03/slides/kor/ml2/week03/figs/ch03_s0s1_mdp.png"
plt.savefig(out, dpi=200, facecolor="white")
print("saved", out)
