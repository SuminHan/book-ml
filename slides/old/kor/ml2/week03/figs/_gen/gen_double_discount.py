import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = "#2E3192"
TAN = "#D6CBB1"
RED = "#C0392B"

gamma = 0.95
N = np.arange(0, 11)

def cum(rate):
    # cumulative sum of rate**k for k=0..n-1 at each n in N
    out = []
    for n in N:
        k = np.arange(0, n)
        out.append(np.sum(rate**k) if n > 0 else 0.0)
    return np.array(out)

undisc = N.astype(float)               # gamma effectively 1
correct = cum(gamma)                   # gamma^k
double = cum(gamma**2)                 # (gamma^2)^k

fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)
ax.plot(N, undisc, "o-", color="#888888", lw=2.4, ms=6, label="할인 없이 더함 (10)")
ax.plot(N, correct, "o-", color=NAVY, lw=2.6, ms=6, label="정확한 할인 $\\gamma=0.95$ (8.03)")
ax.plot(N, double, "o-", color=RED, lw=2.6, ms=6, label="두 번 할인 (6.58)")

for x, y, c in [(10, undisc[10], "#888888"), (10, correct[10], NAVY), (10, double[10], RED)]:
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(8, 2),
                fontsize=12, color=c, weight="bold")

ax.set_xlabel("스텝 수 $N$", fontsize=13)
ax.set_ylabel("누적 리턴 $G_N$", fontsize=13)
ax.set_title("매 스텝 보상 1, 10스텝 에피소드", fontsize=13.5, weight="bold")
ax.tick_params(labelsize=11)
ax.legend(fontsize=11, loc="upper left", frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(0, 10)
fig.tight_layout()

out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week03/slides/kor/ml2/week03/figs/ch03_double_discount.png"
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
