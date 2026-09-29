"""p34: epsilon-decay 그래프. 왼쪽 y축: epsilon (step, 30만 에피소드 0.1 -> 10만 에피소드 0.02).
오른쪽 y축: 정확 정책과의 일치율(104/110 -> 105/110)."""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
RED = '#C0392B'
SAND = '#D6CBB1'

fig, ax1 = plt.subplots(figsize=(9.2, 5.2), dpi=200)

episodes = np.array([0, 300_000, 300_000, 400_000])
eps = np.array([0.1, 0.1, 0.02, 0.02])
ax1.plot(episodes, eps, color=NAVY, lw=3.2, marker='o', markersize=6)
ax1.set_xlabel("누적 학습 에피소드 수", fontsize=12.5)
ax1.set_ylabel(r"$\varepsilon$", fontsize=15, color=NAVY)
ax1.tick_params(axis='y', labelcolor=NAVY, labelsize=11)
ax1.tick_params(axis='x', labelsize=11)
ax1.set_ylim(0, 0.13)
ax1.set_xlim(0, 400_000)
ax1.xaxis.set_major_formatter(lambda x, pos: f"{int(x/1000)}k")
ax1.axvline(300_000, color='gray', lw=1, ls='--', alpha=0.7)
ax1.text(300_000, 0.125, "  30만 에피소드 지점\n  ε: 0.1 → 0.02", fontsize=10, color='gray', va='top')

ax2 = ax1.twinx()
acc_x = [0, 300_000, 300_000, 400_000]
acc_y = [104/110, 104/110, 105/110, 105/110]
ax2.plot(acc_x, acc_y, color=RED, lw=3.2, marker='s', markersize=6, ls='-')
ax2.set_ylabel("정확 정책과의 일치율", fontsize=12.5, color=RED)
ax2.tick_params(axis='y', labelcolor=RED, labelsize=11)
ax2.set_ylim(0.90, 1.0)
ax2.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")

ax2.annotate("104/110 (94.5%)", xy=(150_000, 104/110), xytext=(60_000, 0.965),
             fontsize=10.5, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
ax2.annotate("105/110 (95.5%)", xy=(400_000, 105/110), xytext=(280_000, 0.925),
             fontsize=10.5, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

ax1.set_title(r"$\varepsilon$-decay: 30만 에피소드(ε=0.1) 뒤 10만 에피소드(ε=0.02) 추가 학습",
              fontsize=13, color=NAVY, pad=14)
for spine in ['top']:
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

plt.tight_layout()
out = "/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week05/slides/kor/ml2/week05/figs/ch05_2_epsilon_decay.png"
plt.savefig(out, dpi=200, facecolor='white')
print("saved", out)
