"""p42: 지평선이 한 칸씩 넓어지고, k=4에서 고정.
k=1,2,3,4 각각의 V_k(s)를 막대그래프 small-multiples로 시각화.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

Vt = vstar()  # [-3.439, -2.710, -1.900, -1.000, 0.0]

V = np.zeros((6, N))
for k in range(1, 6):
    prev = V[k - 1]
    row = np.zeros(N)
    for s in range(N - 1):
        row[s] = -1 + GAMMA * prev[s + 1]
    row[N - 1] = 0.0
    V[k] = row

ks = [1, 2, 3, 4]
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), sharey=True)
states = np.arange(N)
labels = ["칸0", "칸1", "칸2", "칸3", "칸4\n(목표)"]

for ax, k in zip(axes, ks):
    vals = V[k]
    colors = [KSA_RED if abs(vals[s] - Vt[s]) < 1e-6 and s < 4 else KSA_BLUE for s in range(N)]
    colors[4] = KSA_GOLD_DK
    ax.bar(states, vals, color=colors, width=0.62)
    for s in range(N):
        ax.text(s, vals[s] - 0.15, f"{vals[s]:.2f}", ha='center', va='top', fontsize=10.5)
    ax.set_xticks(states)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_title(f"$k={k}$", fontsize=14, color=KSA_BLUE_DK)
    ax.set_ylim(-4.2, 0.6)
    ax.axhline(0, color='gray', lw=0.6)
    # 지평선 경계: 도달 못한 칸(목표까지 k보다 먼 칸)은 옅게
    horizon = 4 - k  # 이 칸 이상 왼쪽(작은 번호)은 아직 목표에 못 닿음
    if horizon > 0:
        ax.axvspan(-0.5, horizon - 0.5, color=KSA_GOLD, alpha=0.28)
    if 0 <= 0:
        pass

axes[0].set_ylabel("$V_k(s)$", fontsize=13)
fig.suptitle("지평선(horizon)이 한 칸씩 넓어지다 $k=4$ 에서 고정 (빨강 = $V^{*}$ 와 일치)",
             fontsize=14, color=KSA_BLUE_DK, y=1.05)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'vi_horizon_widening.png')
savefig(fig, out)
print("saved", out)
