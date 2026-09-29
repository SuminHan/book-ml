"""p11: '정보의 전파'로 보는 수렴 -- synchronous sweep 히트맵.
5칸 GridWorld, '오른쪽' 정책, reward -1, gamma=0.9.
sweep마다 목표(칸4)에서 한 칸씩 왼쪽으로 '정확한 값'이 퍼진다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

GAMMA = 0.9
N_STATES = 5  # 0..4, 4=goal
N_SWEEPS = 5

V = np.zeros((N_SWEEPS + 1, N_STATES))
for k in range(1, N_SWEEPS + 1):
    prev = V[k - 1]
    row = np.zeros(N_STATES)
    for s in range(N_STATES - 1):
        row[s] = -1 + GAMMA * prev[s + 1]
    row[N_STATES - 1] = 0.0
    V[k] = row

Vt = vstar()

fig, ax = plt.subplots(figsize=(8.6, 4.6))
data = V[:, :4]  # 칸0~3 (칸4는 항상 0, 목표)
im = ax.imshow(data, cmap='Blues_r', vmin=-4, vmax=0, aspect='auto')

for k in range(N_SWEEPS + 1):
    for s in range(4):
        val = data[k, s]
        converged = abs(val - Vt[s]) < 1e-6
        txt = f"{val:.2f}" if not converged else f"{val:.3f}*"
        color = 'white' if val < -2.0 else 'black'
        weight = 'bold' if converged else 'normal'
        ax.text(s, k, txt, ha='center', va='center', fontsize=11.5,
                color=color, fontweight=weight)

ax.set_xticks(range(4))
ax.set_xticklabels([f"칸 {i}" for i in range(4)], fontsize=13)
ax.set_yticks(range(N_SWEEPS + 1))
ax.set_yticklabels([f"sweep {k}" for k in range(N_SWEEPS + 1)], fontsize=12)
ax.set_title("정보는 sweep마다 목표(칸 4)에서 한 칸씩 왼쪽으로 전파된다\n"
             "(* = 최종값 $V^{*}$ 에 도달, 오른쪽 정책, $\\gamma=0.9$)",
             fontsize=13.5, color=KSA_BLUE_DK, pad=12)
ax.set_xlabel("상태 (칸 4 = 목표, 표시 생략)", fontsize=12.5)

# wavefront 화살표
for k in range(1, N_SWEEPS):
    s_new = 4 - 1 - k  # newly-converged cell index this sweep (0-indexed among 0..3)
    if 0 <= s_new <= 3:
        ax.annotate('', xy=(s_new, k), xytext=(s_new + 1, k - 1),
                    arrowprops=dict(arrowstyle='->', color=KSA_RED, lw=1.8))

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'gw_propagation.png')
savefig(fig, out)
print("saved", out)
