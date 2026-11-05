"""p26: 정책 반복의 단조성 -- 개선은 절대 가치를 나쁘게 만들지 않는다.
이 장의 예제(5칸 GridWorld)에서 '왼쪽' 정책(outer iter 0)이
'오른쪽'(=최적) 정책(outer iter 1)으로 개선되며 모든 칸의 V가 증가.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

Vt = vstar()
V_left = [-10.0, -10.0, -10.0, -10.0]
V_right = Vt[:4]

states = np.arange(4)
w = 0.35
fig, ax = plt.subplots(figsize=(7.6, 4.6))
b1 = ax.bar(states - w / 2, V_left, width=w, color=KSA_GOLD_DK, label='outer iter 0 ("왼쪽" 정책)')
b2 = ax.bar(states + w / 2, V_right, width=w, color=KSA_BLUE, label='outer iter 1 ("오른쪽" = 최적)')

for x, v in zip(states - w / 2, V_left):
    ax.text(x, v - 0.35, f"{v:.1f}", ha='center', fontsize=11)
for x, v in zip(states + w / 2, V_right):
    ax.text(x, v + 0.15, f"{v:.3f}", ha='center', fontsize=11)

for s in states:
    ax.annotate('', xy=(s + w / 2, V_right[s] + 0.05), xytext=(s - w / 2, V_left[s] + 0.05),
                arrowprops=dict(arrowstyle='->', color=KSA_RED, lw=1.6))

ax.set_xticks(states)
ax.set_xticklabels([f"칸 {i}" for i in range(4)], fontsize=12.5)
ax.set_ylabel("$V^\\pi(s)$", fontsize=13)
ax.set_ylim(-11, 0.8)
ax.axhline(0, color='gray', lw=0.6)
ax.set_title("정책 개선은 모든 칸에서 가치를 절대 나쁘게 만들지 않는다\n"
             "(이 예에서는 outer iteration 2번 만에 최적에 도달)",
             fontsize=13, color=KSA_BLUE_DK)
ax.legend(fontsize=10.5, loc='upper left')
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'policy_improvement_monotonic.png')
savefig(fig, out)
print("saved", out)
