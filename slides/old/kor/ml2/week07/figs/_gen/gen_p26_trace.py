import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
GOLD = '#A9976B'
RED = '#C0392B'

# path: t=0 state3, t=1 state2, t=2 state3, t=3 state4
# lambda=0.5, gamma=1
t = np.array([0, 1, 2, 3])
e3 = np.array([1.0, 0.5, 1.25, 0.625])
e2 = np.array([0.0, 1.0, 0.5, 0.25])
e4 = np.array([0.0, 0.0, 0.0, 1.0])

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=200)

ax = axes[0]
w = 0.25
ax.bar(t - w, e3, width=w, color=BLUE, label='$e(3)$')
ax.bar(t, e2, width=w, color=GOLD, label='$e(2)$')
ax.bar(t + w, e4, width=w, color=RED, label='$e(4)$')
ax.set_xticks(t)
ax.set_xticklabels([f'$t={i}$' for i in t])
ax.set_ylabel('흔적 값 $e(s)$', fontsize=12)
ax.set_title('방문 경로 $3\\to2\\to3\\to4$: 흔적의 변화', fontsize=12.5, fontweight='bold', color=BLUE)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
ax2.plot([0,1,2,3], [3,2,3,4], 'o-', color=BLUE, markersize=14, linewidth=2.2)
for i, s in enumerate([3,2,3,4]):
    ax2.annotate(f'$t={i}$\n상태 {s}', (i, s), textcoords='offset points', xytext=(0, 14),
                 ha='center', fontsize=10)
ax2.set_xlim(-0.5, 3.5)
ax2.set_ylim(1.3, 4.7)
ax2.set_yticks([2,3,4])
ax2.set_xticks([0,1,2,3])
ax2.set_xlabel('스텝 $t$', fontsize=12)
ax2.set_ylabel('상태', fontsize=12)
ax2.set_title('실제 방문 경로', fontsize=12.5, fontweight='bold', color=BLUE)
ax2.grid(alpha=0.3)

fig.suptitle(r'적격흔적 $e(s)\leftarrow\gamma\lambda\,e(s)+\mathbb{1}[s=s_t]$  ($\lambda=0.5$)', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('/tmp/gen_out_p26.png', facecolor='white', bbox_inches='tight')
print('done')
