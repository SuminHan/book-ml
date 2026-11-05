"""p37g1: SARSA -- fixed epsilon=0.1 vs epsilon-decay learned path shapes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, RED, GRAY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NROWS, NCOLS = 4, 12

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)

for r in range(NROWS):
    for c in range(NCOLS):
        fc = 'black' if (r == 3 and 1 <= c <= 10) else 'white'
        rect = mpatches.Rectangle((c, NROWS-1-r), 1, 1, facecolor=fc,
                                   edgecolor=GRAY, linewidth=0.8, zorder=1)
        ax.add_patch(rect)

ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)
ax.text(11.5, 0.5, 'G', ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)

def cell_xy(r, c):
    return (c+0.5, NROWS-1-r+0.5)

# epsilon=0.1 고정: 절벽 옆 칸 (2,1)을 한 번 스침
fixed_path = [(3,0),(2,0),(2,1),(1,1),(0,1)] + [(0,c) for c in range(2,12)] + [(1,11),(2,11),(3,11)]
fx = [cell_xy(r,c)[0] for r,c in fixed_path]
fy = [cell_xy(r,c)[1] for r,c in fixed_path]
ax.plot(fx, fy, color=RED, lw=2.6, ls='--', marker='o', ms=3.5, zorder=3,
        label=r'$\varepsilon=0.1$ 고정 --- $(2,1)$을 한 번 스침')

# epsilon decay(0.3->0.01): 절벽 옆 칸을 하나도 스치지 않음
decay_path = [(3,0),(2,0),(1,0),(1,1),(0,1)] + [(0,c) for c in range(2,12)] + [(1,11),(2,11),(3,11)]
dx = [cell_xy(r,c)[0] for r,c in decay_path]
dy = [cell_xy(r,c)[1] for r,c in decay_path]
ax.plot(dx, dy, color=KSA_MAIN, lw=2.6, marker='o', ms=3.5, zorder=2,
        label=r'$\varepsilon$ 감쇠 $0.3\to0.01$ --- 절벽 옆 칸 회피')

ax.set_xlim(0, NCOLS); ax.set_ylim(0, NROWS)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=11.5,
          frameon=False)
ax.set_title(r'SARSA: $\varepsilon$ 고정 vs 감쇠 -- 학습된 경로의 모양 차이', fontsize=14)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_sarsa_epsilon_decay_paths.png')
plt.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
print('saved', out)
