"""p30g1: CliffWalking grid -- short (13-step) vs safe (17-step) candidate paths."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, RED, GRAY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

NROWS, NCOLS = 4, 12  # row 0 = top, row 3 = bottom

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)

# grid cells
for r in range(NROWS):
    for c in range(NCOLS):
        if r == 3 and 1 <= c <= 10:
            fc = 'black'
        else:
            fc = 'white'
        rect = mpatches.Rectangle((c, NROWS-1-r), 1, 1, facecolor=fc,
                                   edgecolor=GRAY, linewidth=0.8, zorder=1)
        ax.add_patch(rect)

ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)
ax.text(11.5, 0.5, 'G', ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)
ax.text(5.5, 0.35, '절벽 (밟으면 $-100$)', ha='center', va='center', fontsize=10.5,
        color='white', zorder=4)

def cell_xy(r, c):
    return (c+0.5, NROWS-1-r+0.5)

# 짧은 길: (3,0)->(2,0)->(2,1..11)->(3,11)  13 steps
short_path = [(3,0),(2,0)] + [(2,c) for c in range(1,12)] + [(3,11)]
sx = [cell_xy(r,c)[0] for r,c in short_path]
sy = [cell_xy(r,c)[1] for r,c in short_path]
ax.plot(sx, sy, color=RED, lw=2.6, ls='--', marker='o', ms=3.5, zorder=3,
        label='짧은 길 --- 13스텝, 리턴 $-12$')

# 안전한 길: (3,0)->(2,0)->(1,0)->(0,0)->(0,1..11)->(1,11)->(2,11)->(3,11)  17 steps
safe_path = [(3,0),(2,0),(1,0),(0,0)] + [(0,c) for c in range(1,12)] + [(1,11),(2,11),(3,11)]
ax_x = [cell_xy(r,c)[0] for r,c in safe_path]
ax_y = [cell_xy(r,c)[1] for r,c in safe_path]
ax.plot(ax_x, ax_y, color=KSA_MAIN, lw=2.6, marker='o', ms=3.5, zorder=2,
        label='안전한 길 --- 17스텝, 리턴 $-16$')

ax.set_xlim(0, NCOLS)
ax.set_ylim(0, NROWS)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=11.5,
          frameon=False)
ax.set_title('CliffWalking: 두 후보 경로 (짧은 길 vs 안전한 길)', fontsize=14)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_cliff_two_paths.png')
plt.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
print('saved', out)
