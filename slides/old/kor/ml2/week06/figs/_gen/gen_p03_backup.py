"""p03g1: DP vs MC vs TD backup-diagram comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, KSA_SUB, RED, GRAY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), dpi=200)

def state_circle(ax, xy, r=0.09, filled=False, color=KSA_MAIN):
    c = mpatches.Circle(xy, r, facecolor='white' if not filled else color,
                         edgecolor=color, linewidth=2.2, zorder=5)
    ax.add_patch(c)

def terminal_sq(ax, xy, s=0.14, color=GRAY):
    r = mpatches.Rectangle((xy[0]-s/2, xy[1]-s/2), s, s,
                            facecolor=color, edgecolor='black', linewidth=1.2, zorder=5)
    ax.add_patch(r)

# --- Panel 1: DP ---
ax = axes[0]
root = (0.5, 0.92)
state_circle(ax, root, r=0.075)
leaves = [(0.18, 0.55), (0.5, 0.55), (0.82, 0.55)]
for lx in leaves:
    ax.plot([root[0], lx[0]], [root[1], lx[1]], color=GRAY, lw=1.6, zorder=2)
    state_circle(ax, lx, r=0.065, color=GRAY)
ax.text(0.5, 0.30, r'$\mathbb{E}[\,r+\gamma V(s\')\,]$' + '\n(모든 $s\'$ 기댓값,\n모델 $P$ 필요)',
        ha='center', va='top', fontsize=12)
ax.set_title('DP', fontsize=16, color=KSA_MAIN, fontweight='bold')

# --- Panel 2: MC ---
ax = axes[1]
ys = np.linspace(0.92, 0.30, 7)
xs = 0.5 + 0.10*np.sin(np.linspace(0, 2, 7))
for i in range(len(ys)-1):
    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], color=RED, lw=2.0, zorder=2)
for i in range(len(ys)-1):
    state_circle(ax, (xs[i], ys[i]), r=0.06)
terminal_sq(ax, (xs[-1], ys[-1]))
ax.text(0.5, 0.14, r'$G_t$' + ' (실제 리턴,\n에피소드 끝까지 샘플)',
        ha='center', va='top', fontsize=12)
ax.set_title('MC', fontsize=16, color=KSA_MAIN, fontweight='bold')

# --- Panel 3: TD(0) ---
ax = axes[2]
s0 = (0.5, 0.92)
s1 = (0.5, 0.62)
state_circle(ax, s0, r=0.075)
ax.annotate('', xy=s1, xytext=s0,
            arrowprops=dict(arrowstyle='-|>', color=KSA_MAIN, lw=2.2))
state_circle(ax, s1, r=0.065, color=KSA_MAIN)
ax.text(0.62, 0.77, '$r$', fontsize=13, color=RED)
ax.text(0.5, 0.42, r'$r+\gamma V(s\')$' + '\n(한 스텝 샘플 +\n다음 상태 추정치)',
        ha='center', va='top', fontsize=12)
ax.set_title('TD(0)', fontsize=16, color=KSA_MAIN, fontweight='bold')

for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

fig.suptitle('목표값(backup) 구성 비교', fontsize=15, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_backup_dp_mc_td.png')
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
