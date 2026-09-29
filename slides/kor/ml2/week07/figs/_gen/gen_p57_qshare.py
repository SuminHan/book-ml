import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
GOLD = '#D6CBB1'
RED = '#C0392B'
GREEN = '#1E7A3D'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=200)

def box(ax, x, y, w, h, text, color, textcolor='white', fontsize=11):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                     linewidth=1.5, edgecolor=BLUE, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold')

def arrow(ax, p1, p2, color, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=16, color=color, linewidth=2,
                          connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)

ax = axes[0]
box(ax, 0.3, 2.6, 2.0, 0.9, '실제 경험\n갱신', BLUE)
box(ax, 0.3, 0.3, 2.0, 0.9, '계획(planning)\n갱신', RED)
box(ax, 3.1, 1.45, 2.1, 0.9, '공유 $Q$\n테이블', GOLD, textcolor='black', fontsize=13)
arrow(ax, (2.3, 3.0), (3.1, 2.1), BLUE)
arrow(ax, (2.3, 0.75), (3.1, 1.65), RED)
ax.set_xlim(-0.3, 5.6)
ax.set_ylim(0, 3.9)
ax.axis('off')
ax.set_title('올바름: 하나의 $Q$를 공유', fontsize=13, fontweight='bold', color=GREEN)
ax.text(2.7, 3.55, 'OK', fontsize=16, color=GREEN, ha='center', fontweight='bold')

ax2 = axes[1]
box(ax2, 0.3, 2.6, 2.0, 0.9, '실제 경험\n갱신', BLUE)
box(ax2, 0.3, 0.3, 2.0, 0.9, '계획(planning)\n갱신', RED)
box(ax2, 3.1, 2.6, 2.3, 0.9, 'Q_real', BLUE, fontsize=13)
box(ax2, 3.1, 0.3, 2.3, 0.9, 'Q_plan\n(반영 안 됨)', RED, fontsize=11)
arrow(ax2, (2.3, 3.05), (3.1, 3.05), BLUE)
arrow(ax2, (2.3, 0.75), (3.1, 0.75), RED)
ax2.set_xlim(-0.3, 5.6)
ax2.set_ylim(0, 3.9)
ax2.axis('off')
ax2.set_title('실수: 두 $Q$로 분리', fontsize=13, fontweight='bold', color=RED)
ax2.text(2.7, 3.55, 'X', fontsize=18, color=RED, ha='center', fontweight='bold')

fig.tight_layout()
fig.savefig('/tmp/gen_out_p57.png', facecolor='white', bbox_inches='tight')
print('done')
