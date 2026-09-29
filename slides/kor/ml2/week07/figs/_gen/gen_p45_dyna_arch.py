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

fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=200)

def box(x, y, w, h, text, color, textcolor='white', fontsize=12):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                     linewidth=1.5, edgecolor=BLUE, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold')

def arrow(p1, p2, color=RED, rad=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=18, color=color,
                          linewidth=2, connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)

box(0.5, 3.3, 2.2, 1.0, '실제 환경\n(experience)', BLUE)
box(3.9, 3.3, 2.2, 1.0, '모델\n(model)', GOLD, textcolor='black')
box(3.9, 0.9, 2.2, 1.0, 'Q / 정책\n(value/policy)', BLUE)
box(0.5, 0.9, 2.2, 1.0, '행동 선택\n($\\varepsilon$-greedy)', GOLD, textcolor='black')

arrow((2.7, 3.8), (3.9, 3.8))
ax.text(3.3, 4.05, '모델 학습', ha='center', fontsize=10.5, color=RED, fontweight='bold')
ax.text(3.3, 3.55, '(model learning)', ha='center', fontsize=9, color=RED)

arrow((1.6, 3.3), (1.6, 1.9))
ax.text(1.6, 2.75, '직접 강화학습', ha='center', fontsize=10.5, color=RED, fontweight='bold')
ax.text(1.6, 2.5, '(direct RL, TD(0))', ha='center', fontsize=9, color=RED)

arrow((3.9, 1.4), (2.7, 1.4), color=BLUE)
ax.text(3.3, 1.62, '행동 결정', ha='center', fontsize=10.5, color=BLUE, fontweight='bold')

arrow((0.5, 1.4), (0.5, 3.3), rad=-0.4)
ax.text(-0.55, 2.35, '실제\n경험', ha='center', fontsize=10.5, color=RED, fontweight='bold')

arrow((5.0, 3.3), (5.0, 1.9), color=RED)
ax.text(5.75, 2.75, '계획', ha='center', fontsize=10.5, color=RED, fontweight='bold')
ax.text(5.75, 2.5, '(planning,\n가상 TD(0))', ha='center', fontsize=9, color=RED)

ax.set_xlim(-1.3, 7.3)
ax.set_ylim(0.5, 4.8)
ax.axis('off')
ax.set_title('Dyna 아키텍처: 학습과 계획이 하나의 루프로', fontsize=14, fontweight='bold', color=BLUE)
fig.tight_layout()
fig.savefig('/tmp/gen_out_p45.png', facecolor='white', bbox_inches='tight')
print('done')
