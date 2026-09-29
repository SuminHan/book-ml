"""p03g1: MLP/CNN의 순서 정보 손실 vs RNN의 순차 처리 비교 다이어그램."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# ---- Left panel: MLP/CNN, bag of words (order lost) ----
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('MLP / CNN: 단어를 벡터 하나로 뭉침', fontsize=14, color=KSA_MAIN, pad=10)

words_a = ['강아지가', '고양이를', '쫓는다']
words_b = ['고양이가', '강아지를', '쫓는다']

for i, w in enumerate(words_a):
    box = FancyBboxPatch((0.3, 6.4 - i*1.15), 3.0, 0.9, boxstyle="round,pad=0.08",
                          fc='white', ec=KSA_MAIN, lw=1.6)
    ax.add_patch(box)
    ax.text(1.8, 6.85 - i*1.15, w, ha='center', va='center', fontsize=12)
    ax.add_patch(FancyArrowPatch((3.4, 6.85 - i*1.15), (5.9, 4.2),
                                  arrowstyle='-|>', mutation_scale=12, color=KSA_GRAY, lw=1.2))

for i, w in enumerate(words_b):
    box = FancyBboxPatch((0.3, 2.6 - i*1.15), 3.0, 0.9, boxstyle="round,pad=0.08",
                          fc='white', ec=KSA_MAIN, lw=1.6)
    ax.add_patch(box)
    ax.text(1.8, 3.05 - i*1.15, w, ha='center', va='center', fontsize=12)
    ax.add_patch(FancyArrowPatch((3.4, 3.05 - i*1.15), (5.9, 4.0),
                                  arrowstyle='-|>', mutation_scale=12, color=KSA_GRAY, lw=1.2))

bag = Circle((7.3, 4.1), 1.35, fc=KSA_SUB, ec=KSA_MAIN, lw=1.8)
ax.add_patch(bag)
ax.text(7.3, 4.3, '같은 벡터', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(7.3, 3.75, '(순서 정보 사라짐)', ha='center', va='center', fontsize=9.5, color=KSA_RED)

# ---- Right panel: RNN sequential processing ----
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('RNN: 시간 순서대로 하나씩 처리', fontsize=14, color=KSA_MAIN, pad=10)

steps = ['강아지가', '고양이를', '쫓는다']
xs = [1.6, 5.0, 8.4]
hy = 4.3
for i, (x, w) in enumerate(zip(xs, steps)):
    box = FancyBboxPatch((x-1.05, 5.6), 2.1, 0.85, boxstyle="round,pad=0.08",
                          fc='white', ec=KSA_MAIN, lw=1.6)
    ax.add_patch(box)
    ax.text(x, 6.02, w, ha='center', va='center', fontsize=11.5)
    ax.add_patch(FancyArrowPatch((x, 5.6), (x, hy+0.55), arrowstyle='-|>',
                                  mutation_scale=12, color=KSA_GRAY, lw=1.2))
    circ = Circle((x, hy), 0.55, fc=KSA_SUB, ec=KSA_MAIN, lw=1.8)
    ax.add_patch(circ)
    ax.text(x, hy, f'$h_{i+1}$', ha='center', va='center', fontsize=12, fontweight='bold')
    if i < len(xs) - 1:
        ax.add_patch(FancyArrowPatch((x+0.6, hy), (xs[i+1]-0.6, hy),
                                      arrowstyle='-|>', mutation_scale=14, color=KSA_RED, lw=1.8))
ax.text(xs[0]-1.2, hy, '$h_0$', ha='center', va='center', fontsize=11, color=KSA_GRAY)
ax.add_patch(FancyArrowPatch((xs[0]-1.05, hy), (xs[0]-0.6, hy),
                              arrowstyle='-|>', mutation_scale=12, color=KSA_GRAY, lw=1.2))
ax.annotate('', xy=(9.6, 1.5), xytext=(0.6, 1.5),
            arrowprops=dict(arrowstyle='-|>', color=KSA_MAIN, lw=1.6))
ax.text(5.1, 1.05, '시간의 방향 (과거 $\\to$ 미래, 되돌릴 수 없음)',
        ha='center', va='center', fontsize=10.5, color=KSA_MAIN)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_1_order_matters.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out)
