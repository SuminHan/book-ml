import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

fig, ax = plt.subplots(figsize=(7.6, 4.6))

groups = ['8$\\times$8\n(64차원)', '28$\\times$28\n(784차원)']
gbdt = [0.944, 0.944]
cnn = [0.939, 0.961]

x = np.arange(2)
w = 0.32
b1 = ax.bar(x - w/2, gbdt, width=w, color=KSA_SUB, edgecolor='black', linewidth=0.8, label='GBDT', zorder=3)
b2 = ax.bar(x + w/2, cnn, width=w, color=KSA_MAIN, edgecolor='black', linewidth=0.8, label='CNN', zorder=3)

for xi, v in zip(x - w/2, gbdt):
    ax.text(xi, v + 0.004, f'{v:.3f}', ha='center', va='bottom', fontsize=13)
for xi, v in zip(x + w/2, cnn):
    ax.text(xi, v + 0.004, f'{v:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=14)
ax.set_ylabel('test 정확도', fontsize=13)
ax.set_ylim(0.90, 1.0)
ax.set_title('같은 이미지, 해상도만 바꾸자 우위가 역전', fontsize=14, pad=12)
ax.legend(fontsize=12, loc='lower left')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.25, zorder=0)

ax.annotate('', xy=(0, 0.958), xytext=(1, 0.958),
            arrowprops=dict(arrowstyle='->', color=KSA_RED, lw=1.6))
ax.text(0.5, 0.965, '격차 $-$0.006 $\\to$ $+$0.017', fontsize=12, color=KSA_RED, ha='center')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_1_8x8_vs_28x28.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
