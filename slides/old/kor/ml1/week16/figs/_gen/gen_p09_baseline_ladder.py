import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

fig, ax = plt.subplots(figsize=(7.6, 4.4))

labels = ['1층\n다수 클래스\n(하한)', '2층\nGBDT\n(정형 상한)', '본인 모델\n작은 CNN']
vals = [0.100, 0.944, 0.939]
colors = [KSA_GRAY, KSA_SUB, KSA_MAIN]
x = np.arange(3)
bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor='black', linewidth=0.8, zorder=3)

for xi, v in zip(x, vals):
    ax.text(xi, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=15, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel('test 정확도', fontsize=13)
ax.set_ylim(0, 1.12)
ax.axhline(1.0, color='gray', lw=0.6, ls=':')
ax.set_title('두 층의 베이스라인을 모두 넘어야 "딥러닝이 필요했다"', fontsize=14, pad=12)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.25, zorder=0)

# bracket annotation: CNN vs GBDT gap
ax.annotate('', xy=(2, 0.939), xytext=(1, 0.944),
            arrowprops=dict(arrowstyle='-', color=KSA_RED, lw=1.4, linestyle='--'))
ax.text(1.5, 0.975, '격차 0.006\n(64차원에서는 소폭 열세)', fontsize=11, color=KSA_RED,
        ha='center', va='bottom')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_1_baseline_ladder.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
