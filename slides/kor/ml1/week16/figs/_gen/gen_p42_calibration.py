import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

fig, ax = plt.subplots(figsize=(6.6, 5.2))

ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.6, label='완벽 교정(대각선)')

# bin centers and empirical frequencies (illustrative, consistent with "0.90+ -> 0.980")
bin_centers = np.array([0.15, 0.35, 0.55, 0.75, 0.95])
empirical = np.array([0.18, 0.30, 0.58, 0.71, 0.980])

ax.plot(bin_centers, empirical, 'o-', color=KSA_MAIN, lw=2.2, markersize=9, label='모델의 교정 곡선')

# highlight the 0.90+ bucket point discussed in the deck
ax.scatter([0.95], [0.980], s=160, facecolors='none', edgecolors=KSA_RED, linewidths=2.4, zorder=5)
ax.annotate('"0.90+ 예측" 100건\n실제 비율 0.980', xy=(0.95, 0.980), xytext=(0.42, 0.92),
            fontsize=11.5, color=KSA_RED,
            arrowprops=dict(arrowstyle='->', color=KSA_RED, lw=1.6))

ax.set_xlabel('예측 확률 (모델이 말한 신뢰도)', fontsize=12.5)
ax.set_ylabel('실제 비율 (그 구간에서 맞은 비율)', fontsize=12.5)
ax.set_title('정확도(0.953)와 교정(0.98)은 다른 축', fontsize=13.5, pad=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=11, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_2_calibration_curve.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
