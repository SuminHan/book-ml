import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

S = np.array([8, 32, 64, 224])
fc = S**2 * 256
conv = np.full_like(S, 320)

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.plot(S, fc, 'o-', color=KSA_RED, linewidth=2.4, markersize=9, label='FC (flatten $\\to$ 256):  $S^2\\times256$')
ax.plot(S, conv, 's-', color=KSA_MAIN, linewidth=2.4, markersize=9, label='3$\\times$3 conv(1$\\to$32): 320 (상수)')
ax.set_yscale('log')
ax.set_xlabel('해상도 $S$ (한 변 픽셀 수)', fontsize=13)
ax.set_ylabel('파라미터 수 (log scale)', fontsize=13)
ax.set_title('"상수 vs 제곱" --- 해상도가 커질수록 벌어지는 격차', fontsize=13.5, pad=12)
ax.legend(fontsize=12, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, which='both', alpha=0.25)

for s, f in zip(S, fc):
    ax.annotate(f'{f:,}', xy=(s, f), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=10.5)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_1_fc_vs_conv_scaling.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
