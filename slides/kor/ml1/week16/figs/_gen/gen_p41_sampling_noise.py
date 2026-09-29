import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np

rng = np.random.default_rng(0)
# same-model resampled test accuracy over 60 resamples, range approx 0.930-0.990
resamples = rng.normal(loc=0.960, scale=0.014, size=60)
resamples = np.clip(resamples, 0.930, 0.990)

fig, ax = plt.subplots(figsize=(8.2, 3.6))

ax.hist(resamples, bins=14, range=(0.925, 0.995), color=KSA_SUB, edgecolor='white', zorder=2,
        label='같은 모델, test 100건만 60번 재추출')

models = {'GBDT': 0.942, 'MLP': 0.953, 'logreg': 0.959}
colors = {'GBDT': KSA_MAIN, 'MLP': KSA_RED, 'logreg': '#1F7A3D'}
ymax = ax.get_ylim()[1]
for i, (name, acc) in enumerate(models.items()):
    ax.axvline(acc, color=colors[name], lw=2.2, zorder=3)
    ax.text(acc, ymax*(0.86 - 0.14*i), f'{name} {acc:.3f}', color=colors[name],
            fontsize=11.5, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=colors[name], lw=1))

ax.set_xlabel('test 정확도', fontsize=13)
ax.set_ylabel('빈도 (60회 재추출)', fontsize=12)
ax.set_title('모델 간 차이(0.018) < 표본 재추출 폭(약 0.06)', fontsize=13.5, pad=10)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0.92, 1.0)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch16_2_sampling_noise.png')
plt.savefig(out, bbox_inches='tight')
print('saved', out)
