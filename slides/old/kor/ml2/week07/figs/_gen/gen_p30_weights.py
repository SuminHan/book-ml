import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
GOLD = '#A9976B'
RED = '#C0392B'

ns = np.arange(1, 13)
lam = 0.8
weights = (1 - lam) * lam ** (ns - 1)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=200)

ax = axes[0]
sel_n = 5
bars_colors = [RED if n == sel_n else BLUE for n in ns]
ax.bar(ns, [1.0 if n == sel_n else 0 for n in ns], color=bars_colors, width=0.6)
ax.set_xlabel('$n$', fontsize=12)
ax.set_ylabel('가중치', fontsize=12)
ax.set_title('$n$-step TD: 단 하나의 $n$ 선택 (예: $n=5$)', fontsize=12, fontweight='bold', color=BLUE)
ax.set_ylim(0, 1.15)
ax.set_xticks(ns)
ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
ax2.bar(ns, weights, color=GOLD, width=0.6, edgecolor=BLUE)
ax2.set_xlabel('$n$', fontsize=12)
ax2.set_ylabel('가중치 $(1-\\lambda)\\lambda^{n-1}$', fontsize=12)
ax2.set_title('TD($\\lambda=0.8$): 모든 $n$에 기하급수 가중치', fontsize=12, fontweight='bold', color=BLUE)
ax2.set_xticks(ns)
ax2.grid(axis='y', alpha=0.3)
ax2.axvline(1/(1-lam), color=RED, linestyle='--', linewidth=1.5)
ax2.text(1/(1-lam)+0.15, weights.max()*0.85, '유효 수명\n$\\approx 1/(1-\\lambda)=5$', color=RED, fontsize=10)

fig.suptitle('``하나를 고르는 문제''(왼쪽) vs ``분포의 형태를 고르는 문제''(오른쪽)', fontsize=13, y=1.03)
fig.tight_layout()
fig.savefig('/tmp/gen_out_p30.png', facecolor='white', bbox_inches='tight')
print('done')
