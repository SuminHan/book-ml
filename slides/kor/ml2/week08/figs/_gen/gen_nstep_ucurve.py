"""ch08_3_nstep_bias_variance.png -- n-step 리턴의 편향-분산 U자 곡선 (p60 보강)"""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'

n = np.linspace(1, 20, 400)
# stylised (qualitative) curves: bias decreases toward 0, variance increases
bias = 3.2 * np.exp(-0.35 * (n - 1))
variance = 0.10 * (n - 1) ** 1.35
total = bias + variance

n_star = n[np.argmin(total)]
y_star = total.min()

fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=200)
ax.plot(n, bias, color=TAN, lw=2.6, label='편향 (bias) --- 부트스트래핑 비중')
ax.plot(n, variance, color='#8C8C8C', lw=2.6, label='분산 (variance) --- 실제 리턴 비중')
ax.plot(n, total, color=NAVY, lw=3.4, label='총 오차 $\\approx$ 편향 + 분산')

ax.plot([n_star], [y_star], 'o', color=RED, ms=9, zorder=6)
ax.annotate(f'최적 $n^*\\approx{n_star:.0f}$', (n_star, y_star),
            textcoords='offset points', xytext=(10, 18), fontsize=12.5,
            color=RED, fontweight='bold')

ax.text(2.0, 0.55, '$n=1$: TD(0)\n(편향 O, 분산 최소)', fontsize=11,
        color='black', linespacing=1.3, ha='left', va='center',
        bbox=dict(fc='white', ec=TAN, lw=1.5, pad=0.4))
ax.text(15.5, 0.85, '$n\\to\\infty$: MC\n(편향 없음, 분산 최대)', fontsize=11,
        color='black', linespacing=1.3, ha='left', va='center',
        bbox=dict(fc='white', ec='#8C8C8C', lw=1.5, pad=0.4))

ax.set_xlabel('$n$ (n-step 리턴의 스텝 수)', fontsize=13)
ax.set_ylabel('오차 (개념도, 정성적 스케일)', fontsize=13)
ax.set_title('n-step 리턴: 편향-분산 트레이드오프의 U자 곡선', fontsize=14)
ax.set_xlim(1, 20)
ax.set_ylim(0, 4.2)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11, loc='upper center', frameon=True)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25)

plt.tight_layout(pad=0.4)
out = '/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week08/slides/kor/ml2/week08/figs/ch08_3_nstep_bias_variance.png'
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
