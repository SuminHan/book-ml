import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
GOLD = '#A9976B'
RED = '#C0392B'

n = 3
ts = [3, 4, 5, 6]
taus = [t - n + 1 for t in ts]  # 1,2,3,4

fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)

ax.plot(ts, [1]*len(ts), 'o-', color=BLUE, markersize=12, linewidth=2, label='현재 시점 $t$')
ax.plot(ts, [0]*len(ts), 's-', color=RED, markersize=12, linewidth=2, label='갱신 대상 $\\tau = t-n+1$')

for t, tau in zip(ts, taus):
    ax.annotate('', xy=(t, 0.12), xytext=(t, 0.88),
                arrowprops=dict(arrowstyle='-', color='gray', lw=1, linestyle=':'))
    ax.text(t, 1.18, f'$t={t}$', ha='center', fontsize=11)
    ax.text(t, -0.32, f'$\\tau={tau}$', ha='center', fontsize=11, color=RED, fontweight='bold')

ax.annotate('', xy=(ts[0]-0.15, 0), xytext=(ts[-1]+0.15, 0),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.6))
ax.text((ts[0]+ts[-1])/2, -0.62, r'$\tau$도 함께 한 칸씩 앞으로 (지연폭 $n-1=2$ 스텝, 고정)', ha='center', fontsize=11, color=RED)

ax.set_xlim(ts[0]-0.6, ts[-1]+0.6)
ax.set_ylim(-0.9, 1.5)
ax.axis('off')
ax.set_title(f'갱신 시차 (예: $n={n}$) --- $t$가 늘면 $\\tau$도 같이 밀려남', fontsize=13, fontweight='bold', color=BLUE)
ax.legend(loc='upper right', fontsize=10, frameon=False)

fig.tight_layout()
fig.savefig('/tmp/gen_out_p09.png', facecolor='white', bbox_inches='tight')
print('done')
