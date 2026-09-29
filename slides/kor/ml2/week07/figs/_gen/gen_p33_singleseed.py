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
GRAY = '#5B5B5B'

states = [1, 2, 3, 4, 5]
n1   = [0.00, 0.02, 0.08, 0.16, 0.48]
n5   = [0.05, 0.07, 0.30, 0.44, 0.67]
n50  = [0.12, 0.20, 0.37, 0.56, 0.81]
true = [0.17, 0.33, 0.50, 0.67, 0.83]

fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
ax.plot(states, true, 'o-', color='black', linewidth=2.5, markersize=8, label='참값 $V^\\pi(s)=s/6$')
ax.plot(states, n1, 's--', color=GRAY, linewidth=2, markersize=7, label='$n=1$ (RMS 0.370)')
ax.plot(states, n5, '^--', color=BLUE, linewidth=2, markersize=7, label='$n=5$ (RMS 0.201)')
ax.plot(states, n50, 'D--', color=RED, linewidth=2, markersize=7, label='$n=50$ (RMS 0.099) $\\leftarrow$ 최적')

ax.set_xticks(states)
ax.set_xlabel('상태', fontsize=13)
ax.set_ylabel('$V(s)$ 추정값', fontsize=13)
ax.set_title('단일 실행(시드 0)의 학습 결과: 세 $n$의 곡선', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 0.95)
fig.tight_layout()
fig.savefig('/tmp/gen_out_p33.png', facecolor='white')
print('done')
