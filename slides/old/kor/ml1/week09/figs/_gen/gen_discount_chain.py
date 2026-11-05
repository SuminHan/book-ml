import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

KSA_BLUE = '#2E3192'
RED = '#C0392B'

layers = np.arange(0, 11)
vanish = 0.25 ** layers          # 시그모이드 최댓값 0.25씩 곱
explode = 1.5 ** layers          # 가중치가 커서 1.5배씩 증폭

fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=200)
ax.semilogy(layers, vanish, 'o-', color=KSA_BLUE, lw=2.5, ms=6,
            label='곱셈 인자 0.25 (소실)')
ax.semilogy(layers, explode, 's-', color=RED, lw=2.5, ms=6,
            label='곱셈 인자 1.5 (폭발)')
ax.axhline(1.0, color='grey', lw=1.2, ls='--')
ax.text(0.15, 1.3, '기준선 = 1', fontsize=11, color='grey')

ax.annotate('$0.25^{10}\\approx 9.5\\times10^{-7}$',
            xy=(10, vanish[-1]), xytext=(5.3, 3e-5),
            fontsize=12, color=KSA_BLUE,
            arrowprops=dict(arrowstyle='->', color=KSA_BLUE))
ax.annotate('$1.5^{10}\\approx 57.7$',
            xy=(10, explode[-1]), xytext=(5.3, 12),
            fontsize=12, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED))

ax.set_xlabel('층 번호 (layer)', fontsize=13)
ax.set_ylabel('신호 크기 (곱의 누적, 로그 스케일)', fontsize=13)
ax.set_title('연쇄법칙의 곱 = "할인율"의 누적', fontsize=14)
ax.set_xticks(layers)
ax.legend(fontsize=11, frameon=False, loc='center left')
ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
fig.tight_layout()
fig.savefig('../ch09_2_discount_chain.png', dpi=200, facecolor='white')
print('saved')
