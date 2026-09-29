"""p48g1: 세 승수(0.45/0.75/1.5)에 따른 그래디언트 곱적의 운명 (교재 표 재현)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np

T = np.arange(2, 51)
m1, m2, m3 = 0.45, 0.75, 1.5
g1 = m1 ** (T - 1)
g2 = m2 ** (T - 1)
g3 = m3 ** (T - 1)

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.semilogy(T, g1, color=KSA_MAIN, lw=2.4, label='승수 0.45 (소실) --- $\\tanh\'{=}0.5,\\,w_{hh}{=}0.9$')
ax.semilogy(T, g2, color='#9a8258', lw=2.4, label='승수 0.75 (느린 소실) --- $w_{hh}{=}1.5$, 포화')
ax.semilogy(T, g3, color=KSA_RED, lw=2.4, label='승수 1.5 (폭발) --- $w_{hh}{=}1.5$, 미포화')

for m, col, marker_T in [(m1, KSA_MAIN, 20), (m2, '#9a8258', 20), (m3, KSA_RED, 20)]:
    y = m ** (marker_T - 1)
    ax.scatter([marker_T], [y], color=col, zorder=5, s=45)

ax.axhline(1.0, color='gray', lw=1.0, linestyle=':')
ax.text(45, 1.3, '기준선 = 1', fontsize=9, color='gray')

ax.set_xlabel('시퀀스 길이 $T$', fontsize=12)
ax.set_ylabel(r'$\prod \tanh\'(z_t)\, w_{hh}$ (로그 스케일)', fontsize=12)
ax.set_title('승수 1을 살짝 넘고 못 넘음에 따라 갈리는 운명', fontsize=13.5, color=KSA_MAIN)
ax.legend(loc='center left', fontsize=9.7, frameon=True, facecolor='white',
          edgecolor='none', framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_3_gradient_fate.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out)
