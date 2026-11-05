"""p43g1: 잘라내기 BPTT --- 먼 과거 시점의 그래디언트 기여가 이미 작아 손실이 작다는 것을
개념적으로 보여주는 감쇠 막대그래프 (w_hh=0.9 가정, 11.1절과 같은 감쇠 상수)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np

k = np.arange(0, 11)  # k 시점 전
w = 0.9
contrib = w ** k
window = 4  # 잘라내기 창 길이

kept = contrib[k < window].sum()
total = contrib.sum()
pct_kept = kept / total * 100

fig, ax = plt.subplots(figsize=(8.4, 5.2))
colors = [KSA_MAIN if kk < window else KSA_GRAY for kk in k]
bars = ax.bar(k, contrib, color=colors, edgecolor='white', width=0.72, zorder=3)

ax.axvline(window - 0.5, color=KSA_RED, lw=1.8, linestyle='--', zorder=2)
ax.text(window - 0.4, 1.28, '창 경계\n(길이 4)', color=KSA_RED, fontsize=10)

ax.text(1.5, 1.28, f'창 안 (최근 4시점)\n크기 합 {pct_kept:.0f}%', color=KSA_MAIN,
        fontsize=10.5, ha='center', fontweight='bold')
ax.text(7.5, 1.28, f'창 밖 (먼 과거)\n크기 합 {100-pct_kept:.0f}%',
        color=KSA_GRAY, fontsize=10.5, ha='center')

ax.set_xlabel('몇 시점 전인가 ($k$)', fontsize=12)
ax.set_ylabel(r'그래디언트 기여 $\propto w_{hh}^{\,k}$', fontsize=12)
ax.set_title('먼 과거일수록 기여가 지수적으로 작아 잘라내기 손실이 작다 (개념도, $w_{hh}=0.9$)',
             fontsize=12.3, color=KSA_MAIN)
ax.set_xticks(k)
ax.set_ylim(0, 1.55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_2_truncated_bptt.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out, f'{pct_kept:.1f}% kept')
