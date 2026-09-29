"""p12g1: tanh 포화로 누적합 추정이 아래로 벗어나는 모습 (교재 수치)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np

t = np.array([1, 2, 3, 4])
true_sum = np.array([2, 5, 6, 10])
est_sum = np.array([1.97, 4.44, 4.62, 6.73])

fig, ax = plt.subplots(figsize=(7.2, 4.6))
w = 0.32
ax.bar(t - w/2, true_sum, width=w, color=KSA_SUB, edgecolor=KSA_MAIN, linewidth=1.3, label='실제 누적합')
ax.bar(t + w/2, est_sum, width=w, color=KSA_MAIN, edgecolor=KSA_MAIN, linewidth=1.3, label='추정치 $\\hat{s}_t = 10 h_t$')

gap_x_off = w/2 + 0.22
for i in range(len(t)):
    ax.text(t[i]-w/2, true_sum[i]+0.28, f'{true_sum[i]:.0f}', ha='center', fontsize=10.5)
    ax.text(t[i]+w/2, est_sum[i]+0.28, f'{est_sum[i]:.2f}', ha='center', fontsize=10.5, color=KSA_MAIN)
    gap = true_sum[i] - est_sum[i]
    if gap > 0.15:
        ax.annotate('', xy=(t[i]+gap_x_off, est_sum[i]+0.05), xytext=(t[i]+gap_x_off, true_sum[i]-0.05),
                     arrowprops=dict(arrowstyle='-', color=KSA_RED, lw=1.4, linestyle=(0, (3, 2))))
        ax.text(t[i]+gap_x_off+0.06, (true_sum[i]+est_sum[i])/2, f'-{gap:.2f}', color=KSA_RED, fontsize=9.5, va='center')

ax.set_xticks(t)
ax.set_xticklabels([f'$t={i}$' for i in t], fontsize=12)
ax.set_ylabel('값', fontsize=12)
ax.set_ylim(0, 11.5)
ax.set_xlim(0.4, 4.75)
ax.set_title('입력이 클수록 $\\tanh$ 포화로 추정이 아래로 벗어남', fontsize=13.5, color=KSA_MAIN)
ax.legend(loc='upper left', fontsize=11, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_1_tanh_saturation.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out)
