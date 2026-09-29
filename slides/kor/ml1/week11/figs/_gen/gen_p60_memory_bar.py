"""p60g1: 기억 과제(T=100) 최종 정확도 --- 모델별 평균 + 3개 seed 산점 (교재 표 수치)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
import numpy as np

models = ['기본 RNN', 'GRU', 'LSTM']
seeds = {
    '기본 RNN': [0.87, 0.91, 0.89],
    'GRU': [1.00, 1.00, 0.98],
    'LSTM': [0.61, 1.00, 0.96],
}
avgs = [np.mean(seeds[m]) for m in models]
colors = [KSA_GRAY, KSA_MAIN, KSA_RED]

fig, ax = plt.subplots(figsize=(6.6, 4.8))
x = np.arange(len(models))
bars = ax.bar(x, avgs, color=colors, width=0.55, alpha=0.85, edgecolor='black', linewidth=0.6)

for i, m in enumerate(models):
    jitter = np.linspace(-0.12, 0.12, 3)
    ax.scatter(x[i] + jitter, seeds[m], color='black', zorder=5, s=38, marker='o')
    ax.text(x[i], avgs[i] + 0.025, f'평균 {avgs[i]:.2f}', ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('\n'.join('최종정확도'), fontsize=12, labelpad=12, rotation=0, ha='center', va='center')
ax.set_ylim(0.5, 1.08)
ax.axhline(1.0, color='gray', lw=0.8, linestyle=':')
ax.set_title('기억 과제 ($T{=}100$): 검은 점 = 개별 seed, 막대 = 3-seed 평균', fontsize=12.5, color=KSA_MAIN)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_3_memory_bar.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out)
