"""p62g1: RNN의 순차성 --- 100번째 단어를 처리하려면 1~99번째를 순서대로 다 거쳐야 함."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

n = 7
xs = np.linspace(1.0, 11.0, n)
labels = ['$t{=}1$', '$t{=}2$', '$t{=}3$', '$\\cdots$', '$t{=}98$', '$t{=}99$', '$t{=}100$']

for i, (x, lab) in enumerate(zip(xs, labels)):
    box = FancyBboxPatch((x-0.62, 3.5), 1.24, 1.0, boxstyle="round,pad=0.07",
                          fc=KSA_SUB if lab != '$\\cdots$' else 'white',
                          ec=KSA_MAIN if lab != '$\\cdots$' else 'none', lw=1.6)
    if lab != '$\\cdots$':
        ax.add_patch(box)
    ax.text(x, 4.0, lab, ha='center', va='center', fontsize=11)
    if i < n - 1 and labels[i+1] != '$\\cdots$' and lab != '$\\cdots$':
        ax.add_patch(FancyArrowPatch((x+0.62, 4.0), (xs[i+1]-0.62, 4.0),
                                      arrowstyle='-|>', mutation_scale=13, color=KSA_RED, lw=1.8))
    elif lab == '$\\cdots$' or (i+1 < n and labels[i+1] == '$\\cdots$'):
        if i+1 < n:
            ax.add_patch(FancyArrowPatch((x+0.62, 4.0), (xs[i+1]-0.4, 4.0),
                                          arrowstyle='-|>', mutation_scale=13, color=KSA_RED, lw=1.6))

ax.text(6.0, 5.3, '$t{=}100$을 계산하려면 $t{=}1, \\ldots, 99$를 반드시 순서대로 먼저 거쳐야 한다',
        ha='center', fontsize=12, color='black')
ax.text(6.0, 2.6, '$\\Rightarrow$ 병렬화 불가 --- GPU가 아무리 많아도 시점 순서를 건너뛸 수 없음',
        ha='center', fontsize=12, color=KSA_MAIN, fontweight='bold')
ax.text(6.0, 1.6, '(대비: CNN·MLP는 위치/뉴런별로 동시에 계산 가능)',
        ha='center', fontsize=10, color=KSA_GRAY)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_3_sequential.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out)
