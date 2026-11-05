"""p28g1: BPTT 재귀 구조 - 본 시점 오차(A) + 다음 시점에서 흘러온 그래디언트(B)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7.2)
ax.axis('off')

xs = [1.8, 5.0, 8.2, 11.0]
labels = ['$h_1$', '$h_2$', '$h_3$', '$h_4$']
y = 4.6

# forward chain (top, gray, forward direction)
for i in range(len(xs)-1):
    ax.add_patch(FancyArrowPatch((xs[i]+0.55, y+0.15), (xs[i+1]-0.55, y+0.15),
                                  arrowstyle='-|>', mutation_scale=13, color=KSA_GRAY, lw=1.3))
ax.text(6.4, y+0.75, '순전파 (시간 순서대로)', fontsize=10.5, color=KSA_GRAY, ha='center')

for x, lab in zip(xs, labels):
    ax.add_patch(Circle((x, y), 0.55, fc=KSA_SUB, ec=KSA_MAIN, lw=1.8))
    ax.text(x, y, lab, ha='center', va='center', fontsize=13, fontweight='bold')

# output error boxes (A term) below each h
for i, x in enumerate(xs):
    box = FancyBboxPatch((x-0.85, 1.1), 1.7, 0.85, boxstyle="round,pad=0.06",
                          fc='white', ec=KSA_RED, lw=1.4)
    ax.add_patch(box)
    ax.text(x, 1.52, f'$A_{i+1}$: 시점 출력 오차', ha='center', va='center', fontsize=8.7, color=KSA_RED)
    ax.add_patch(FancyArrowPatch((x, 1.95), (x, y-0.6), arrowstyle='-|>',
                                  mutation_scale=11, color=KSA_RED, lw=1.3))

# backward chain (B term), red, right to left
for i in range(len(xs)-1, 0, -1):
    ax.add_patch(FancyArrowPatch((xs[i]-0.6, y-0.25), (xs[i-1]+0.6, y-0.25),
                                  arrowstyle='-|>', mutation_scale=15, color=KSA_MAIN, lw=2.1))
    ax.text((xs[i]+xs[i-1])/2, y-0.85, '$B$ (누적된 그래디언트)', ha='center',
            fontsize=9, color=KSA_MAIN)

ax.text(6.4, 6.5, r'$\dfrac{\partial L}{\partial h_t} = A_t + B_t$'
        '   (' + '$A_t$' + ' = 본 시점 오차, ' + '$B_t$' + ' = 다음 시점에서 흘러온 것)',
        fontsize=13.5, ha='center', color='black')
ax.text(6.4, 0.35, '역전파: 오른쪽(미래)에서 왼쪽(과거)로, $W_{hh}$가 공유되므로 같은 텐서에 $+=$로 누적',
        fontsize=10, ha='center', color=KSA_GRAY)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch11_2_bptt_recursion.png')
plt.savefig(out, dpi=DPI, bbox_inches='tight')
print('saved', out)
