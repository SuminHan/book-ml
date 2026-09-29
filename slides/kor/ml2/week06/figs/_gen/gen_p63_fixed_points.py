"""p63g1: same bootstrapping, different fixed points -- SARSA (Q^b) vs Q-learning (Q*)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, RED, GRAY, KSA_SUB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200)

def box(xy, w, h, text, color, fontsize=12.5, fontcolor='white'):
    x, y = xy
    r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=color, edgecolor='black', linewidth=1.2, zorder=3)
    ax.add_patch(r)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=fontcolor,
            zorder=4, linespacing=1.4)

root = (0.5, 0.88)
box(root, 0.62, 0.16, 'TD 오차 $\\delta=0$인 지점\n(같은 부트스트래핑 논리)', GRAY, fontsize=12)

left = (0.24, 0.55)
right = (0.76, 0.55)
box(left, 0.42, 0.16, '목표값의 다음 항\n$Q(s\',a\')$ --- 실제로 고른 $a\'$', KSA_MAIN, fontsize=11.5)
box(right, 0.42, 0.16, '목표값의 다음 항\n$\\max_{a\'}Q(s\',a\')$ --- 최선 가정', RED, fontsize=11.5)

ax.annotate('', xy=(left[0]+0.05, left[1]+0.08), xytext=(root[0]-0.08, root[1]-0.08),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6))
ax.annotate('', xy=(right[0]-0.05, right[1]+0.08), xytext=(root[0]+0.08, root[1]-0.08),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6))

left2 = (0.24, 0.20)
right2 = (0.76, 0.20)
box(left2, 0.46, 0.20, 'SARSA 고정점\n$Q^{b}$ --- 행동정책 $b$의\n벨만 기대방정식', KSA_SUB,
    fontsize=11.5, fontcolor='black')
box(right2, 0.46, 0.20, 'Q-learning 고정점\n$Q^{*}$ --- 벨만 최적방정식\n(이상적 최적 정책)', KSA_SUB,
    fontsize=11.5, fontcolor='black')

ax.annotate('', xy=(left2[0], left2[1]+0.10), xytext=(left[0], left[1]-0.08),
            arrowprops=dict(arrowstyle='-|>', color=KSA_MAIN, lw=1.8))
ax.annotate('', xy=(right2[0], right2[1]+0.10), xytext=(right[0], right[1]-0.08),
            arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.8))

ax.text(0.24, 0.02, '"실제로 이렇게 행동한다면"', ha='center', fontsize=11, style='italic')
ax.text(0.76, 0.02, '"이상적으로 항상 최선을 다한다면"', ha='center', fontsize=11, style='italic')

ax.set_xlim(0, 1); ax.set_ylim(-0.06, 1.0)
ax.axis('off')
ax.set_title('같은 TD 오차 논리, 다른 목표값 $\\to$ 다른 고정점', fontsize=14)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_fixed_points_qstar_qpi.png')
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
