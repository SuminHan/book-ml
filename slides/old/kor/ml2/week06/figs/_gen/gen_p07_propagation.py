"""p07g1: information propagation in the 7-state random walk (TD(0))."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import KSA_MAIN, KSA_SUB, RED, GRAY
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), dpi=200)

rows = [
    {"title": "에피소드 1 종료: $3\\to4\\to5\\to6$",
     "known": {3: 0.5, 4: 0.5, 5: 0.5}, "new": None},
    {"title": "에피소드 2: 상태 4를 다시 방문 $\\to 5$",
     "known": {3: 0.5, 4: 0.5, 5: 0.5}, "new": 4, "target": "목표 $=1+V(5)=1.5$"},
    {"title": "에피소드 3: 상태 3을 다시 방문 $\\to 4$",
     "known": {3: 0.5, 4: 1.5, 5: 0.5}, "new": 3, "target": "목표 $=1+V(4)=2.5$"},
]

for ax, row in zip(axes, rows):
    for i in range(7):
        x = i
        if i in (0, 6):
            rect = mpatches.Rectangle((x-0.18, -0.18), 0.36, 0.36,
                                       facecolor=GRAY, edgecolor='black', zorder=5)
            ax.add_patch(rect)
        else:
            color = KSA_MAIN if i == row.get("new") else (KSA_SUB if i in row["known"] else 'white')
            edge = RED if i == row.get("new") else KSA_MAIN
            lw = 3.0 if i == row.get("new") else 1.8
            c = mpatches.Circle((x, 0), 0.18, facecolor=color, edgecolor=edge,
                                 linewidth=lw, zorder=5)
            ax.add_patch(c)
        ax.text(x, 0, str(i), ha='center', va='center', fontsize=12, zorder=6,
                color='white' if i in row["known"] or i in (0, 6) else 'black',
                fontweight='bold')
        if i in row["known"]:
            ax.text(x, 0.42, f"V={row['known'][i]}", ha='center', fontsize=10.5, color=KSA_MAIN)
    ax.plot([0, 6], [0, 0], color=GRAY, lw=1.2, zorder=1)
    if row.get("new") is not None:
        n = row["new"]
        ax.annotate('', xy=(n, 0), xytext=(n+1, 0),
                    arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.2,
                                     connectionstyle='arc3,rad=-0.35'))
        ax.text(n+0.5, -0.55, row["target"], ha='center', fontsize=11, color=RED)
    ax.set_title(row["title"], fontsize=13, loc='left')
    ax.set_xlim(-0.6, 6.6); ax.set_ylim(-0.8, 0.7)
    ax.axis('off')

fig.suptitle('정보 전파: $V(5)$의 갱신이 $V(4)$의 목표값을, $V(4)$가 $V(3)$의 목표값을 바꾼다',
             fontsize=13, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(__file__), '..', 'ch06_td_propagation.png')
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
