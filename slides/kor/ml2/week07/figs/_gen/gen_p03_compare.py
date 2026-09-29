import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
GOLD = '#D6CBB1'
RED = '#C0392B'
GRAY = '#5B5B5B'

fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)

rows = [
    ("TD(0)", [("R_t (실제)", 1, BLUE)], "V(s_{t+1}) (추정치)"),
    ("n-step TD", [("R_t..R_{t+n-1} (실제, n개)", 4, BLUE)], "V(s_{t+n}) (추정치)"),
    ("MC", [("끝까지 전부 실제 보상", 8, BLUE)], None),
]

y_positions = [2, 1, 0]
bar_h = 0.5
total_w = 9

for (label, segs, est), y in zip(rows, y_positions):
    x = 0
    for seg_label, seg_w, color in segs:
        ax.barh(y, seg_w, left=x, height=bar_h, color=color, edgecolor='white')
        ax.text(x + seg_w/2, y, seg_label, ha='center', va='center', color='white',
                 fontsize=10, fontweight='bold')
        x += seg_w
    if est is not None:
        remain = total_w - x
        ax.barh(y, remain, left=x, height=bar_h, color=GOLD, edgecolor='white', hatch='//')
        ax.text(x + remain/2, y, est, ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    ax.text(-0.3, y, label, ha='right', va='center', fontsize=13, fontweight='bold', color=BLUE)

ax.set_xlim(-4.2, total_w + 0.3)
ax.set_ylim(-0.7, 2.7)
ax.axis('off')
ax.set_title('목표값 구성: 실제 보상 몇 개 + 나머지 추정치', fontsize=14, fontweight='bold', color=BLUE)

real_patch = mpatches.Patch(color=BLUE, label='실제 보상 (관측)')
est_patch = mpatches.Patch(color=GOLD, hatch='//', label='추정치 (부트스트래핑)')
ax.legend(handles=[real_patch, est_patch], loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=11, frameon=False)

fig.tight_layout()
fig.savefig('/tmp/gen_out_p03.png', facecolor='white', bbox_inches='tight')
print('done')
