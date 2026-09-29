"""ch08_2_rubric_donut.png -- 채점 기준 비중 도넛 차트 (p18 보강):
정직성 관련 두 항목(25%+20%=45%)이 전체의 거의 절반임을 시각화."""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'
GRAY = '#8C8C8C'
LNAVY = '#7A7DC7'

labels = ['문제 정의와\n환경\n15%', '방법론\n적절성\n20%',
          '실험 프로토콜의\n정직성\n25%', '수식적\n정당화\n20%',
          '결과의 정직성과\n반성\n20%']
sizes = [15, 20, 25, 20, 20]
colors = [GRAY, TAN, RED, LNAVY, NAVY]
explode = [0, 0, 0.06, 0, 0.06]

fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
wedges, texts = ax.pie(sizes, colors=colors, startangle=90, explode=explode,
                        wedgeprops=dict(width=0.42, edgecolor='white', linewidth=2))

for i, (w, lab) in enumerate(zip(wedges, labels)):
    ang = (w.theta2 + w.theta1) / 2.0
    x = 1.28 * np.cos(np.radians(ang))
    y = 1.28 * np.sin(np.radians(ang))
    ha = 'left' if x >= 0 else 'right'
    fw = 'bold' if i in (2, 4) else 'normal'
    col = RED if i == 2 else (NAVY if i == 4 else 'black')
    ax.annotate(lab, (np.cos(np.radians(ang)) * 0.79, np.sin(np.radians(ang)) * 0.79),
                xytext=(x, y), ha=ha, va='center', fontsize=11.5, fontweight=fw, color=col)

ax.text(0, 0, '정직성\n45%', ha='center', va='center', fontsize=17, fontweight='bold', color=RED)
ax.set_title('채점 기준 5항목 --- "정직성" 관련 두 항목이 45%', fontsize=14)
ax.set_aspect('equal')

plt.tight_layout(pad=0.6)
out = '/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week08/slides/kor/ml2/week08/figs/ch08_2_rubric_donut.png'
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
