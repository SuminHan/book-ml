"""ch08_1_eps_schedule_bar.png -- eps 감쇠 스케줄을 바꿔도 greedy 평가 점수가
그대로임을 보여주는 막대그래프 (p11 보강, needs_figure 힌트)"""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'

schedules = ['고정 $\\varepsilon=0.1$', '선형 감쇠\n$1.0\\to0.01$', '빠른 감쇠\n(200 ep)']
sarsa = [-17, -17, -17]
qlearn = [-13, -13, -13]

x = np.arange(len(schedules))
w = 0.32

fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=200)
b1 = ax.bar(x - w / 2, sarsa, width=w, color=TAN, edgecolor=NAVY, lw=1.2, label='SARSA')
b2 = ax.bar(x + w / 2, qlearn, width=w, color=NAVY, label='Q-learning')

for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f'{h:.0f}', (rect.get_x() + rect.get_width() / 2, h),
                    textcoords='offset points', xytext=(0, -16 if h < 0 else 6),
                    ha='center', fontsize=12, fontweight='bold',
                    color='white' if rect.get_facecolor()[0] < 0.3 else NAVY)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(schedules, fontsize=11.5)
ax.set_ylabel('greedy 평가 리턴 (CliffWalking, 시드 0/1/2 평균)', fontsize=12)
ax.set_title(r'$\varepsilon$ 감쇠 스케줄을 바꿔도 결과는 한 치도 안 변한다', fontsize=13.5)
ax.set_ylim(-22, 2)
ax.legend(fontsize=12, loc='lower right', frameon=True)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.25)

ax.text(1.0, -20.5, '부정적 결과(negative result)도 결과다 --- 세 스케줄 모두 SARSA -17 / Q-learning -13 그대로',
        fontsize=10.5, color=RED, ha='center', fontweight='bold')

plt.tight_layout(pad=0.4)
out = '/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week08/slides/kor/ml2/week08/figs/ch08_1_eps_schedule_bar.png'
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
