"""ch08_2_baseline_bar.png -- 베이스라인 없이는 "잘 배웠다"를 말할 수 없음
(p33 패턴5 보강): 무작위 정책(~-5000) vs 학습된 정책 greedy 리턴(-13)을
로그 스케일 막대로 비교."""
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

NAVY = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'

labels = ['무작위 정책\n(베이스라인)', '학습된 정책\n(Q-learning, greedy)']
vals = [5000, 13]  # magnitudes of negative returns
colors = [TAN, NAVY]

fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=200)
bars = ax.bar(labels, vals, color=colors, edgecolor='black', lw=0.8, width=0.55)
ax.set_yscale('log')
ax.set_ylim(1, 20000)

for rect, v, raw in zip(bars, vals, [-5000, -13]):
    ax.annotate(f'{raw}', (rect.get_x() + rect.get_width() / 2, v),
                textcoords='offset points', xytext=(0, 8), ha='center',
                fontsize=15, fontweight='bold', color=RED if raw == -5000 else NAVY)

ax.set_ylabel('평균 리턴의 크기 $|G|$ (로그 스케일, CliffWalking)', fontsize=12)
ax.set_title('베이스라인 없이 "$-13$이 나왔다"만으로는 판정 불가', fontsize=13.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', which='both', alpha=0.2)

ax.annotate('', xy=(1, 13), xytext=(0, 5000),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2.2, connectionstyle='arc3,rad=-0.25'))
ax.text(0.5, 700, '$\\approx$385배 개선', fontsize=12.5, color=RED, fontweight='bold', ha='center')

plt.tight_layout(pad=0.4)
out = '/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml2-week08/slides/kor/ml2/week08/figs/ch08_2_baseline_bar.png'
plt.savefig(out, dpi=200, facecolor='white')
print('saved', out)
