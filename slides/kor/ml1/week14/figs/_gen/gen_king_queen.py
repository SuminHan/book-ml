import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
RED = '#C0392B'

# toy coordinates: x = 성별 축(여성 방향 +), y = 지위 축(왕족 방향 +)
man = np.array([1.0, 0.3])
woman = np.array([1.0, 2.3])   # man shifted along gender axis
king = np.array([4.2, 3.6])
queen = king - man + woman

fig, ax = plt.subplots(figsize=(8, 7), dpi=200)

pts = {'남자': man, '왕': king, '여자': woman, '여왕(예측)': queen}
colors = {'남자': BLUE, '왕': BLUE, '여자': RED, '여왕(예측)': RED}
for name, p in pts.items():
    ax.scatter(*p, s=90, color=colors[name], zorder=5)
    dx = 0.15 if name != '여왕(예측)' else 0.2
    ax.annotate(name, p, xytext=(p[0]+dx, p[1]+0.1), fontsize=14, color=colors[name])

# man -> king (status axis shift)
ax.annotate('', xy=king, xytext=man, arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.4))
ax.text((man[0]+king[0])/2 - 0.55, (man[1]+king[1])/2, '지위 축\n(남자$\\to$왕)', fontsize=11, color=BLUE)

# woman -> queen (parallel, same status shift)
ax.annotate('', xy=queen, xytext=woman, arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.4, linestyle='--'))
ax.text((woman[0]+queen[0])/2 + 0.15, (woman[1]+queen[1])/2 - 0.35, '지위 축\n(여자$\\to$여왕)', fontsize=11, color=RED)

# man -> woman (gender axis)
ax.annotate('', xy=woman, xytext=man, arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.8))
ax.text(man[0]-0.9, (man[1]+woman[1])/2, '성별 축\n(남자$\\to$여자)', fontsize=11, color='gray')

# king -> queen (gender axis, parallel)
ax.annotate('', xy=queen, xytext=king, arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.8, linestyle='--'))

# dashed parallelogram outline
for a, b in [(man, king), (king, queen), (queen, woman), (woman, man)]:
    ax.plot([a[0], b[0]], [a[1], b[1]], color='lightgray', lw=1, zorder=1)

ax.text(0.5, 6.0, '왕 $-$ 남자 $+$ 여자 $\\approx$ 여왕', fontsize=17, color='black',
        bbox=dict(boxstyle='round', fc='#F4F1EA', ec=BLUE))

ax.set_xlabel('성별 축 (남자 $\\to$ 여자)', fontsize=12)
ax.set_ylabel('지위 축 (평민 $\\to$ 왕족)', fontsize=12)
ax.set_xlim(-0.3, 5.3)
ax.set_ylim(-0.3, 6.6)
ax.set_title('"왕 $-$ 남자 $+$ 여자 = 여왕"의 병렬 4각형 구조', fontsize=15, color=BLUE, pad=12)
plt.tight_layout()
plt.savefig('/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week14/slides/kor/ml1/week14/figs/ch14_king_queen.png',
            bbox_inches='tight', facecolor='white')
print("done")
