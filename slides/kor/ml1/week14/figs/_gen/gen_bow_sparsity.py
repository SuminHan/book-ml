import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Rectangle

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

BLUE = '#2E3192'
TAN = '#D6CBB1'
RED = '#C0392B'

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=200,
                          gridspec_kw={'width_ratios': [1.5, 1]})

# panel a: sparse one-hot vectors for "고양이" and "강아지" and "사과" over V=100000 dims (show a slice)
ax = axes[0]
V = 60  # visual slice representing "어휘 10만 차원 중 일부"
rng = np.random.default_rng(3)
idx_cat = 12
idx_dog = 37
idx_apple = 50

words = ['고양이(idx=12)', '강아지(idx=37)', '사과(idx=50)']
idxs = [idx_cat, idx_dog, idx_apple]
for row, (w, idx) in enumerate(zip(words, idxs)):
    vec = np.zeros(V)
    vec[idx] = 1
    y0 = row
    ax.bar(np.arange(V), vec * 0.8, bottom=y0, width=1.0, color=BLUE, edgecolor='none')
    ax.text(-3, y0 + 0.3, w, ha='right', va='center', fontsize=11)
    ax.axhline(y0, color='gray', lw=0.4)

ax.set_xlim(-16, V)
ax.set_ylim(-0.3, 3.1)
ax.set_yticks([])
ax.set_xlabel('어휘 차원 (10만 차원 중 60개 슬라이스, 99.99%가 0)', fontsize=11)
ax.set_title('(a) one-hot / bag-of-words: 극도로 희소한 벡터', fontsize=13)

# panel b: orthogonality — dot product = 0 illustration with 2D toy axes labeled "고양이 축" "사과 축"
ax = axes[1]
ax.annotate('', xy=(1, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=3))
ax.annotate('', xy=(0, 1), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=RED, lw=3))
ax.text(1.05, 0.02, '고양이\n(idx=12) 축', fontsize=11, color=BLUE, va='center')
ax.text(0.02, 1.05, '사과\n(idx=50) 축', fontsize=11, color=RED)
# right-angle marker
ax.plot([0.08, 0.08, 0], [0, 0.08, 0.08], color='black', lw=1)
ax.text(0.15, 0.15, '90°\n내적=0\n(코사인=0)', fontsize=12, color='black')
ax.set_xlim(-0.15, 1.3)
ax.set_ylim(-0.15, 1.3)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(b) 서로 다른 단어 = 서로 다른 축\n$\\to$ 항상 직교, 거리 정보 없음', fontsize=13)

fig.suptitle('bag-of-words의 두 가지 문제: 희소성과 의미 없는 거리', fontsize=15, color=BLUE, y=1.05)
plt.tight_layout()
plt.savefig('/tmp/claude-1002/-home-smhan/029e91ef-1260-463c-974d-c55f55e786a0/scratchpad/decks/ml1-week14/slides/kor/ml1/week14/figs/ch14_bow_sparsity.png',
            bbox_inches='tight', facecolor='white')
print("done")
